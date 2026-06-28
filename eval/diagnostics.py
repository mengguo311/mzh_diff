"""
eval/diagnostics.py — 补充诊断指标 (roughness / regime / patch-16 PSD / burst)

动机:
    eval/score.py 的 10 个加权指标覆盖了分布矩、|r| 波动聚集 (ACF)、相关性、
    Wasserstein，但**不计算** 外部 PDF 报告里的 roughness_texture / regime_summary /
    volatility_burst 等族。本脚本补齐这一评估闭环，使"降粗糙度 / 改善 regime"的
    实验有可量化、可对比的依据。

设计:
    - 纯 numpy / pandas / scipy，不依赖训练模型 / scaler / config。
    - 数据口径与 eval/score.py.load_and_preprocess_data 完全一致:
        * fake = 宽表 (sp500_i, dgs10_i)，每行一条路径
        * real = 长表 (sp500, DGS10) → 按 fake 的序列长度 L 滑窗 (stride=5)
        * 通道 0 = sp500 日收益率, 通道 1 = dgs10 日差分; 均为真实量级日变化
    - regime 的"高波动阈值"统一取自真实数据，再施加到 fake，保证可比。

用法:
    conda run -n ts_diffusion python eval/diagnostics.py \
        --real /home/u00134/data/train_sp500_us10y.csv \
        --fake output/deep_v9_20k.csv \
        --json eval/diag_v9_20k.json \
        --fig  outputs/figures/diag_v9_20k.png \
        --label v9_20k
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

STRIDE = 5          # 与 score.py / config.STRIDE 对齐
VOL_WINDOW = 21     # regime 滚动波动率窗口 (≈1 个月交易日)


# ──────────────────────────────────────────────
# 数据加载 (镜像 eval/score.py 的口径)
# ──────────────────────────────────────────────
def load_changes(csv_path: str, target_seq_len: int = None) -> np.ndarray:
    """返回 (N, 2, L) 的日变化数组。通道 0=sp500 收益率, 1=dgs10 差分。"""
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()

    sp_cols = sorted(
        [c for c in cols if c.startswith("sp500_") and c.split("_")[-1].isdigit()],
        key=lambda c: int(c.split("_")[-1]),
    )
    dg_cols = sorted(
        [c for c in cols if c.startswith("dgs10_") and c.split("_")[-1].isdigit()],
        key=lambda c: int(c.split("_")[-1]),
    )
    is_wide = len(sp_cols) >= 2 and len(dg_cols) >= 2

    if is_wide:
        sp = df[sp_cols].values.astype(np.float64)
        dg = df[dg_cols].values.astype(np.float64)
        return np.stack([sp, dg], axis=1)  # (N, 2, L)

    # 长表
    col1, col2 = None, None
    for c1, c2 in [("sp500", "DGS10"), ("SP500", "DGS10")]:
        if c1 in df.columns and c2 in df.columns:
            col1, col2 = c1, c2
            break
    if col1 is None:
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c.lower() not in {"day", "index", "date"}]
        col1, col2 = numeric[0], numeric[1]

    d = df[[col1, col2]].copy().ffill().bfill()
    if d[col1].abs().mean() > 1.0:  # level → changes
        d = pd.DataFrame({col1: d[col1].pct_change(), col2: d[col2].diff()}).dropna()
    raw = d.values.astype(np.float64)  # (n_days, 2)

    n = len(raw)
    L = target_seq_len if target_seq_len else n
    if n < L:
        raise ValueError(f"real 数据长度 {n} < 目标窗口 {L}")
    idx = range(0, n - L + 1, STRIDE)
    wins = [raw[s:s + L].T for s in idx]  # 每个 (2, L)
    return np.stack(wins, axis=0)         # (M, 2, L)


# ──────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────
def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """x: (N, L) → (N, L-w+1) 滚动标准差，cumsum 实现 (省内存)。"""
    N, _ = x.shape
    z = np.zeros((N, 1))
    c1 = np.concatenate([z, np.cumsum(x, axis=1)], axis=1)
    c2 = np.concatenate([z, np.cumsum(x * x, axis=1)], axis=1)
    s1 = c1[:, w:] - c1[:, :-w]
    s2 = c2[:, w:] - c2[:, :-w]
    mean = s1 / w
    var = np.maximum(s2 / w - mean ** 2, 0.0)
    return np.sqrt(var)


def lag1_autocorr(x: np.ndarray) -> np.ndarray:
    """x: (N, L) → (N,) 每条序列收益率的 lag-1 自相关 (锯齿/高频噪声指标)。"""
    a = x - x.mean(axis=1, keepdims=True)
    num = (a[:, 1:] * a[:, :-1]).sum(axis=1)
    den = (a * a).sum(axis=1)
    return np.where(den > 0, num / den, 0.0)


def summarize(fake_arr: np.ndarray, real_arr: np.ndarray) -> dict:
    """单个标量指标的 real/fake 对比 (均值、中位、gap、1D-Wasserstein)。"""
    return {
        "real_mean": float(np.mean(real_arr)),
        "real_median": float(np.median(real_arr)),
        "fake_mean": float(np.mean(fake_arr)),
        "fake_median": float(np.median(fake_arr)),
        "gap_mean": float(np.mean(fake_arr) - np.mean(real_arr)),
        "wasserstein": float(wasserstein_distance(fake_arr, real_arr)),
    }


# ──────────────────────────────────────────────
# 诊断族
# ──────────────────────────────────────────────
def roughness_block(r_fake: np.ndarray, r_real: np.ndarray) -> dict:
    """roughness: 总变差 / 二阶差分能量 / 收益率 lag-1 自相关。"""
    def feats(r):
        tv = np.abs(np.diff(r, axis=1)).mean(axis=1)        # 总变差
        d2 = np.diff(r, n=2, axis=1)
        d2_energy = (d2 ** 2).mean(axis=1)                  # 二阶差分能量 (越高越粗糙)
        ret_acf1 = lag1_autocorr(r)                         # 越负 = 越锯齿
        return tv, d2_energy, ret_acf1

    f = feats(r_fake)
    g = feats(r_real)
    keys = ["tv", "d2_energy", "ret_acf1"]
    return {k: summarize(f[i], g[i]) for i, k in enumerate(keys)}


def psd_block(r_fake: np.ndarray, r_real: np.ndarray, L: int, patch: int = 16) -> dict:
    """平均功率谱 + patch 边界伪影检测 (周期=patch 及其谐波处是否有异常尖峰)。"""
    def mean_psd(r):
        R = np.fft.rfft(r - r.mean(axis=1, keepdims=True), axis=1)
        return (np.abs(R) ** 2).mean(axis=0)  # (L//2+1,)

    psd_f = mean_psd(r_fake)
    psd_r = mean_psd(r_real)

    k_patch = max(1, round(L / patch))                       # 周期 patch 对应的 FFT bin
    harmonics = [m * k_patch for m in range(1, 8) if m * k_patch < len(psd_f) - 2]

    def spike_ratio(psd):
        med = float(np.median(psd[1:]))                      # 去 DC 后的整体基线
        if med <= 0:
            return 0.0
        vals = [float(psd[max(1, h - 1):h + 2].max()) for h in harmonics]  # ±1 bin 取峰
        return float(np.mean(vals) / med)

    # 高频功率占比 (频率 > 1/8 周期的能量占比)
    def hf_ratio(psd):
        lo = max(1, len(psd) // 8)
        return float(psd[lo:].sum() / psd[1:].sum())

    return {
        "patch": patch,
        "k_patch": k_patch,
        "harmonics_bins": harmonics,
        "patch_spike_fake": spike_ratio(psd_f),
        "patch_spike_real": spike_ratio(psd_r),
        "patch_spike_excess": (spike_ratio(psd_f) / spike_ratio(psd_r)
                               if spike_ratio(psd_r) > 0 else float("nan")),
        "hf_power_ratio_fake": hf_ratio(psd_f),
        "hf_power_ratio_real": hf_ratio(psd_r),
        # 谱本身存盘，便于画图/复查 (降采样到 512 点以内)
        "_psd_fake": psd_f.tolist(),
        "_psd_real": psd_r.tolist(),
    }


def regime_block(r_fake: np.ndarray, r_real: np.ndarray, vol_w: int) -> tuple[dict, float]:
    """regime: 以真实数据滚动波动率中位数为阈值，比较高波动占比/切换频率/持续长度/burst。"""
    rs_real = rolling_std(r_real, vol_w)
    threshold = float(np.median(rs_real))                    # 阈值统一取自真实

    def feats(r, rs):
        state = (rs > threshold)
        high_frac = state.mean(axis=1)
        flips = np.abs(np.diff(state.astype(np.int8), axis=1)).sum(axis=1)
        switch_rate = flips / state.shape[1]
        mean_run = state.shape[1] / (flips + 1.0)            # 平均状态持续长度
        max_vol = rs.max(axis=1)                             # burst 强度
        vol_of_vol = rs.std(axis=1)                          # 波动的波动
        return high_frac, switch_rate, mean_run, max_vol, vol_of_vol

    rs_fake = rolling_std(r_fake, vol_w)
    f = feats(r_fake, rs_fake)
    g = feats(r_real, rs_real)
    keys = ["high_vol_frac", "switch_rate", "mean_run_len", "max_rolling_vol", "vol_of_vol"]
    block = {k: summarize(f[i], g[i]) for i, k in enumerate(keys)}
    block["_threshold"] = threshold
    return block, threshold


# ──────────────────────────────────────────────
# 画图 (可选，失败不影响 JSON 输出)
# ──────────────────────────────────────────────
def make_figure(report: dict, r_fake: np.ndarray, r_real: np.ndarray, fig_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] 跳过画图 (matplotlib 不可用: {e})")
        return

    psd = report["psd"]
    psd_f = np.array(psd["_psd_fake"])
    psd_r = np.array(psd["_psd_real"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    ax.loglog(np.arange(1, len(psd_r)), psd_r[1:], label="real", alpha=0.8)
    ax.loglog(np.arange(1, len(psd_f)), psd_f[1:], label="fake", alpha=0.8)
    for h in psd["harmonics_bins"]:
        ax.axvline(h, color="red", ls=":", alpha=0.4)
    ax.set_title(f"PSD (sp500 returns)\npatch-{psd['patch']} harmonics = red")
    ax.set_xlabel("freq bin"); ax.set_ylabel("power"); ax.legend()

    ax = axes[1]
    ax.hist(np.diff(r_real, n=2, axis=1).reshape(-1), bins=120, density=True,
            histtype="step", label="real")
    ax.hist(np.diff(r_fake, n=2, axis=1).reshape(-1), bins=120, density=True,
            histtype="step", label="fake")
    ax.set_title("2nd-difference of returns (roughness)")
    ax.set_yscale("log"); ax.legend()

    ax = axes[2]
    rs_r = rolling_std(r_real, VOL_WINDOW).reshape(-1)
    rs_f = rolling_std(r_fake, VOL_WINDOW).reshape(-1)
    ax.hist(rs_r, bins=120, density=True, histtype="step", label="real")
    ax.hist(rs_f, bins=120, density=True, histtype="step", label="fake")
    ax.axvline(report["regime"]["_threshold"], color="k", ls="--", alpha=0.5)
    ax.set_title("rolling-21d vol (regime)")
    ax.set_yscale("log"); ax.legend()

    fig.suptitle(f"diagnostics: {report['label']}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"  [fig] 已保存 {fig_path}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def run(real_csv: str, fake_csv: str, label: str, vol_w: int) -> dict:
    print(f"[diag] 加载 fake: {fake_csv}")
    fake = load_changes(fake_csv)                  # (N, 2, L)
    L = fake.shape[-1]
    print(f"  fake: {fake.shape[0]} 条路径, L={L}")

    print(f"[diag] 加载 real: {real_csv} (窗口 L={L}, stride={STRIDE})")
    real = load_changes(real_csv, target_seq_len=L)
    print(f"  real: {real.shape[0]} 个窗口")

    # 通道 0 = sp500 收益率 (roughness/regime/PSD 主要看它)
    rf, rr = fake[:, 0, :], real[:, 0, :]
    # 通道 1 = dgs10 差分 (roughness 也算一份)
    df_, dr = fake[:, 1, :], real[:, 1, :]

    regime, _ = regime_block(rf, rr, vol_w)
    report = {
        "label": label,
        "L": L,
        "n_fake": int(fake.shape[0]),
        "n_real": int(real.shape[0]),
        "vol_window": vol_w,
        "roughness_sp": roughness_block(rf, rr),
        "roughness_dg": roughness_block(df_, dr),
        "psd": psd_block(rf, rr, L),
        "regime": regime,
    }
    return report, (rf, rr)


def print_summary(report: dict):
    psd = report["psd"]
    print("\n" + "=" * 70)
    print(f"  诊断摘要: {report['label']}  (fake {report['n_fake']} / real {report['n_real']}, L={report['L']})")
    print("=" * 70)
    print("  [Roughness · sp500]            real      fake       gap     Wass")
    for k in ["d2_energy", "tv", "ret_acf1"]:
        s = report["roughness_sp"][k]
        print(f"    {k:14s} {s['real_mean']:10.3e} {s['fake_mean']:10.3e} "
              f"{s['gap_mean']:+9.2e} {s['wasserstein']:8.2e}")
    print(f"\n  [Patch-{psd['patch']} 伪影]  spike  real={psd['patch_spike_real']:.2f}  "
          f"fake={psd['patch_spike_fake']:.2f}  excess={psd['patch_spike_excess']:.2f}  "
          f"(>1.5 提示 patch 边界伪影)")
    print(f"  [高频功率占比]      real={psd['hf_power_ratio_real']:.3f}  "
          f"fake={psd['hf_power_ratio_fake']:.3f}")
    print("\n  [Regime · sp500]               real      fake       gap     Wass")
    for k in ["high_vol_frac", "switch_rate", "mean_run_len", "max_rolling_vol", "vol_of_vol"]:
        s = report["regime"][k]
        print(f"    {k:14s} {s['real_mean']:10.3e} {s['fake_mean']:10.3e} "
              f"{s['gap_mean']:+9.2e} {s['wasserstein']:8.2e}")
    print("=" * 70 + "\n")


def main():
    p = argparse.ArgumentParser(description="补充诊断: roughness / regime / patch-16 PSD")
    p.add_argument("--real", type=str, default="/home/u00134/data/train_sp500_us10y.csv")
    p.add_argument("--fake", type=str, required=True)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--fig", type=str, default=None)
    p.add_argument("--vol-window", type=int, default=VOL_WINDOW)
    args = p.parse_args()

    label = args.label or os.path.splitext(os.path.basename(args.fake))[0]
    report, (rf, rr) = run(args.real, args.fake, label, args.vol_window)
    print_summary(report)

    if args.fig:
        make_figure(report, rf, rr, args.fig)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        # 谱数据体积大，存盘时保留 (便于复查)，但打印时已省略
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  [json] 已保存 {args.json}")


if __name__ == "__main__":
    main()
