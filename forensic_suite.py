#!/usr/bin/env python3
"""
forensic_suite.py — 金融时间序列扩散模型【取证评分套件】(单文件, 自包含编排)

按 `PROJECT_SUMMARY_v13.md` 的评价标准, 把项目散落的诚实评估闸门收敛成【一个入口】:
传入他人的模拟数据 CSV, 即可一键得到 ① 打分 + ② 关键统计量 + ③ 可视化图。

================================ 评价标准(对齐阶段性报告)================================
本套件【不用 score.py 总分 / ddpm_mse】(自指缺陷, 见报告 §3.2), 只用【非自指 data-vs-data】量具:
  1. 记忆化(memorization): 复制率 = 候选与最近邻真实窗【原序列 Pearson>0.95】占比;
     real-vs-real 非重叠底噪 ≈ 0.0% → 任何显著非零都是"背诵训练集"。
  2. 诚实排名(novelty_rerank): 剔除复制样本后, 只在【新颖子集】上重算鉴别器:
       - C2ST_新颖 (↓越接近 real-vs-real 标定越真);
       - SigP_新颖  (↑越接近/超过标定越像真; <0.05=签名被检出为假/塌缩)。
  3. 鉴别器(c2st / signature): 全样本 C2ST 检出 acc、Sig-MMD p 值。
  4. 诊断三族(diagnostics): roughness(d2_energy/tv/ret_acf1)、regime(high_vol_frac/
     switch_rate/mean_run_len/vol_of_vol)、PSD(高频功率占比 / patch-16 接缝 spike)。
  5. 典型事实(stylized facts): 峰度、偏度、|r|-ACF(波动聚集)、杠杆效应、厚尾(QQ)。
  ★ 发散哨兵(本套件相对原工作流的改进): C1 自回归曾出现"被损坏数据骗出漂亮诚实指标"
     (复制率假性极低、签名 p 假性高, 实则 d2_energy 110x / patch-16 spike 516x)。
     故先做数值健全性体检(极端值/毛刺/接缝伪影/std·峰度爆表), 损坏即【判结果不可信】,
     绝不让损坏数据拿到高分。

================================ 输入数据格式 ================================
候选 CSV 支持两种(load_changes 自动识别):
  - 宽表(推荐, 与本项目 output/*.csv 一致): 列 sp500_0..sp500_{L-1}, dgs10_0..dgs10_{L-1};
    每行一条长度 L 的双通道路径(通道0=SP500 日 log 收益, 通道1=DGS10 日差分; 原始量级)。
  - 长表: 列 sp500, DGS10 (日变化); 或价格/利率水平(abs 均值>1 时自动转 pct_change/diff),
    按候选 L 滑窗(stride=5)。
真实基准默认 /home/u00134/data/train_sp500_us10y.csv。所有标定按【候选自身的 L】重算
(报告 §3.8: 短窗更自相似, 标定线随 L 上移, 绝不能跨尺度套用)。

================================ 用法 ================================
  conda run -n ts_diffusion python forensic_suite.py --candidate path/to/their_data.csv
  # 可选: --label NAME --real REAL.csv --out-dir DIR --max-samples 5120 --quick --no-fig
产出: <out-dir>/forensic_<label>.json + 控制台摘要 + <out-dir>/figs/*.png
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurtosis, norm as _norm

# ── 复用项目 eval 量具(它们彼此 import, 只需把 eval/ 加入 path) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_HERE, "eval")
sys.path.insert(0, _EVAL)
import diagnostics as D          # noqa: E402  load_changes/rolling_std/roughness_block/psd_block/regime_block/VOL_WINDOW/STRIDE
import c2st as C                 # noqa: E402  featurize / c2st
import signature as S            # noqa: E402  sig_features / sig_mmd_test
import memorization as M         # noqa: E402  nn_to_bank / nn_real_ref / _std / _pearson

REAL_DEFAULT = "/home/u00134/data/train_sp500_us10y.csv"


# ════════════════════════════════ 典型事实工具 ════════════════════════════════
def _flat_stats(x):
    """x: (N,L) → 该通道的 std / 偏度 / 超额峰度(展平后)。"""
    f = x.ravel()
    return {"std": float(f.std()),
            "skew": float(_skew(f)),
            "kurt": float(_kurtosis(f))}   # Fisher 超额峰度(正态=0)


def abs_acf_curve(x, max_lag):
    """|r| 的自相关曲线(波动聚集): 逐窗算再跨窗平均。x:(N,L) → list[max_lag]。"""
    a = np.abs(x)
    a = a - a.mean(1, keepdims=True)
    var = (a * a).mean(1) + 1e-12
    out = []
    for lag in range(1, max_lag + 1):
        num = (a[:, lag:] * a[:, :-lag]).mean(1)
        out.append(float(np.mean(num / var)))
    return out


def leverage_curve(x, max_lag):
    """杠杆效应 corr(r_t, |r_{t+k}|): 逐窗算再平均。x:(N,L) → list[max_lag]。"""
    out = []
    for lag in range(1, max_lag + 1):
        rt = x[:, :-lag]
        av = np.abs(x[:, lag:])
        rc = rt - rt.mean(1, keepdims=True)
        ac = av - av.mean(1, keepdims=True)
        num = (rc * ac).sum(1)
        den = np.sqrt((rc * rc).sum(1) * (ac * ac).sum(1)) + 1e-12
        out.append(float(np.mean(num / den)))
    return out


def _closeness(a, b):
    """两个标量的接近度 ∈ (0,1], 1=完全一致(按比例)。"""
    a = abs(float(a)) + 1e-12
    b = abs(float(b)) + 1e-12
    return min(a, b) / max(a, b)


# ════════════════════════════════ 主评估流程 ════════════════════════════════
def run_forensic(candidate_csv, real_csv=REAL_DEFAULT, label=None,
                 n_sig=300, k_sub=2, l_sub=None, depth=3, n_perm=300,
                 thresh=0.95, max_samples=None, seed=42, acf_lag=30, lev_lag=10):
    label = label or os.path.splitext(os.path.basename(candidate_csv))[0]

    # ── 加载候选 + 真实(按候选 L 对齐) ──
    fake_all = D.load_changes(candidate_csv)             # (N,2,L) 全量
    L = int(fake_all.shape[-1])
    real = D.load_changes(real_csv, target_seq_len=L)    # (M,2,L)
    n_finite_bad = int((~np.isfinite(fake_all)).sum())
    fake_all = np.nan_to_num(fake_all, nan=0.0, posinf=0.0, neginf=0.0)

    # ★ 逐行发散检测(在【全量】上做, 不被子采样/均值掩盖): 某行任一通道 max|·| 超物理界
    #   bound = max(0.5, 30×真实通道 std) → 捕获 C1 式"少数行灾难性爆值"(如 max|r|=29.9)。
    bound = np.array([max(0.5, 30.0 * float(real[:, 0, :].std())),
                      max(0.5, 30.0 * float(real[:, 1, :].std()))])
    rowmax = np.abs(fake_all).max(axis=2)                # (N,2)
    diverged = (rowmax[:, 0] > bound[0]) | (rowmax[:, 1] > bound[1])
    n_div = int(diverged.sum()); frac_div = float(diverged.mean())
    fake_all = fake_all[~diverged]                       # 剔除发散行后再评估干净剩余
    n_kept = int(len(fake_all))

    if max_samples and len(fake_all) > max_samples:
        rng = np.random.default_rng(seed)
        fake = fake_all[rng.choice(len(fake_all), max_samples, replace=False)]
    else:
        fake = fake_all

    if l_sub is None:
        l_sub = min(200, max(16, L - 1))

    # ── 特征空间 + 标定标尺(全部按本 L 重算) ──
    vol_thr = float(np.median(D.rolling_std(real[:, 0, :], D.VOL_WINDOW)))
    scale = np.array([real[:, 0, :].std(), real[:, 1, :].std()]) + 1e-12
    Xreal = C.featurize(real, vol_thr)
    mu, sd = Xreal.mean(0), Xreal.std(0) + 1e-12
    Rz = M._std(Xreal, mu, sd)

    guard = int(np.ceil(L / D.STRIDE))
    ref, ref_idx = M.nn_real_ref(Rz, guard)
    ref_med = float(np.median(ref))
    ref_rawp = np.array([np.mean([M._pearson(real[i, ch], real[ref_idx[i], ch]) for ch in range(2)])
                         for i in range(len(real))])
    copy_floor = float((ref_rawp > thresh).mean())

    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(len(Xreal)); half = len(perm) // 2
    c2_cal = C.c2st(Xreal[perm[:half]], Xreal[perm[half:]])["test_acc"]
    A_real = S.sig_features(real, scale, n_sig, k_sub, l_sub, depth, seed=1)
    A_real2 = S.sig_features(real, scale, n_sig, k_sub, l_sub, depth, seed=2)
    sig_cal = S.sig_mmd_test(A_real, A_real2, n_perm=n_perm, seed=10)["p_value"]

    # ── 1. 记忆化(复制率) ──
    Ff = C.featurize(fake, vol_thr)
    Fz = M._std(Ff, mu, sd)
    dmin, nn_idx = M.nn_to_bank(Fz, Rz)
    raw_p = np.array([np.mean([M._pearson(fake[j, ch], real[nn_idx[j], ch]) for ch in range(2)])
                      for j in range(len(fake))])
    copy_rate = float((raw_p > thresh).mean())
    copy99 = float((raw_p > 0.99).mean())
    dNN_med = float(np.median(dmin)); ratio = dNN_med / (ref_med + 1e-12)

    # ── 2. 诚实排名(剔除复制后只看新颖子集) ──
    mask = raw_p > thresh
    novel = fake[~mask]; n_nov = int((~mask).sum())
    c2_all = C.c2st(Xreal, Ff)["test_acc"]
    c2_nov = C.c2st(Xreal, C.featurize(novel, vol_thr))["test_acc"] if n_nov >= 20 else c2_all
    B_all = S.sig_features(fake, scale, n_sig, k_sub, l_sub, depth, seed=3)
    p_all = S.sig_mmd_test(A_real, B_all, n_perm=n_perm, seed=11)["p_value"]
    if n_nov >= 50:
        B_nov = S.sig_features(novel, scale, min(n_sig, n_nov), k_sub, l_sub, depth, seed=3)
        p_nov = S.sig_mmd_test(A_real, B_nov, n_perm=n_perm, seed=11)["p_value"]
    else:
        p_nov = p_all

    # ── 4. 诊断三族 ──
    rf, rr = fake[:, 0, :], real[:, 0, :]
    dff, dr = fake[:, 1, :], real[:, 1, :]
    rough_sp = D.roughness_block(rf, rr)
    rough_dg = D.roughness_block(dff, dr)
    psd = D.psd_block(rf, rr, L)
    regime, _ = D.regime_block(rf, rr, D.VOL_WINDOW)

    # ── 5. 典型事实 ──
    sp_fake, sp_real = _flat_stats(rf), _flat_stats(rr)
    dg_fake, dg_real = _flat_stats(dff), _flat_stats(dr)
    acf_fake = abs_acf_curve(rf, acf_lag); acf_real = abs_acf_curve(rr, acf_lag)
    lev_fake = leverage_curve(rf, lev_lag); lev_real = leverage_curve(rr, lev_lag)

    # ── ★ 发散/损坏哨兵(C1 教训: 防被部分损坏数据骗分) ──
    #   guard_flags = 系统性损坏(致命→拒绝打分); notes = 已自动处理的问题(剔除发散行等)。
    d2_ratio = rough_sp["d2_energy"]["fake_mean"] / (rough_sp["d2_energy"]["real_mean"] + 1e-12)
    patch_excess = float(psd["patch_spike_excess"])
    std_ratio = sp_fake["std"] / (sp_real["std"] + 1e-12)
    guard_flags, notes = [], []
    if n_finite_bad > 0:
        notes.append(f"修复 {n_finite_bad} 个非有限值(NaN/Inf→0)")
    if n_div > 0:
        notes.append(f"剔除 {n_div} 条发散行({frac_div:.2%}; 任一通道 max|·| 超物理界), 余 {n_kept} 条参评")
    if frac_div > 0.05:        # 发散行过多 = 系统性发散
        guard_flags.append(f"发散行占比 {frac_div:.1%}(>5%, 系统性发散)")
    if d2_ratio > 5:           # 剔除发散行后仍整体过毛刺
        guard_flags.append(f"d2_energy 为真实 {d2_ratio:.0f}x(高频毛刺)")
    if not np.isnan(patch_excess) and patch_excess > 10:
        guard_flags.append(f"patch-16 接缝 spike 为真实 {patch_excess:.0f}x(拼接伪影)")
    if std_ratio > 3 or std_ratio < 0.2:
        guard_flags.append(f"std 为真实 {std_ratio:.1f}x(尺度异常)")
    if sp_fake["kurt"] > 100:
        guard_flags.append(f"峰度 {sp_fake['kurt']:.0f} 爆表")
    corrupted = len(guard_flags) > 0

    # ── 判定(逐项 + 综合) ──
    def verdict_mem(cr):
        if cr <= copy_floor + 0.02: return "PASS"
        if cr <= 0.20: return "WARN"
        return "FAIL"

    def verdict_c2(cn):
        if cn <= c2_cal + 0.10: return "PASS"
        if cn <= 0.75: return "WARN"
        return "FAIL"

    def verdict_sig(pn):
        if pn >= max(0.05, 0.5 * sig_cal): return "PASS"
        if pn >= 0.05: return "WARN"
        return "FAIL"   # <0.05 = 签名塌/被检出

    v_mem, v_c2, v_sig = verdict_mem(copy_rate), verdict_c2(c2_nov), verdict_sig(p_nov)

    # ── 综合"诚实真实度评分"(0-100, 启发式; 损坏则不打分) ──
    if corrupted:
        composite, comps = None, None
    else:
        s_mem = max(0.0, 1.0 - min(copy_rate, 0.5) / 0.5)                       # 复制越少越高
        s_c2 = max(0.0, 1.0 - (c2_nov - c2_cal) / max(1.0 - c2_cal, 1e-6))      # C2ST_新颖越接近标定越高
        s_c2 = float(np.clip(s_c2, 0, 1))
        s_sig = float(np.clip(p_nov / max(sig_cal, 1e-6), 0, 1))                # SigP_新颖入标定带→1
        s_sty = float(np.mean([_closeness(sp_fake["kurt"], sp_real["kurt"]),
                               _closeness(acf_fake[0], acf_real[0]),
                               _closeness(regime["high_vol_frac"]["fake_mean"],
                                          regime["high_vol_frac"]["real_mean"]),
                               _closeness(rough_sp["d2_energy"]["fake_mean"],
                                          rough_sp["d2_energy"]["real_mean"])]))
        comps = {"novelty_c2st": round(s_c2, 3), "novelty_sig": round(s_sig, 3),
                 "stylized": round(s_sty, 3), "anti_memorization": round(s_mem, 3)}
        composite = float(100.0 * (0.30 * s_c2 + 0.25 * s_sig + 0.30 * s_sty + 0.15 * s_mem))

    report = {
        "label": label, "candidate": candidate_csv, "real": real_csv,
        "L": L, "n_candidate": int(len(fake)), "n_real": int(len(real)),
        "calibration_L_specific": {
            "copy_floor_p95": copy_floor, "c2st_real_vs_real": c2_cal,
            "sig_p_real_vs_real": sig_cal, "real_ref_dNN_median": ref_med},
        "memorization": {
            "copy_rate_p95": copy_rate, "copy_rate_p99": copy99,
            "raw_pearson_mean": float(raw_p.mean()),
            "dNN_median": dNN_med, "ratio_vs_ref": ratio},
        "novelty_rerank": {
            "n_novel": n_nov, "copy_rate": copy_rate,
            "c2st_all": c2_all, "c2st_novel": c2_nov,
            "sig_p_all": p_all, "sig_p_novel": p_nov},
        "diagnostics": {
            "roughness_sp": rough_sp, "roughness_dg": rough_dg,
            "regime": {k: regime[k] for k in regime if not k.startswith("_")},
            "psd": {k: psd[k] for k in psd if not k.startswith("_")}},
        "stylized_facts": {
            "sp500": {"real": sp_real, "fake": sp_fake},
            "dgs10": {"real": dg_real, "fake": dg_fake},
            "abs_acf_real": acf_real, "abs_acf_fake": acf_fake,
            "leverage_real": lev_real, "leverage_fake": lev_fake},
        "divergence_guard": {
            "corrupted": corrupted, "flags": guard_flags, "notes": notes,
            "n_diverged_rows": n_div, "frac_diverged": frac_div, "n_kept": n_kept,
            "d2_energy_ratio": float(d2_ratio), "patch_spike_excess": patch_excess,
            "std_ratio": float(std_ratio), "kurt_fake": sp_fake["kurt"]},
        "verdict": {"memorization": v_mem, "novelty_c2st": v_c2, "novelty_signature": v_sig,
                    "composite_score": composite, "score_components": comps},
    }
    # 画图需要的原始数组(不入 JSON)
    arrays = {"rf": rf, "rr": rr, "dff": dff, "dr": dr,
              "psd_f": np.array(psd["_psd_fake"]), "psd_r": np.array(psd["_psd_real"]),
              "harmonics": psd["harmonics_bins"], "acf_lag": acf_lag, "lev_lag": lev_lag}
    return report, arrays


# ════════════════════════════════ 可视化 ════════════════════════════════
def make_figures(report, arrays, fig_dir, seed=42):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] 跳过画图(matplotlib 不可用: {e})")
        return []
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    rng = np.random.default_rng(seed)
    rf, rr = arrays["rf"], arrays["rr"]
    label = report["label"]
    cal = report["calibration_L_specific"]; nr = report["novelty_rerank"]; mem = report["memorization"]
    GREEN, RED, GREY = "#2ca02c", "#d62728", "0.5"

    def _sub(x, n=3000):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    # ── 图1: 取证判定仪表板 ──
    try:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        v = report["verdict"]; g = report["divergence_guard"]
        vc = {"PASS": GREEN, "WARN": "#ff7f0e", "FAIL": RED}
        # (a) 复制率 vs 底噪
        ax[0].bar(["copy floor\n(real-real)", f"candidate\n{label}"],
                  [cal["copy_floor_p95"] * 100, mem["copy_rate_p95"] * 100],
                  color=[GREY, vc[v["memorization"]]])
        ax[0].set_title("Memorization: copy rate (Pearson>0.95, %)")
        ax[0].set_ylabel("copy rate (%)")
        for i, val in enumerate([cal["copy_floor_p95"] * 100, mem["copy_rate_p95"] * 100]):
            ax[0].text(i, val, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax[0].annotate(f"verdict: {v['memorization']}", (0.5, 0.92), xycoords="axes fraction",
                       ha="center", color=vc[v["memorization"]], fontweight="bold")
        # (b) C2ST_novel vs 标定
        ax[1].bar(["calib\n(real-real)", "C2ST_all", "C2ST_novel"],
                  [cal["c2st_real_vs_real"], nr["c2st_all"], nr["c2st_novel"]],
                  color=[GREY, "#1f77b4", vc[v["novelty_c2st"]]])
        ax[1].axhline(0.5, color="k", ls=":", alpha=0.4)
        ax[1].set_title("Novelty C2ST (lower=more real, 0.5=ideal)")
        ax[1].set_ylim(0.4, 1.0)
        for i, val in enumerate([cal["c2st_real_vs_real"], nr["c2st_all"], nr["c2st_novel"]]):
            ax[1].text(i, val, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
        # (c) SigP_novel vs 标定
        ax[2].bar(["calib\n(real-real)", "SigP_all", "SigP_novel"],
                  [cal["sig_p_real_vs_real"], nr["sig_p_all"], nr["sig_p_novel"]],
                  color=[GREY, "#1f77b4", vc[v["novelty_signature"]]])
        ax[2].axhline(0.05, color=RED, ls=":", alpha=0.6, label="p=0.05 (detected)")
        ax[2].set_title("Novelty Sig-MMD p (higher=more real)")
        ax[2].legend(fontsize=8)
        for i, val in enumerate([cal["sig_p_real_vs_real"], nr["sig_p_all"], nr["sig_p_novel"]]):
            ax[2].text(i, val, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
        title = f"Forensic verdict dashboard — {label}   (L={report['L']}, N={report['n_candidate']})"
        if g["corrupted"]:
            title += "\n[!!] DATA CORRUPTED / DIVERGED — metrics UNTRUSTWORTHY: " + "; ".join(g["flags"])
            fig.patch.set_facecolor("#ffecec")
        elif v["composite_score"] is not None:
            title += f"   composite honest-realism score = {v['composite_score']:.1f}/100"
            if g.get("n_diverged_rows", 0) > 0:
                title += f"   [dropped {g['n_diverged_rows']} diverged rows]"
        fig.suptitle(title, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = os.path.join(fig_dir, "01_verdict_dashboard.png"); fig.savefig(p, dpi=130); plt.close(fig)
        paths.append(p)
    except Exception as e:
        print(f"  [fig] 图1 失败: {e}")

    # ── 图2: 典型事实 ──
    try:
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        fr, frr = _sub(rf.ravel()), _sub(rr.ravel())
        sf = report["stylized_facts"]["sp500"]
        # (a) 分布
        lo, hi = np.percentile(frr, [0.5, 99.5])
        bins = np.linspace(lo * 1.5, hi * 1.5, 120)
        ax[0, 0].hist(frr, bins=bins, density=True, histtype="step", color="k", lw=2, label="real")
        ax[0, 0].hist(np.clip(fr, bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                      color="#1f77b4", lw=1.5, label=label)
        ax[0, 0].set_title("SP500 daily return distribution")
        ax[0, 0].annotate(f"kurtosis  real={sf['real']['kurt']:.1f}  fake={sf['fake']['kurt']:.1f}\n"
                          f"std  real={sf['real']['std']:.4f}  fake={sf['fake']['std']:.4f}",
                          (0.5, 0.82), xycoords="axes fraction", ha="center", fontsize=9,
                          bbox=dict(boxstyle="round", fc="w", alpha=0.7))
        ax[0, 0].legend()
        # (b) 尾部(semilogy)
        ax[0, 1].hist(frr, bins=bins, density=True, histtype="step", color="k", lw=2, label="real")
        ax[0, 1].hist(np.clip(fr, bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                      color="#1f77b4", lw=1.5, label=label)
        ax[0, 1].set_yscale("log"); ax[0, 1].set_title("Tails (semilog-y) — heavy-tail check"); ax[0, 1].legend()
        # (c) |r|-ACF 波动聚集
        lags = np.arange(1, arrays["acf_lag"] + 1)
        ax[1, 0].plot(lags, report["stylized_facts"]["abs_acf_real"], "k-o", ms=3, label="real")
        ax[1, 0].plot(lags, report["stylized_facts"]["abs_acf_fake"], "-s", ms=3,
                      color="#1f77b4", label=label)
        ax[1, 0].axhline(0, color=GREY, lw=0.8)
        ax[1, 0].set_title("Volatility clustering: ACF of |return|")
        ax[1, 0].set_xlabel("lag"); ax[1, 0].set_ylabel("ACF(|r|)")
        ax[1, 0].annotate(f"ACF(1)  real={report['stylized_facts']['abs_acf_real'][0]:.3f}  "
                          f"fake={report['stylized_facts']['abs_acf_fake'][0]:.3f}",
                          (0.5, 0.9), xycoords="axes fraction", ha="center", fontsize=9)
        ax[1, 0].legend()
        # (d) QQ
        srt_r = np.sort((frr - frr.mean()) / (frr.std() + 1e-12))
        srt_f = np.sort((fr - fr.mean()) / (fr.std() + 1e-12))
        q = np.linspace(0.001, 0.999, min(len(srt_r), len(srt_f), 2000))
        ax[1, 1].plot(np.quantile(srt_r, q), np.quantile(srt_r, q), color=GREY, ls="--", label="y=x (real ref)")
        ax[1, 1].plot(np.quantile(srt_r, q), np.quantile(srt_f, q), color="#1f77b4", label=label)
        ax[1, 1].set_title("QQ: candidate vs real quantiles")
        ax[1, 1].set_xlabel("real quantile (z)"); ax[1, 1].set_ylabel("candidate quantile (z)"); ax[1, 1].legend()
        fig.suptitle(f"Stylized facts — {label}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p = os.path.join(fig_dir, "02_stylized_facts.png"); fig.savefig(p, dpi=130); plt.close(fig)
        paths.append(p)
    except Exception as e:
        print(f"  [fig] 图2 失败: {e}")

    # ── 图3: 诊断三族 + 发散哨兵 ──
    try:
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        reg = report["diagnostics"]["regime"]; rgh = report["diagnostics"]["roughness_sp"]
        # (a) regime 高波动占比 + 持续长度
        labels_r = ["high_vol_frac", "mean_run_len"]
        realv = [reg["high_vol_frac"]["real_mean"], reg["mean_run_len"]["real_mean"]]
        fakev = [reg["high_vol_frac"]["fake_mean"], reg["mean_run_len"]["fake_mean"]]
        x = np.arange(2); w = 0.35
        ax[0, 0].bar(x - w / 2, realv, w, label="real", color="k")
        ax[0, 0].bar(x + w / 2, fakev, w, label=label, color="#1f77b4")
        ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(labels_r)
        ax[0, 0].set_title("Regime: high-vol fraction & run length")
        for i in range(2):
            ax[0, 0].text(i - w / 2, realv[i], f"{realv[i]:.2f}", ha="center", va="bottom", fontsize=8)
            ax[0, 0].text(i + w / 2, fakev[i], f"{fakev[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax[0, 0].legend()
        # (b) roughness d2_energy / tv
        keys = ["d2_energy", "tv", "ret_acf1"]
        realr = [rgh[k]["real_mean"] for k in keys]; faker = [rgh[k]["fake_mean"] for k in keys]
        x = np.arange(3)
        ax[0, 1].bar(x - w / 2, realr, w, label="real", color="k")
        ax[0, 1].bar(x + w / 2, faker, w, label=label, color="#1f77b4")
        ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(keys)
        ax[0, 1].set_title("Roughness (d2_energy=smoothness probe)")
        ax[0, 1].set_yscale("symlog", linthresh=1e-4); ax[0, 1].legend()
        # (c) PSD loglog + patch-16 谐波
        psd_f, psd_r = arrays["psd_f"], arrays["psd_r"]
        ax[1, 0].loglog(np.arange(1, len(psd_r)), psd_r[1:], color="k", label="real", alpha=0.85)
        ax[1, 0].loglog(np.arange(1, len(psd_f)), psd_f[1:], color="#1f77b4", label=label, alpha=0.85)
        for h in arrays["harmonics"]:
            ax[1, 0].axvline(h, color=RED, ls=":", alpha=0.35)
        ax[1, 0].set_title(f"PSD (red=patch-16 harmonics)  patch_spike_excess="
                           f"{report['divergence_guard']['patch_spike_excess']:.1f}x")
        ax[1, 0].set_xlabel("freq bin"); ax[1, 0].set_ylabel("power"); ax[1, 0].legend()
        # (d) 杠杆 + 哨兵文本
        lags = np.arange(1, arrays["lev_lag"] + 1)
        ax[1, 1].plot(lags, report["stylized_facts"]["leverage_real"], "k-o", ms=3, label="real")
        ax[1, 1].plot(lags, report["stylized_facts"]["leverage_fake"], "-s", ms=3,
                      color="#1f77b4", label=label)
        ax[1, 1].axhline(0, color=GREY, lw=0.8)
        ax[1, 1].set_title("Leverage effect corr(r_t, |r_{t+k}|)")
        ax[1, 1].set_xlabel("lag k"); ax[1, 1].legend()
        g = report["divergence_guard"]
        txt = "DIVERGENCE GUARD: " + ("CORRUPTED\n" + "\n".join(g["flags"]) if g["corrupted"] else "OK")
        ax[1, 1].annotate(txt, (0.5, 0.05), xycoords="axes fraction", ha="center", va="bottom",
                          fontsize=8, color=(RED if g["corrupted"] else GREEN),
                          bbox=dict(boxstyle="round", fc="w", alpha=0.7))
        fig.suptitle(f"Diagnostics (roughness/regime/PSD) + divergence guard — {label}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p = os.path.join(fig_dir, "03_diagnostics.png"); fig.savefig(p, dpi=130); plt.close(fig)
        paths.append(p)
    except Exception as e:
        print(f"  [fig] 图3 失败: {e}")

    return paths


# ════════════════════════════════ 控制台摘要 ════════════════════════════════
def print_summary(report):
    r = report; v = r["verdict"]; g = r["divergence_guard"]
    cal = r["calibration_L_specific"]; nr = r["novelty_rerank"]; mem = r["memorization"]
    sp = r["stylized_facts"]["sp500"]; reg = r["diagnostics"]["regime"]; rgh = r["diagnostics"]["roughness_sp"]
    line = "=" * 78
    print("\n" + line)
    print(f"  取证评分摘要: {r['label']}   (L={r['L']}, 候选 {r['n_candidate']} 条 / 真实 {r['n_real']} 窗)")
    print(line)
    if g.get("notes"):
        print("  [i] 数据预处理: " + "; ".join(g["notes"]))
    if g["corrupted"]:
        print("  [!!] 发散哨兵: 候选数据【系统性损坏】—— 以下诚实指标不可信, 已拒绝打分!")
        for f in g["flags"]:
            print(f"        · {f}")
        print("  " + "-" * 74)
    print(f"  标定标尺(本 L={r['L']} 重算): 复制率底噪={cal['copy_floor_p95']:.1%}  "
          f"C2ST_real-real={cal['c2st_real_vs_real']:.3f}  SigP_real-real={cal['sig_p_real_vs_real']:.3f}")
    print("  " + "-" * 74)
    print(f"  {'判据':<22}{'读数':>14}{'标定/目标':>16}{'判定':>8}")
    print(f"  {'① 复制率(记忆化)':<20}{mem['copy_rate_p95']*100:>12.1f}%{'底噪 '+format(cal['copy_floor_p95']*100,'.1f')+'%':>16}{v['memorization']:>8}")
    print(f"  {'② C2ST_新颖(↓真)':<20}{nr['c2st_novel']:>13.3f}{'≈'+format(cal['c2st_real_vs_real'],'.3f'):>16}{v['novelty_c2st']:>8}")
    print(f"  {'③ SigP_新颖(↑真)':<20}{nr['sig_p_novel']:>13.3f}{'≥'+format(cal['sig_p_real_vs_real'],'.3f'):>16}{v['novelty_signature']:>8}")
    print("  " + "-" * 74)
    print(f"  典型事实  峰度 real={sp['real']['kurt']:.1f} / fake={sp['fake']['kurt']:.1f}   "
          f"|r|-ACF1 real={r['stylized_facts']['abs_acf_real'][0]:.3f} / fake={r['stylized_facts']['abs_acf_fake'][0]:.3f}")
    print(f"  诊断      high_vol_frac real={reg['high_vol_frac']['real_mean']:.2f} / fake={reg['high_vol_frac']['fake_mean']:.2f}   "
          f"d2_energy fake/real={g['d2_energy_ratio']:.2f}x   patch_spike={g['patch_spike_excess']:.1f}x")
    print("  " + "-" * 74)
    if v["composite_score"] is None:
        print("  综合诚实真实度评分: N/A (数据损坏, 拒绝打分)")
    else:
        c = v["score_components"]
        print(f"  综合诚实真实度评分 = {v['composite_score']:.1f}/100   "
              f"(新颖C2ST {c['novelty_c2st']} · 新颖签名 {c['novelty_sig']} · 典型事实 {c['stylized']} · 抗记忆 {c['anti_memorization']})")
    print(f"  判读: PASS/WARN/FAIL 对标 real-vs-real 标尺; 评分为启发式综合, 主依据是逐项判定。")
    print(line + "\n")


def main():
    ap = argparse.ArgumentParser(description="金融扩散模型取证评分套件(单文件)")
    ap.add_argument("--candidate", required=True, help="他人模拟数据 CSV(宽表 sp500_i/dgs10_i 或长表)")
    ap.add_argument("--real", default=REAL_DEFAULT, help="真实基准 CSV")
    ap.add_argument("--label", default=None, help="候选标签(默认取文件名)")
    ap.add_argument("--out-dir", default=None, help="输出目录(默认 eval/forensic_out/<label>)")
    ap.add_argument("--max-samples", type=int, default=None, help="抽样候选条数上限(控时)")
    ap.add_argument("--quick", action="store_true", help="快速模式(减少签名/置换次数)")
    ap.add_argument("--no-fig", action="store_true", help="不出图")
    ap.add_argument("--thresh", type=float, default=0.95, help="复制判定 Pearson 阈值")
    args = ap.parse_args()

    label = args.label or os.path.splitext(os.path.basename(args.candidate))[0]
    out_dir = args.out_dir or os.path.join(_HERE, "eval", "forensic_out", label)
    os.makedirs(out_dir, exist_ok=True)
    n_sig, n_perm = (150, 150) if args.quick else (300, 300)

    print(f"[forensic] 候选={args.candidate}\n[forensic] 真实={args.real}\n[forensic] 输出={out_dir}")
    report, arrays = run_forensic(args.candidate, args.real, label=label,
                                  n_sig=n_sig, n_perm=n_perm,
                                  max_samples=args.max_samples, thresh=args.thresh)
    print_summary(report)

    figs = []
    if not args.no_fig:
        figs = make_figures(report, arrays, os.path.join(out_dir, "figs"))
        for p in figs:
            print(f"  [fig] {p}")
    report["figures"] = figs

    json_path = os.path.join(out_dir, f"forensic_{label}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [json] {json_path}")


if __name__ == "__main__":
    main()
