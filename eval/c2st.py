"""
eval/c2st.py — Classifier Two-Sample Test (feature-space) 假数据鉴别器

非自指 (data-vs-data): 在 ~24 维 stylized-fact 特征空间训练分类器区分 real vs candidate。
  - test accuracy / AUC 越接近 0.5 → 越不可区分 → candidate 越真实 (生成目标);
  - 越接近 1.0 → 越易被识破为假 (鉴别目标)。
不碰原始 4096 维 (独立窗口太少会过拟合饱和)。

标定: 同源对半 (real-vs-real / fake-vs-fake) 应给 acc≈0.5。
近似 p 值用 C2ST 高斯零分布 N(0.5, 1/(4 n_test)) (Lopez-Paz & Oquab, ICLR 2017);
注意 real 窗口 stride=5 重叠 → 有效独立样本偏少, p 值偏乐观, 故以"相对排序 + 同源标定"为主。

用法:
  conda run -n ts_diffusion python eval/c2st.py \
    --real /home/u00134/data/train_sp500_us10y.csv \
    --fakes v9=output/deep_v9_20k.csv v10_sampling=output/deep_v10.csv v10_retrained=output/deep_v10_retrained.csv \
    --json eval/c2st_results.json
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import skew, kurtosis, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnostics import load_changes, rolling_std, lag1_autocorr, VOL_WINDOW  # noqa: E402

ACF_LAG = 5


def _abs_acf(x, max_lag):
    a = np.abs(x); a = a - a.mean()
    var = float((a * a).mean()) + 1e-12
    return [float((a[lag:] * a[:-lag]).mean() / var) for lag in range(1, max_lag + 1)]


def _cross_row(sp, d10, d2):
    """10 维跨通道/收益率曲线取证特征 (v14-fusion, productionized A4)。
    通道约定 sp500 / DGS10_diff / DGS2_diff。这些是【单通道边际看不到】的取证维度:
    破坏跨通道结构而保各通道边际的 fake, marginal AUC≈0.5 / 加这些后→1.0 (A4 受控演示)。"""
    def cc(a, b):
        if len(a) < 4:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else 0.0
    c_sp10, c_sp2, c_102 = cc(sp, d10), cc(sp, d2), cc(d10, d2)        # 股债 / 股-2Y / 利率联动
    thr = sp.mean() - 1.5 * sp.std(); mk = sp < thr                   # 危机期(sp 跌穿 -1.5σ)
    t_sp10 = cc(sp[mk], d10[mk]) if mk.sum() > 3 else 0.0             # 尾部股债避险协动
    t_102 = cc(d10[mk], d2[mk]) if mk.sum() > 3 else 0.0             # 尾部曲线协动
    slope = d10 - d2                                                  # 2s10s 斜率变化(无需 level 锚)
    s_std = float(slope.std()); s_kur = float(np.clip(kurtosis(slope), -20, 200))
    sign_ag = float(np.mean(np.sign(d10) == np.sign(d2)))            # 曲线平行移动一致性/无套利
    ll_p = cc(d10[1:], d2[:-1]); ll_m = cc(d10[:-1], d2[1:])         # lead-lag(±1)
    return [c_sp10, c_sp2, c_102, t_sp10, t_102, s_std, s_kur, sign_ag, ll_p, ll_m]


def featurize(windows: np.ndarray, vol_threshold: float, vol_w: int = VOL_WINDOW,
              dgs10_fp: bool = False, cross_channel: bool = False) -> np.ndarray:
    """windows: (N,C,L) -> (N, D) stylized-fact 特征矩阵。通道0=sp500, 1=dgs10(, 2=dgs2)。
    dgs10_fp=True 时额外追加 3 维 DGS10 真实量化指纹(rounding_0.01/zero_diff/unique_ratio):
    真实 DGS10 差分量化到 0.01(on_grid≈1.0), 扩散生成连续浮点(≈0) → 鉴别器近完美抓手。
    cross_channel=True 且 C≥3 时追加 10 维跨通道/收益率曲线取证特征(_cross_row, v14-fusion)。
    ⚠️ 默认 False: 诚实闸门(memorization/novelty_rerank/forensic_suite)用默认口径不受影响;
    仅 eval/forensic_auc.py / forensic_cross.py 等鉴别器路径显式开启(否则会把所有扩散模型 C2ST
    压到 ≈1.0, 丢失模型间排序; 且每加一维就移动 go/no-go 球门)。"""
    sp, dg = windows[:, 0, :], windows[:, 1, :]
    N = windows.shape[0]
    has_cross = cross_channel and windows.shape[1] >= 3
    d2_all = windows[:, 2, :] if has_cross else None

    # 向量化的 roughness / regime (基于 sp 通道)
    rs = rolling_std(sp, vol_w)                                   # (N, L-w+1)
    racf1 = lag1_autocorr(sp)
    d2 = np.diff(sp, n=2, axis=1); d2e = (d2 * d2).mean(1)
    tv = np.abs(np.diff(sp, axis=1)).mean(1)
    state = rs > vol_threshold
    high = state.mean(1)
    flips = np.abs(np.diff(state.astype(np.int8), axis=1)).sum(1)
    switch = flips / state.shape[1]
    runlen = state.shape[1] / (flips + 1.0)
    maxv = rs.max(1); volvol = rs.std(1)

    feats = []
    for i in range(N):
        s, d = sp[i], dg[i]
        sk_s = np.clip(skew(s), -20, 20); ku_s = np.clip(kurtosis(s), -20, 200)
        sk_d = np.clip(skew(d), -20, 20); ku_d = np.clip(kurtosis(d), -20, 200)
        c = np.corrcoef(s, d)[0, 1]
        thr = s.mean() - 1.5 * s.std(); m = s < thr
        tc = np.corrcoef(s[m], d[m])[0, 1] if m.sum() > 3 else 0.0
        row = [sk_s, ku_s, sk_d, ku_d,
               c if np.isfinite(c) else 0.0,
               tc if np.isfinite(tc) else 0.0]
        row += _abs_acf(s, ACF_LAG) + _abs_acf(d, ACF_LAG)
        row += [float(d2e[i]), float(tv[i]), float(racf1[i]),
                float(high[i]), float(switch[i]), float(runlen[i]),
                float(maxv[i]), float(volvol[i])]
        if dgs10_fp:                                              # DGS10 真实量化指纹(真实量级 d)
            row += [float(np.mean(np.abs(d * 100.0 - np.round(d * 100.0)) < 1e-6)),  # on 0.01 grid(容float误差)
                    float(np.mean(np.abs(d) < 1e-12)),                                # zero diff
                    float(len(np.unique(np.round(d, 8))) / len(d))]                   # unique ratio
        if has_cross:                                             # 跨通道/收益率曲线特征(C≥3, v14-fusion)
            row += _cross_row(s, d, d2_all[i])
        feats.append(row)
    X = np.array(feats, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def c2st(Xa: np.ndarray, Xb: np.ndarray, seed: int = 42, balance: bool = True):
    """Xa(label0) vs Xb(label1) 的 C2ST。返回 test_acc, auc, n_test, approx_p。"""
    rng = np.random.default_rng(seed)
    if balance:
        n = min(len(Xa), len(Xb))
        Xa = Xa[rng.choice(len(Xa), n, replace=False)]
        Xb = Xb[rng.choice(len(Xb), n, replace=False)]
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    try:
        auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    except Exception:
        auc = float("nan")
    n_test = len(yte)
    # 高斯零分布近似 p (单侧): acc ~ N(0.5, 1/(4 n_test))
    z = (acc - 0.5) / np.sqrt(1.0 / (4 * n_test))
    p = float(1.0 - norm.cdf(z))
    return {"test_acc": acc, "auc": auc, "n_test": int(n_test), "approx_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="/home/u00134/data/train_sp500_us10y.csv")
    ap.add_argument("--fakes", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--json", default=None)
    ap.add_argument("--vol-window", type=int, default=VOL_WINDOW)
    args = ap.parse_args()

    # 用第一个 fake 的 L 决定 real 窗口长度
    first_path = args.fakes[0].split("=", 1)[1]
    L = load_changes(first_path).shape[-1]
    print(f"[c2st] 窗口长度 L={L}")

    real = load_changes(args.real, target_seq_len=L)
    vol_thr = float(np.median(rolling_std(real[:, 0, :], args.vol_window)))
    print(f"[c2st] real {real.shape[0]} 窗口; 全局高波动阈值={vol_thr:.5f}")
    Xreal = featurize(real, vol_thr, args.vol_window)

    results = {"L": L, "n_real": int(real.shape[0]), "features": Xreal.shape[1], "calibration": {}, "detect": {}}

    # ── 标定: 同源对半 (期望 acc≈0.5) ──
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xreal)); half = len(perm) // 2
    results["calibration"]["real_vs_real"] = c2st(Xreal[perm[:half]], Xreal[perm[half:]])

    # ── real vs 各 candidate ──
    rows = []
    for spec in args.fakes:
        label, path = spec.split("=", 1)
        fake = load_changes(path)
        Xfake = featurize(fake, vol_thr, args.vol_window)
        r = c2st(Xreal, Xfake)
        # fake-vs-fake 标定 (该源自身对半)
        pf = rng.permutation(len(Xfake)); hf = len(pf) // 2
        r["fake_vs_fake_calib_acc"] = c2st(Xfake[pf[:hf]], Xfake[pf[hf:]])["test_acc"]
        results["detect"][label] = r
        rows.append((label, r))

    # ── 打印 ──
    cal = results["calibration"]["real_vs_real"]
    print(f"\n标定 real-vs-real: acc={cal['test_acc']:.3f} auc={cal['auc']:.3f} "
          f"p={cal['approx_p']:.3f}  (应≈0.5 / p大)")
    print(f"\n{'candidate':16s} {'C2ST_acc':>9s} {'AUC':>7s} {'approx_p':>9s} {'ff_calib':>9s}   越低越真实")
    print("-" * 70)
    for label, r in sorted(rows, key=lambda kv: kv[1]["test_acc"]):
        print(f"{label:16s} {r['test_acc']:9.3f} {r['auc']:7.3f} {r['approx_p']:9.2e} "
              f"{r['fake_vs_fake_calib_acc']:9.3f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[json] 已保存 {args.json}")


if __name__ == "__main__":
    main()
