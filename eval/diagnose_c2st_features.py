#!/usr/bin/env python3
"""
eval/diagnose_c2st_features.py — 诊断 C2ST_新颖 平台: 哪些 stylized 特征仍把【新颖子集】与真实分开。

对候选 CSV: featurize → 剔除复制(raw pearson>0.95)留新颖子集 → 逐特征算【1D 分离力 AUC】
(单特征区分 real vs 新颖的能力, 0.5=不可分/已逼真, →1=该特征仍暴露假) + 标准化均值差
+ c2st LogReg 系数(分类器最看重哪些特征)。Top AUC 特征 = 下一杠杆应攻的目标。

用法: conda run -n ts_diffusion python eval/diagnose_c2st_features.py <real.csv> <candidate.csv>
"""
import sys
import numpy as np

sys.path.insert(0, "/home/u00134/src/eval")
from diagnostics import load_changes, rolling_std, VOL_WINDOW  # noqa: E402
import c2st as C  # noqa: E402
from memorization import nn_to_bank, _std, _pearson  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

NAMES = (["sk_s", "ku_s(峰度sp)", "sk_d", "ku_d(峰度dg)", "corr(双通道)", "tail_corr(尾相关)"]
         + [f"sp_|r|acf{l}" for l in range(1, 6)] + [f"dg_|r|acf{l}" for l in range(1, 6)]
         + ["d2e(粗糙)", "tv", "racf1", "high_vol_frac", "switch", "runlen(regime黏)", "maxv", "volvol"])

real_path, cand_path = sys.argv[1], sys.argv[2]
fake = load_changes(cand_path); L = fake.shape[-1]
real = load_changes(real_path, target_seq_len=L)
vol_thr = float(np.median(rolling_std(real[:, 0, :], VOL_WINDOW)))
Xr = C.featurize(real, vol_thr); mu, sd = Xr.mean(0), Xr.std(0) + 1e-12; Rz = _std(Xr, mu, sd)
Ff = C.featurize(fake, vol_thr); Fz = _std(Ff, mu, sd)

# 新颖子集
_, nn = nn_to_bank(Fz, Rz)
raw_p = np.array([np.mean([_pearson(fake[j, ch], real[nn[j], ch]) for ch in range(2)]) for j in range(len(fake))])
novel = fake[raw_p <= 0.95]
Xn = C.featurize(novel, vol_thr)
print(f"[diag] L={L} real {len(Xr)} 窗 / 候选 {len(fake)}(新颖 {len(Xn)}, 复制率 {(raw_p>0.95).mean():.1%})")

# 逐特征 1D 分离力
rows = []
for i, nm in enumerate(NAMES):
    rv, fv = Xr[:, i], Xn[:, i]
    smd = (fv.mean() - rv.mean()) / (rv.std() + 1e-9)
    y = np.r_[np.zeros(len(rv)), np.ones(len(fv))]; s = np.r_[rv, fv]
    try:
        auc = roc_auc_score(y, s); auc = max(auc, 1 - auc)
    except Exception:
        auc = 0.5
    rows.append((nm, float(rv.mean()), float(fv.mean()), float(smd), float(auc)))
rows.sort(key=lambda r: -r[4])

print(f"\n{'特征':<16}{'真实均值':>11}{'新颖均值':>11}{'标准化差':>9}{'1D分离力AUC':>12}  (AUC→0.5=已逼真, →1=仍暴露假)")
print("-" * 74)
for nm, rm, fm, smd, auc in rows:
    flag = "  ★主分离源" if auc > 0.75 else ("  ·次要" if auc > 0.65 else "")
    print(f"{nm:<16}{rm:>11.4f}{fm:>11.4f}{smd:>+9.2f}{auc:>12.3f}{flag}")

# c2st LogReg 系数(分类器权重)
n = min(len(Xr), len(Xn)); rng = np.random.default_rng(0)
Xrb = Xr[rng.choice(len(Xr), n, replace=False)]; Xnb = Xn[rng.choice(len(Xn), n, replace=False)]
X = np.vstack([Xrb, Xnb]); yy = np.r_[np.zeros(n), np.ones(n)]
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
clf.fit(X, yy)
coef = clf.named_steps["logisticregression"].coef_[0]
order = np.argsort(-np.abs(coef))[:6]
print(f"\nc2st LogReg |系数| Top6(分类器最依赖的判别特征): " +
      ", ".join(f"{NAMES[i]}({coef[i]:+.2f})" for i in order))
print(f"\n判读: AUC>0.75 的特征是新颖样本仍被判别器抓住的【主分离源】→ 下一个 copy-safe 杠杆应直攻它们。")
