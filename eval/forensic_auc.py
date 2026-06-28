#!/usr/bin/env python3
"""
eval/forensic_auc.py — OUR forensic 鉴别器(增强版): 标定 ROC-AUC + 时间 held-out + DGS10 量化指纹。

双重目标的"鉴别器"那一半。相对 c2st.py 的三处升级(line1 路线图 D1+D2):
  1. **特征加 DGS10 量化指纹**(c2st.featurize(dgs10_fp=True)): 真实 DGS10 量化到 0.01、生成连续浮点
     → 近完美抓手。
  2. **时间 held-out 切分**(治泄露): c2st.py 在 stride=5 的重叠窗上随机 train_test_split, 相邻窗共享
     >99% 点 = 时间邻接泄露。这里真实窗按【时间】切(前 80% 训 / 后 20% 测)+ 两侧丢 guard=ceil(L/stride)
     缓冲带, 消除重叠泄露。
  3. **标定 AUC 自检**(real-vs-real, 前后时间半切): 须 ≈0.5(∈[0.45,0.55])才放行; 远离则有泄露或
     真实非平稳(金融时序合法 >0.5, 勿强压到 0.5)。

报告每个候选: detect_AUC(stylized) / detect_AUC(+DGS指纹) / 指纹边际增益。AUC→1.0=易识破=该生成器假。
⚠️ 仅离线评估/选型, 绝不回喂训练损失(项目红线; DGS 指纹易被 np.round 绕过, 见路线图 §D5)。

用法:
  conda run -n ts_diffusion python eval/forensic_auc.py \
    --fakes v10=output/deep_v10.csv c1=output/deep_v13_c1_ctx_realctx2048.csv --json eval/forensic_auc.json
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2st as C  # noqa: E402
from diagnostics import load_changes, rolling_std, STRIDE, VOL_WINDOW  # noqa: E402


def _feat(w, vol_thr, fp):
    return C.featurize(w, vol_thr, dgs10_fp=fp)


def _auc(Xa_tr, Xb_tr, Xa_te, Xb_te, seed=42):
    """二分类 AUC: (Xa=0, Xb=1) 训 → 在 held-out 上算 ROC-AUC。"""
    Xtr = np.vstack([Xa_tr, Xb_tr]); ytr = np.r_[np.zeros(len(Xa_tr)), np.ones(len(Xb_tr))]
    Xte = np.vstack([Xa_te, Xb_te]); yte = np.r_[np.zeros(len(Xa_te)), np.ones(len(Xb_te))]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def time_split(n, frac, guard):
    """时间序窗 → (前 frac − guard 缓冲) 训 / (后 1−frac − guard 缓冲) 测。"""
    cut = int(n * frac)
    return np.arange(0, max(1, cut - guard)), np.arange(min(n - 1, cut + guard), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="/home/u00134/data/train_sp500_us10y.csv")
    ap.add_argument("--fakes", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    L = load_changes(args.fakes[0].split("=", 1)[1]).shape[-1]
    real = load_changes(args.real, target_seq_len=L)           # 时间序窗
    vol_thr = float(np.median(rolling_std(real[:, 0, :], VOL_WINDOW)))
    guard = int(np.ceil(L / STRIDE))                           # 缓冲带(消重叠窗泄露)
    nR = len(real)
    print(f"[forensic_auc] L={L} real {nR} 窗 guard={guard}(=ceil(L/stride))")

    # ── 标定自检: real-vs-real 前后时间半切(带缓冲), AUC 须 ≈0.5 ──
    e_idx, l_idx = time_split(nR, 0.5, guard)
    Re, Rl = real[e_idx], real[l_idx]
    for fp in (False, True):
        # 各类内部再切训/测算 held-out AUC
        ce, cl = len(Re) // 2, len(Rl) // 2
        cal = _auc(_feat(Re[:ce], vol_thr, fp), _feat(Rl[:cl], vol_thr, fp),
                   _feat(Re[ce:], vol_thr, fp), _feat(Rl[cl:], vol_thr, fp))
        tag = "stylized+DGS指纹" if fp else "stylized"
        flag = "✅≈0.5" if 0.45 <= cal <= 0.55 else ("⚠️非平稳/泄露" if cal > 0.55 else "?")
        print(f"  标定 real-vs-real({tag}): AUC={cal:.3f} {flag}")
        if not fp:
            cal_styl = cal
        else:
            cal_fp = cal

    # ── 检出: 真实(时间 held-out) vs 各候选 ──
    r_tr, r_te = time_split(nR, 0.8, guard)
    R_tr, R_te = real[r_tr], real[r_te]
    rng = np.random.default_rng(0)
    results = {"L": L, "guard": guard, "n_real": nR,
               "calib_real_vs_real": {"stylized": cal_styl, "stylized+dgsfp": cal_fp},
               "detect": {}}
    rows = []
    for spec in args.fakes:
        label, path = spec.split("=", 1)
        fake = load_changes(path)
        pf = rng.permutation(len(fake)); h = int(len(fake) * 0.7)
        F_tr, F_te = fake[pf[:h]], fake[pf[h:]]
        auc_styl = _auc(_feat(R_tr, vol_thr, False), _feat(F_tr, vol_thr, False),
                        _feat(R_te, vol_thr, False), _feat(F_te, vol_thr, False))
        auc_fp = _auc(_feat(R_tr, vol_thr, True), _feat(F_tr, vol_thr, True),
                      _feat(R_te, vol_thr, True), _feat(F_te, vol_thr, True))
        results["detect"][label] = {"auc_stylized": auc_styl, "auc_stylized_dgsfp": auc_fp,
                                     "dgsfp_marginal": auc_fp - auc_styl, "n_fake": len(fake)}
        rows.append((label, auc_styl, auc_fp))

    print(f"\n  标定 real-vs-real: stylized AUC={cal_styl:.3f} / +DGS指纹 AUC={cal_fp:.3f}  (应≈0.5)")
    print(f"\n  {'candidate':18s} {'AUC(stylized)':>14s} {'AUC(+DGS指纹)':>15s} {'指纹边际':>9s}   AUC↑=越易识破为假")
    print("  " + "-" * 70)
    for label, a1, a2 in sorted(rows, key=lambda r: -r[2]):
        print(f"  {label:18s} {a1:>14.3f} {a2:>15.3f} {a2-a1:>+9.3f}")

    if args.json:
        json.dump(results, open(args.json, "w"), indent=2, ensure_ascii=False)
        print(f"\n  [json] {args.json}")


if __name__ == "__main__":
    main()
