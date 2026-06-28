#!/usr/bin/env python3
"""
eval/forensic_cross.py — v14-fusion 多通道(收益率曲线)鉴别器: OUR forensic 的 3 通道增强版。

承 forensic_auc.py(2ch: stylized + DGS10 量化指纹 + 时间 held-out + 标定 AUC 自检),
新增【跨通道/收益率曲线】取证维度(c2st._cross_row 的 10 维: 股债/利率联动相关、尾部协动、
2s10s 斜率变化分布、同号率、lead-lag) —— 这是【单通道边际看不到】的取证维度。

为何要它(双重目标的鉴别器侧, A4 受控演示已证): 一个把各通道【边际分布】都做对、却把
【跨通道联合结构】(股债相关/收益率曲线)做错的 fake, 仅靠边际特征 AUC≈0.5(检不出);
加入跨通道特征后 AUC→1.0(近完美检出)。多通道给了鉴别器边际盲区之外的判别力。

诚实护栏(务必遵守):
  - 时间 held-out 切分(前 frac 训 / 后段测 + guard 缓冲带)消 stride 重叠窗的邻接泄露。
  - real-vs-real 标定自检: 两套特征都报; 50yr 利率体制非平稳会让它 >0.5(合法), 故真正的
    "无信号对照"看 marginal_only 在【保边际/毁联合】扰动上≈0.5(见 A4)。
  - 这些跨通道特征【只进本鉴别器 / 可选报告】, 【绝不进 go/no-go 诚实闸门】(防移动球门、防 Goodhart);
    诚实闸门(复制率/C2ST_新颖/SigP_新颖)仍在一致的边际+签名空间上判, 且须对 3ch+L 重标定。
  - 纯离线评估/选型, 绝不回喂训练损失。hw01 三栏只作外部参照, 绝不去 fool。

用法:
  conda run -n ts_diffusion python eval/forensic_cross.py \
    --fakes fusion=output/fusion_v14_ar2048.csv [v10=output/...csv ...] \
    --json eval/forensic_cross_results.json
real 默认由主CSV(sp500,DGS10)+ FRED(DGS2_diff)按交易日 join 构造 3 通道。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnostics import rolling_std, VOL_WINDOW            # noqa: E402
import c2st as C                                            # noqa: E402

MAIN = "/home/u00134/data/train_sp500_us10y.csv"
FRED = "/home/u00134/data/fred_treasury.csv"
STRIDE = 5
# 候选 CSV 的列名前缀 → 通道顺序 (与 config.OUTPUT_PREFIX / fusion 一致)
CH_PREFIX = ["sp500", "dgs10", "dgs2"]


def load_real_3ch(L):
    """主CSV(sp500,DGS10=已差分) + FRED(DGS2_diff) 按交易日 join → (N,3,L) 滑窗。"""
    m = pd.read_csv(MAIN, index_col=0, parse_dates=True)
    f = pd.read_csv(FRED, index_col=0, parse_dates=True)
    j = m.join(f[["DGS2_diff"]], how="inner").dropna(subset=["sp500", "DGS10", "DGS2_diff"])
    arr = j[["sp500", "DGS10", "DGS2_diff"]].values.astype(np.float64)   # (T,3)
    idx = range(0, len(arr) - L + 1, STRIDE)
    return np.stack([arr[s:s + L].T for s in idx], axis=0)               # (N,3,L)


def load_cand_3ch(path):
    """候选生成 CSV (宽表 sp500_*/dgs10_*/dgs2_*) → (N,3,L)。"""
    df = pd.read_csv(path)
    chans = []
    for pre in CH_PREFIX:
        cols = sorted([c for c in df.columns if c.startswith(pre + "_") and c.split("_")[-1].isdigit()],
                      key=lambda c: int(c.split("_")[-1]))
        if len(cols) < 2:
            raise SystemExit(f"[forensic_cross] {path} 缺通道 '{pre}_*' (多通道候选须含 sp500_/dgs10_/dgs2_)")
        chans.append(df[cols].values.astype(np.float64))
    return np.stack(chans, axis=1)                                       # (N,3,L)


def cross_only_feats(W):
    return np.nan_to_num(np.array([C._cross_row(w[0], w[1], w[2]) for w in W], dtype=np.float64))


def time_split(n, frac, guard):
    cut = int(n * frac)
    return np.arange(0, max(1, cut - guard)), np.arange(min(n - 1, cut + guard), n)


def auc_pair(Xa_tr, Xb_tr, Xa_te, Xb_te, seed=42):
    Xtr = np.vstack([Xa_tr, Xb_tr]); ytr = np.r_[np.zeros(len(Xa_tr)), np.ones(len(Xb_tr))]
    Xte = np.vstack([Xa_te, Xb_te]); yte = np.r_[np.zeros(len(Xa_te)), np.ones(len(Xb_te))]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", nargs="+", required=True, help="label=path ... (3通道宽表)")
    ap.add_argument("--json", default="eval/forensic_cross_results.json")
    ap.add_argument("--frac", type=float, default=0.8)
    args = ap.parse_args()

    # L 由第一个候选决定
    first = args.fakes[0].split("=", 1)[1]
    L = load_cand_3ch(first).shape[-1]
    real = load_real_3ch(L)
    vol_thr = float(np.median(rolling_std(real[:, 0, :], VOL_WINDOW)))
    guard = int(np.ceil(L / STRIDE))
    print(f"[forensic_cross] L={L} real={real.shape} guard={guard} vol_thr={vol_thr:.5f}")

    # real 特征 (marginal=stylized+DGS指纹 / cross=10维曲线)
    Xm_r = C.featurize(real, vol_thr, dgs10_fp=True, cross_channel=False)
    Xc_r = cross_only_feats(real)
    Xmc_r = np.hstack([Xm_r, Xc_r])
    print(f"[forensic_cross] 特征维度 marginal={Xm_r.shape[1]} cross={Xc_r.shape[1]} all={Xmc_r.shape[1]}")

    out = {"L": L, "n_real": int(len(real)), "dim": {"marginal": int(Xm_r.shape[1]), "cross": int(Xc_r.shape[1])},
           "calib_real_vs_real": {}, "detect": {}}

    # ── 标定自检: real-vs-real 早/晚半切 (两套特征都报) ──
    e_idx, l_idx = time_split(len(real), 0.5, guard)
    for name, Xr in [("marginal", Xm_r), ("cross", Xc_r), ("marginal+cross", Xmc_r)]:
        ce, cl = len(e_idx) // 2, len(l_idx) // 2
        Re, Rl = Xr[e_idx], Xr[l_idx]
        out["calib_real_vs_real"][name] = auc_pair(Re[:ce], Rl[:cl], Re[ce:], Rl[cl:])
    print(f"[forensic_cross] 标定 real-vs-real AUC: "
          f"{ {k: round(v,3) for k,v in out['calib_real_vs_real'].items()} } "
          f"(>0.5=50yr非平稳合法; 真负对照见 A4 marginal≈0.5)")

    # ── 检出: real(时间 held-out) vs 各候选 ──
    tr, te = time_split(len(real), args.frac, guard)
    print(f"\n  {'candidate':18s} {'marginal':>9s} {'+cross':>8s} {'cross_only':>11s} {'跨通道增益':>10s}   AUC↑=易识破为假")
    print("  " + "-" * 74)
    for spec in args.fakes:
        label, path = spec.split("=", 1)
        cand = load_cand_3ch(path)
        n = min(len(real), len(cand))
        Xm_f = C.featurize(cand, vol_thr, dgs10_fp=True, cross_channel=False)
        Xc_f = cross_only_feats(cand)
        Xmc_f = np.hstack([Xm_f, Xc_f])
        # real/fake 各自时间切分 (候选无时间序则整体切; 用 real 的 tr/te 索引上限对齐)
        trf = tr[tr < len(cand)]; tef = te[te < len(cand)]
        a_m = auc_pair(Xm_r[tr], Xm_f[trf], Xm_r[te], Xm_f[tef])
        a_mc = auc_pair(Xmc_r[tr], Xmc_f[trf], Xmc_r[te], Xmc_f[tef])
        a_c = auc_pair(Xc_r[tr], Xc_f[trf], Xc_r[te], Xc_f[tef])
        out["detect"][label] = {"marginal": a_m, "marginal+cross": a_mc, "cross_only": a_c,
                                "cross_gain": a_mc - a_m}
        print(f"  {label:18s} {a_m:>9.3f} {a_mc:>8.3f} {a_c:>11.3f} {a_mc-a_m:>+10.3f}")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=2, ensure_ascii=False)
    print(f"\n[json] {args.json}")
    print("[判读] marginal+cross AUC 越高=该候选越易识破(跨通道结构越假); cross_gain>0=跨通道特征"
          "贡献了边际外的判别力。AUC→0.5=候选连联合结构都逼真(生成目标)。")


if __name__ == "__main__":
    main()
