#!/usr/bin/env python3
"""
eval/novelty_rerank.py — 记忆化校正后的【诚实排名】(v12 步骤 C)

现有 wasserstein/峰度/C2ST/Sig-MMD 都【奖励复制】:复制的样本本就是真实数据, 鉴别器测不出。
本工具把每个候选拆成【复制】(与最近邻训练窗 raw pearson>thresh)与【新颖】两半,
在 "real vs 新颖子集" 上重算 C2ST / Sig-MMD —— 这才反映模型真正的【生成】能力。

诚实判读:
  - c2st_novel / sig_p_novel 才是真实生成质量; c2st_all/sig_p_all 被复制的那半美化。
  - "最佳生成器" = 复制率可接受 且 c2st_novel 最低 / sig_p_novel 最高 的模型。

用法:
  conda run -n ts_diffusion python eval/novelty_rerank.py \
    --fakes v11=output/deep_v11_val.csv v10_retrained=output/deep_v10_retrained.csv \
            v10_sampling=output/deep_v10.csv v9=output/deep_v9_20k.csv \
    --json eval/novelty_rerank.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2st as C  # noqa: E402
import signature as S  # noqa: E402
from diagnostics import load_changes, rolling_std, VOL_WINDOW  # noqa: E402
from memorization import nn_to_bank, _std, _pearson  # noqa: E402


def copy_split(real, fake, Rz, mu, sd, vol_thr, thresh):
    """返回 (copy_mask, raw_pearson)。copy = 与特征最近邻训练窗原序列 pearson>thresh。"""
    Fz = _std(C.featurize(fake, vol_thr), mu, sd)
    _, nn = nn_to_bank(Fz, Rz)
    raw_p = np.array([np.mean([_pearson(fake[j, ch], real[nn[j], ch]) for ch in range(2)])
                      for j in range(len(fake))])
    return raw_p > thresh, raw_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="/home/u00134/data/train_sp500_us10y.csv")
    ap.add_argument("--fakes", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--thresh", type=float, default=0.95)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    L = load_changes(args.fakes[0].split("=", 1)[1]).shape[-1]
    real = load_changes(args.real, target_seq_len=L)
    vol_thr = float(np.median(rolling_std(real[:, 0, :], VOL_WINDOW)))
    Xreal = C.featurize(real, vol_thr)
    mu, sd = Xreal.mean(0), Xreal.std(0) + 1e-12
    Rz = _std(Xreal, mu, sd)
    scale = np.array([real[:, 0, :].std(), real[:, 1, :].std()]) + 1e-12

    # 标定臂 (real-vs-real)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xreal)); half = len(perm) // 2
    c2_cal = C.c2st(Xreal[perm[:half]], Xreal[perm[half:]])["test_acc"]
    A1 = S.sig_features(real, scale, 300, 2, 200, 3, seed=1)
    A2 = S.sig_features(real, scale, 300, 2, 200, 3, seed=2)
    sig_cal = S.sig_mmd_test(A1, A2, n_perm=300, seed=10)["p_value"]
    print(f"[rerank] L={L}, 标定 real-vs-real: C2ST acc={c2_cal:.3f}  Sig-MMD p={sig_cal:.3f}\n")

    results = {"L": L, "thresh": args.thresh,
               "calib": {"c2st_acc": c2_cal, "sig_p": sig_cal}, "detect": {}}
    rows = []
    for spec in args.fakes:
        label, path = spec.split("=", 1)
        fake = load_changes(path)
        mask, _ = copy_split(real, fake, Rz, mu, sd, vol_thr, args.thresh)
        novel = fake[~mask]
        crate = float(mask.mean()); n_nov = int((~mask).sum())
        Xall = C.featurize(fake, vol_thr); Xnov = C.featurize(novel, vol_thr)
        c_all = C.c2st(Xreal, Xall)["test_acc"]
        c_nov = C.c2st(Xreal, Xnov)["test_acc"]
        B_all = S.sig_features(fake, scale, 300, 2, 200, 3, seed=3)
        B_nov = S.sig_features(novel, scale, min(300, n_nov), 2, 200, 3, seed=3)
        p_all = S.sig_mmd_test(A1, B_all, n_perm=300, seed=11)["p_value"]
        p_nov = S.sig_mmd_test(A1, B_nov, n_perm=300, seed=11)["p_value"]
        results["detect"][label] = {"copy_rate": crate, "n_novel": n_nov,
                                    "c2st_all": c_all, "c2st_novel": c_nov,
                                    "sig_p_all": p_all, "sig_p_novel": p_nov}
        rows.append((label, crate, c_all, c_nov, p_all, p_nov))

    print(f"{'candidate':15s} {'复制率':>7s} {'C2ST_all':>9s} {'C2ST_新颖':>9s} {'SigP_all':>9s} {'SigP_新颖':>9s}")
    print("-" * 70)
    # 按"诚实生成质量"排序: 新颖子集 C2ST 越低越好
    for label, cr, ca, cn, pa, pn in sorted(rows, key=lambda r: r[3]):
        print(f"{label:15s} {cr:7.1%} {ca:9.3f} {cn:9.3f} {pa:9.3f} {pn:9.3f}")
    print(f"\n标定 C2ST≈{c2_cal:.3f} / Sig-MMD p≈{sig_cal:.3f}。诚实判读: 看【新颖】列 ——")
    print(f"  C2ST_新颖 越接近标定越好; 若 C2ST_新颖 >> C2ST_all, 说明该模型的'真实感'是复制撑起来的。")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[json] 已保存 {args.json}")


if __name__ == "__main__":
    main()
