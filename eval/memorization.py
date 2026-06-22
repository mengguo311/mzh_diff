#!/usr/bin/env python3
"""
eval/memorization.py — 记忆化检查 (v12 路线图 A1)

问题: 生成样本是否在"背诵"训练窗(记忆化), 还是学到了真实分布?
非自指做法: 在 c2st 的 24 维 stylized 特征空间, 算每条 fake 到【全部 real 训练窗】的最近邻
距离 d_NN, 与【real 窗到非重叠 real 窗】的 d_NN 分布(记忆化零假设地板)对比:
  ratio = median(d_NN_fake) / median(d_NN_real_ref)
    - ratio << 1 (且 bootstrap CI 上界 < 1) → fake 异常贴近训练集 = 记忆化迹象;
    - ratio ≈ 1 → 非记忆(若仍回退, 则病根是分布漂移/过平滑而非背诵)。
top-k 最近对再回 z-score 原序列空间算逐通道 Pearson, 二次确认是否近似复制(>0.9)。

用法:
  conda run -n ts_diffusion python eval/memorization.py \
    --fakes v11=output/deep_v11_val.csv v10_retrained=output/deep_v10_retrained.csv \
            v9=output/deep_v9_20k.csv --json eval/memorization.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2st as _c2st  # noqa: E402
from diagnostics import load_changes, rolling_std, VOL_WINDOW  # noqa: E402


def _std(X, mu, sd):
    return (X - mu) / sd


def nn_to_bank(Q, bank, chunk=512):
    """每条 Q 到 bank 的最近邻欧氏距离 + 最近邻索引 (特征空间)。"""
    dmin = np.empty(len(Q)); amin = np.empty(len(Q), dtype=int)
    for i in range(0, len(Q), chunk):
        d = cdist(Q[i:i + chunk], bank)
        dmin[i:i + chunk] = d.min(1); amin[i:i + chunk] = d.argmin(1)
    return dmin, amin


def nn_real_ref(R, guard, chunk=512):
    """real-to-real 最近邻距离 + 索引, 排除时间上重叠的邻窗 (|i-j| <= guard)。"""
    n = len(R); out = np.empty(n); aidx = np.empty(n, dtype=int)
    for i in range(0, n, chunk):
        d = cdist(R[i:i + chunk], R)                      # (chunk, n)
        for r in range(d.shape[0]):
            gi = i + r
            d[r, max(0, gi - guard):min(n, gi + guard + 1)] = np.inf
        out[i:i + chunk] = d.min(1); aidx[i:i + chunk] = d.argmin(1)
    return out, aidx


def boot_ratio(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    rs = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(n)]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _pearson(a, b):
    az = (a - a.mean()) / (a.std() + 1e-12)
    bz = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.corrcoef(az, bz)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="/home/u00134/data/train_sp500_us10y.csv")
    ap.add_argument("--fakes", nargs="*", default=[], help="label=path ... (留空=只算 L 标定地板)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--stride", type=int, default=5)  # 须与 diagnostics.load_changes 的 STRIDE 一致
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--L", type=int, default=None, help="窗长 (默认从首个 fake 推断; 步0 标定用 --L 512)")
    args = ap.parse_args()

    if args.L is not None:
        L = args.L
    elif args.fakes:
        L = load_changes(args.fakes[0].split("=", 1)[1]).shape[-1]
    else:
        ap.error("需要 --L 或 至少一个 --fakes")
    real = load_changes(args.real, target_seq_len=L)         # (N_real,2,L) 时间序
    n_real = real.shape[0]
    guard = int(np.ceil(L / args.stride))                    # 重叠窗保护半径
    vol_thr = float(np.median(rolling_std(real[:, 0, :], VOL_WINDOW)))
    Rf = _c2st.featurize(real, vol_thr)
    mu, sd = Rf.mean(0), Rf.std(0) + 1e-12
    Rz = _std(Rf, mu, sd)

    print(f"[mem] L={L}, real {n_real} 窗, guard={guard}(排除重叠邻窗), 特征 {Rf.shape[1]} 维")
    ref, ref_idx = nn_real_ref(Rz, guard)
    ref_med = float(np.median(ref))
    # 标定地板: 真实窗到【非重叠】最近真实窗的原序列复制率 (市场自相似的自然底噪)
    ref_rawp = np.array([np.mean([_pearson(real[i, ch], real[ref_idx[i], ch]) for ch in range(2)])
                         for i in range(n_real)])
    ref_f95 = float((ref_rawp > 0.95).mean()); ref_f99 = float((ref_rawp > 0.99).mean())
    print(f"[mem] 标定 real-vs-real(非重叠): d_NN 中位={ref_med:.4f}; "
          f"原序列复制率>.95={ref_f95:.1%} >.99={ref_f99:.1%}  —— 市场自相似自然底噪\n")

    results = {"L": L, "n_real": n_real, "guard": guard, "real_ref_dNN_median": ref_med,
               "real_self_copy_frac_p95": ref_f95, "real_self_copy_frac_p99": ref_f99, "detect": {}}
    rows = []
    for spec in args.fakes:
        label, path = spec.split("=", 1)
        fake = load_changes(path)
        Ff = _c2st.featurize(fake, vol_thr)
        Fz = _std(Ff, mu, sd)
        d, nn_idx = nn_to_bank(Fz, Rz)
        med = float(np.median(d)); ratio = med / ref_med
        lo, hi = boot_ratio(d, ref)
        # 关键指标: 对【全部】fake 样本, 算到其特征-最近邻 real 窗的原序列 Pearson (两通道均值),
        # 复制率 = Pearson 超阈值的比例。复制窗在特征空间也最近, 故特征-NN 能锁定它。
        raw_p = np.array([np.mean([_pearson(fake[j, ch], real[nn_idx[j], ch]) for ch in range(2)])
                          for j in range(len(fake))])
        frac90 = float((raw_p > 0.90).mean()); frac95 = float((raw_p > 0.95).mean())
        frac99 = float((raw_p > 0.99).mean()); pmean = float(raw_p.mean())
        results["detect"][label] = {"dNN_median": med, "ratio_vs_ref": ratio, "ratio_ci95": [lo, hi],
                                    "raw_pearson_mean": pmean, "copy_frac_p90": frac90,
                                    "copy_frac_p95": frac95, "copy_frac_p99": frac99, "n_fake": len(fake)}
        rows.append((label, med, ratio, pmean, frac95, frac99))

    if rows:
        print(f"{'candidate':16s} {'dNN中位':>9s} {'ratio':>7s} {'raw_p均值':>9s} {'复制率>.95':>10s} {'>.99':>8s}  判读")
        print("-" * 78)
        for label, med, ratio, pmean, f95, f99 in sorted(rows, key=lambda r: -r[4]):
            flag = "严重记忆化!" if f95 > 0.20 else ("明显记忆" if f95 > 0.05 else ("零星复制" if f95 > 0.005 else "未见复制"))
            print(f"{label:16s} {med:9.4f} {ratio:7.2f} {pmean:9.3f} {f95:9.1%} {f99:8.1%}  {flag}")
    print(f"\n标定底噪(真实新数据 vs 训练库): 复制率>.95={ref_f95:.1%}  >.99={ref_f99:.1%}")
    print(f"判读: 复制率 = 与最近邻真实窗原序列 Pearson 超阈值的样本占比 (随机对≈0)。")
    print(f"      若 fake 复制率 >> 标定底噪 → 模型在背诵训练集(记忆化); 若 ≈ 底噪 → 市场自相似而非记忆。")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[json] 已保存 {args.json}")


if __name__ == "__main__":
    main()
