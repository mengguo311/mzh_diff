#!/usr/bin/env python3
"""
concat_windows.py — 把 L 宽表样本【朴素首尾拼接】成 K*L 长样本 (v13 A1 长程基线检查)。

A1(缩窗 512)单独无法产出 2048 长样本; 朴素拼接 K 个独立 512 窗 → 故意【无跨窗 regime 连续性】,
用于量化"短窗模型在 2048 全长的长程退化下限"(后续 C1 自回归条件来修)。注意: 这不是 A1 的合格判据,
是诊断 C1 需修多少的标尺。
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4, help="拼接窗数 (512*4=2048)")
    a = ap.parse_args()

    df = pd.read_csv(a.inp)
    spc = sorted([c for c in df.columns if c.startswith("sp500_") and c.split("_")[-1].isdigit()],
                 key=lambda c: int(c.split("_")[-1]))
    dgc = sorted([c for c in df.columns if c.startswith("dgs10_") and c.split("_")[-1].isdigit()],
                 key=lambda c: int(c.split("_")[-1]))
    sp = df[spc].values.astype(np.float64)
    dg = df[dgc].values.astype(np.float64)
    L = sp.shape[1]
    n = (len(sp) // a.k) * a.k
    LL = a.k * L
    sp = sp[:n].reshape(-1, LL)   # 行主序: 每行 = 连续 k 个独立窗首尾相接
    dg = dg[:n].reshape(-1, LL)

    out = {}
    for i in range(LL):
        out[f"sp500_{i}"] = sp[:, i]
    for i in range(LL):
        out[f"dgs10_{i}"] = dg[:, i]
    pd.DataFrame(out).to_csv(a.out, index=False)
    print(f"[concat] {sp.shape[0]} 条 x {LL} (k={a.k}x{L}) 写入 {a.out}")


if __name__ == "__main__":
    main()
