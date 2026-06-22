#!/usr/bin/env python3
"""
eval/ctx_ablation.py — v13 C1 必做消融门: 判定 context 是否真被模型用上(防 Sig-MMD 式 no-op)。

比较【real-ctx 生成】与【zero-ctx(force_null) 生成】的 regime 分布(逐路径 high_vol_frac / 平均
游程长度)。若两者分布显著不同 → 模型确实在用 context → 可继续写/用自回归长程重建; 若无差异 →
context 被忽略, C1 退化为短窗 i.i.d. 拼接, 立即停, 退回 A2 配置。

用法:
  conda run -n ts_diffusion python eval/ctx_ablation.py \
    --real-ctx output/c1_realctx.csv --zero-ctx output/c1_zeroctx.csv --json eval/ctx_ablation.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnostics import load_changes, rolling_std, VOL_WINDOW  # noqa: E402


def regime_feats(csv, vol_thr):
    w = load_changes(csv)
    rs = rolling_std(w[:, 0, :], VOL_WINDOW)              # (N, L-w+1)
    state = rs > vol_thr
    high = state.mean(1)                                  # 逐路径高波动占比
    flips = np.abs(np.diff(state.astype(np.int8), axis=1)).sum(1)
    runlen = state.shape[1] / (flips + 1.0)               # 逐路径平均游程
    return high, runlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-ctx", required=True, dest="real_ctx")
    ap.add_argument("--zero-ctx", required=True, dest="zero_ctx")
    ap.add_argument("--real", default="/home/u00134/data/train_sp500_us10y.csv")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    L = load_changes(a.real_ctx).shape[-1]
    realw = load_changes(a.real, target_seq_len=L)
    vol_thr = float(np.median(rolling_std(realw[:, 0, :], VOL_WINDOW)))

    hr, rr = regime_feats(a.real_ctx, vol_thr)
    hz, rz = regime_feats(a.zero_ctx, vol_thr)
    ks_high = ks_2samp(hr, hz)
    ks_run = ks_2samp(rr, rz)

    print(f"high_vol_frac : real-ctx μ={hr.mean():.4f}  vs  zero-ctx μ={hz.mean():.4f}   KS p={ks_high.pvalue:.2e}")
    print(f"mean_run_len  : real-ctx μ={rr.mean():.2f}  vs  zero-ctx μ={rz.mean():.2f}   KS p={ks_run.pvalue:.2e}")
    used = (ks_high.pvalue < 0.01) or (ks_run.pvalue < 0.01)
    verdict = ("context 被用上 (regime 分布显著不同) → 可继续 C1 自回归长程重建"
               if used else "context 被忽略 (no-op) → 停 C1, 退回 A2 配置")
    print(f">>> {verdict}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"L": L, "high_real_mean": float(hr.mean()), "high_zero_mean": float(hz.mean()),
                       "ks_high_p": float(ks_high.pvalue), "runlen_real_mean": float(rr.mean()),
                       "runlen_zero_mean": float(rz.mean()), "ks_run_p": float(ks_run.pvalue),
                       "context_used": bool(used)}, f, indent=2, ensure_ascii=False)
        print(f"[json] {a.json}")


if __name__ == "__main__":
    main()
