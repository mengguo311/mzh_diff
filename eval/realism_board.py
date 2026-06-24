#!/usr/bin/env python3
"""
eval/realism_board.py — line1【真实度】北极星: 六族否决式真实度向量 + real-vs-real 自检带 + 污染 gap。

替代单标量 composite(其 70% 权重在新颖/抗复制轴 = line2 北极星, 对 line1 容许复制是错的且必被 game)。
判定规则(否决式, 不互相补偿): 六族任一族落出【real-vs-real 自检带】即该族 FAIL → 该候选真实度 NOT-PASS。

六族(诚实诊断 + 分布距离, 不用 hw01 fool):
  1. tail_kurt   —— SP500 收益超额峰度(真实~12-18, clip 天花板; line1 攻厚尾主战场)
  2. wasserstein —— SP500 收益 1D-Wasserstein 到真实(越小越真; 带=real-vs-real 自距)
  3. vol_acf1    —— |r|-ACF lag1(波动聚集)
  4. leverage1   —— corr(r_t,|r_{t+1}|)(杠杆效应)
  5. regime      —— high_vol_frac(→0.5) + mean_run_len(→真实~39)  [两子项须同过]
  6. dgs10_grid  —— DGS10 frac_on_grid(真实=1.0, 量化指纹)

自检带: 真实窗按【时间非重叠半切】+ bootstrap → 每族 mean±K*std(K=3); 断言真实自身落带内(否则带定义 bug)。
污染 gap(M2, 容许复制安全边界): 每族【同时】在全集 + novel 子集(novelty_rerank 非复制子集)各算一份;
  若全集落带但 novel 落不进 → 标 RESTING ON COPIES(真实度是抄来的, 不算数)。复制率只监控不约束。

用法:
  conda run -n ts_diffusion python eval/realism_board.py \
    --fakes v10r=output/deep_v10_retrained.csv v9=output/deep_v9_20k.csv --json eval/realism_board.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import kurtosis as _kurt, wasserstein_distance

REPO = "/home/u00134/src"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
import c2st as C  # noqa: E402
from diagnostics import load_changes, rolling_std, VOL_WINDOW  # noqa: E402
from memorization import _std  # noqa: E402
from novelty_rerank import copy_split  # noqa: E402
import forensic_suite as F  # noqa: E402  复用 abs_acf_curve / leverage_curve

K_BAND = 3.0
REAL_DEFAULT = F.REAL_DEFAULT


# ── 六族标量(输入 windows (N,2,L) 真实量级 → 标量) ──
def fam_values(w):
    sp, dg = w[:, 0, :], w[:, 1, :]
    rs = rolling_std(sp, VOL_WINDOW)
    return {
        "tail_kurt": float(_kurt(sp.ravel())),
        "vol_acf1": float(F.abs_acf_curve(sp, 1)[0]),
        "leverage1": float(F.leverage_curve(sp, 1)[0]),
        "high_vol_frac": float(np.mean(rs > _VOL_THR[0])),
        "mean_run_len": _mean_run_len(rs, _VOL_THR[0]),
        "dgs10_grid": float(np.mean(np.abs(dg * 100.0 - np.round(dg * 100.0)) < 1e-8)),
    }


def _mean_run_len(rs, thr):
    state = rs > thr
    flips = np.abs(np.diff(state.astype(np.int8), axis=1)).sum(1)
    return float(np.mean(state.shape[1] / (flips + 1.0)))


_VOL_THR = [None]   # 真实滚动波动率中位阈值(全局, regime 统一口径)


def real_bands(real, n_boot=120, seed=0):
    """真实窗时间非重叠半切 + bootstrap → 每族 band。wasserstein 用 real-vs-real 自距。"""
    rng = np.random.default_rng(seed)
    n = len(real)
    # 时间非重叠半切(前 50% / 后 50%, 各丢 guard 缓冲带防滑窗重叠污染带宽)
    guard = int(np.ceil(real.shape[-1] / 5))
    A = real[: max(1, n // 2 - guard)]; B = real[n // 2 + guard:]
    # 值族: bootstrap 整个 real → mean±K*std
    keys = ["tail_kurt", "vol_acf1", "leverage1", "high_vol_frac", "mean_run_len", "dgs10_grid"]
    samp = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fam_values(real[idx])
        for k in keys:
            samp[k].append(v[k])
    bands = {}
    for k in keys:
        m, s = float(np.mean(samp[k])), float(np.std(samp[k]) + 1e-9)
        bands[k] = {"lo": m - K_BAND * s, "hi": m + K_BAND * s, "real": m, "type": "two-sided"}
    # wasserstein 族: real-vs-real 自距(A vs B pooled 收益) 的 bootstrap 上界
    wss = []
    spA, spB = A[:, 0, :].ravel(), B[:, 0, :].ravel()
    for _ in range(n_boot):
        a = rng.choice(spA, min(len(spA), 4000)); b = rng.choice(spB, min(len(spB), 4000))
        wss.append(wasserstein_distance(a, b))
    wm, wsd = float(np.mean(wss)), float(np.std(wss) + 1e-9)
    bands["wasserstein"] = {"lo": 0.0, "hi": wm + K_BAND * wsd, "real": wm, "type": "upper"}
    return bands, real[:, 0, :].ravel()


def check(value, band):
    if band["type"] == "upper":
        return value <= band["hi"]
    return band["lo"] <= value <= band["hi"]


def eval_set(w, real_sp_pool, bands):
    """对一组窗算六族值 + wasserstein, 返回 {family: (value, pass)}。"""
    v = fam_values(w)
    sp = w[:, 0, :].ravel()
    rng = np.random.default_rng(1)
    a = rng.choice(sp, min(len(sp), 4000)); b = rng.choice(real_sp_pool, min(len(real_sp_pool), 4000))
    v["wasserstein"] = float(wasserstein_distance(a, b))
    out = {}
    for k, band in bands.items():
        out[k] = {"value": round(v[k], 4), "pass": bool(check(v[k], band))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=REAL_DEFAULT)
    ap.add_argument("--fakes", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--json", default=None)
    ap.add_argument("--thresh", type=float, default=0.95)
    args = ap.parse_args()

    L = load_changes(args.fakes[0].split("=", 1)[1]).shape[-1]
    real = load_changes(args.real, target_seq_len=L)
    _VOL_THR[0] = float(np.median(rolling_std(real[:, 0, :], VOL_WINDOW)))
    print(f"[realism_board] L={L} real {len(real)} 窗; 建 real-vs-real 自检带(时间非重叠半切+bootstrap)")
    bands, real_sp_pool = real_bands(real)

    # 自检: 真实自身须落带内
    self_v = eval_set(real, real_sp_pool, bands)
    self_fail = [k for k, r in self_v.items() if not r["pass"]]
    print("  自检 real 落带: " + ("✅ 全过" if not self_fail else f"⚠️ 落出 {self_fail}(带定义 bug?)"))
    for k, b in bands.items():
        print(f"    {k:14s} 带=[{b['lo']:.4f}, {b['hi']:.4f}] real={b['real']:.4f}")

    # novel 子集所需
    vol_thr = _VOL_THR[0]
    Xreal = C.featurize(real, vol_thr)
    mu, sd = Xreal.mean(0), Xreal.std(0) + 1e-12
    Rz = _std(Xreal, mu, sd)

    results = {"L": L, "bands": bands, "self_check_fail": self_fail, "detect": {}}
    fams = list(bands.keys())
    print(f"\n  {'candidate':14s} " + " ".join(f"{k[:9]:>9s}" for k in fams) + "  真实度  污染")
    print("  " + "-" * 104)
    for spec in args.fakes:
        label, path = spec.split("=", 1)
        fake = load_changes(path)
        allv = eval_set(fake, real_sp_pool, bands)
        # 污染 gap: novel 子集
        mask, raw_p = copy_split(real, fake, Rz, mu, sd, vol_thr, args.thresh)
        novel = fake[~mask]; n_nov = int((~mask).sum()); crate = float(mask.mean())
        novv = eval_set(novel, real_sp_pool, bands) if n_nov >= 200 else None
        realism_pass = all(allv[k]["pass"] for k in fams)        # 否决式
        resting = []
        if novv is not None:
            resting = [k for k in fams if allv[k]["pass"] and not novv[k]["pass"]]
        results["detect"][label] = {"copy_rate": crate, "n_novel": n_nov,
                                    "all": allv, "novel": novv,
                                    "realism_pass": realism_pass, "resting_on_copies": resting}
        marks = "".join("P" if allv[k]["pass"] else "F" for k in fams)
        cont = ("RESTING:" + ",".join(resting)) if resting else ("(novel n<200)" if novv is None else "clean")
        print(f"  {label:14s} " + " ".join(f"{allv[k]['value']:>9.3f}" for k in fams)
              + f"  {'PASS' if realism_pass else 'FAIL':>5s}[{marks}] {cont}")
    print("\n  否决式: 六族(+wass)任一 FAIL → 真实度 FAIL; RESTING=该族全集过但 novel 不过(靠抄). 复制率仅监控.")

    if args.json:
        json.dump(results, open(args.json, "w"), indent=2, ensure_ascii=False, default=float)
        print(f"  [json] {args.json}")


if __name__ == "__main__":
    main()
