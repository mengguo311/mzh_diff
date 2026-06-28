#!/usr/bin/env python3
"""
fhs_baseline.py — E0: FHS / GJR-GARCH(1,1)-t 零训练强基线(纯 numpy/scipy, 不碰 DiT)

研究路线 research/recommended_experiments.md §E0 的落地实现。动机:在 ~7 个独立宏观窗的
硬约束下, 学【~6 参/通道的低维条件波动递归 + 经验残差经验分布】比训 58M DiT 估 2048 维联合
密度【免疫数据稀缺】—— 新 innovation ⇒ 复制率≈0, 经验残差原生带真峰度 ⇒ 直击 line2/C1 头号
FAIL(欠厚尾)。给出"零数据的强先验能到哪"的锚, 是 E1-E4 的对照基线 + E2/E3 的 GARCH 引擎件。

================================ 模型 ================================
每通道独立拟合 GJR-GARCH(1,1) + 标准化 Student-t 新息(MLE, scipy.optimize):
    r_t = μ + ε_t,   ε_t = σ_t z_t,   z_t ~ 标准化 t(ν)  (单位方差)
    σ²_t = ω + α ε²_{t-1} + γ ε²_{t-1}·1[ε_{t-1}<0] + β σ²_{t-1}
  γ = 杠杆项(负冲击抬波动更多), ν = 厚尾。参数 θ=[μ,ω,α,γ,β,ν] 共 6 个/通道。
  平稳约束 α + γ/2 + β < 1 (E[1[ε<0]]=1/2)。为数值稳健, 先把每通道缩放到单位方差再拟合。

================================ FHS 生成 ================================
1) MLE 拟合两通道 → 取标准化残差 z^sp_t, z^dgs_t;
2) 【联合自助】每个 (路径,步) 抽【同一时间索引 τ】, 同时取 (z^sp_τ, z^dgs_τ) 两通道残差
   → 保 SP500↔DGS10 同期相关(签名 level-2 对称项); iid 抽样会丢跨通道 lead-lag(反对称
   /Lévy area), 但日频股债 lead-lag 很弱, 主风险是长程 vol-ACF 几何 vs 双曲(见 §pitfall);
3) 各通道 GARCH 递归注回条件波动 → r̂_t = μ + σ̂_t ẑ_t;
   初始 σ²_0 从【历史拟合方差经验分布】随机抽 → 跨路径异质起始 regime(增强 regime 多样性)。
无接缝(递归连续) ⇒ 可生成任意 L(默认 2048, 对标全长真实窗 + 展示无拼接长程)。

================================ 用法 ================================
  conda run -n ts_diffusion python eval/fhs_baseline.py --selftest         # 拟合+自检(不导出)
  conda run -n ts_diffusion python eval/fhs_baseline.py --n 5120 --L 2048 \
        --output output/fhs_baseline_L2048.csv                              # 生成候选 CSV
  conda run -n ts_diffusion python forensic_suite.py --candidate output/fhs_baseline_L2048.csv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import skew as _skew, kurtosis as _kurtosis

REAL_DEFAULT = "/home/u00134/data/train_sp500_us10y.csv"
CHANNELS = ("sp500", "DGS10")          # 列名(真实 CSV 长表)
OUT_PREFIX = ("sp500", "dgs10")        # 输出宽表列前缀(对齐 diagnostics.load_changes)


# ════════════════════════════════ 标准化 Student-t ════════════════════════════════
def std_t_logpdf(z, nu):
    """单位方差 Student-t(ν>2) 的对数密度。"""
    c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    return c - ((nu + 1.0) / 2.0) * np.log1p(z * z / (nu - 2.0))


# ════════════════════════════════ GJR-GARCH(1,1) 滤波 ════════════════════════════════
def gjr_filter(params, x):
    """给定参数与(已缩放)序列 x, 返回 (σ²_t 序列, ε_t 序列)。σ²_0 = 样本方差。"""
    mu, omega, alpha, gamma, beta, _nu = params
    n = len(x)
    eps = x - mu
    s2 = np.empty(n)
    s2[0] = np.var(x)
    e2 = eps * eps
    neg = (eps < 0.0).astype(np.float64)
    for t in range(1, n):
        s2[t] = omega + alpha * e2[t - 1] + gamma * e2[t - 1] * neg[t - 1] + beta * s2[t - 1]
    return s2, eps


def _nll(params, x):
    mu, omega, alpha, gamma, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or nu <= 2.001 or (alpha + gamma / 2.0 + beta) >= 0.999:
        return 1e12
    s2, eps = gjr_filter(params, x)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e12
    z = eps / np.sqrt(s2)
    ll = std_t_logpdf(z, nu) - 0.5 * np.log(s2)
    if not np.isfinite(ll.sum()):
        return 1e12
    return -float(ll.sum())


def fit_gjr_t(r, label="", persist_cap=0.99):
    """对【原始量级】收益 r 拟合 GJR-GARCH(1,1)-t。内部缩放到单位方差以助优化。
    persist_cap: 平稳性上限 α+γ/2+β < persist_cap (默认 0.99, 防近-IGARCH 仿真发散;
    实测 DGS10 MLE 会顶到 1.0 → 须钳, 否则条件方差随机游走致峰度爆 130+)。
    返回 dict: 原始量级参数 + 标准化残差 z + 历史条件波动(原始量级) + 拟合诊断。"""
    s = float(r.std())                                   # 缩放因子 → 单位方差
    x = (r - r.mean()) / s                               # 缩放序列(零均值单位方差)
    x_mu0 = float(r.mean()) / s                          # 真实均值映到缩放空间(供下方还原)

    # 初值 + 边界 + 平稳约束(SLSQP)
    p0 = np.array([x.mean(), 0.05, 0.05, 0.05, 0.85, 7.0])
    bounds = [(-1, 1), (1e-6, 5), (0.0, 1.0), (-0.5, 1.0), (0.0, persist_cap), (2.1, 60.0)]
    cons = [{"type": "ineq", "fun": lambda p: persist_cap - (p[2] + p[3] / 2.0 + p[4])}]
    best = None
    for nu0 in (5.0, 8.0, 12.0):                          # 多起点取最优(ν 初值敏感)
        p0[5] = nu0
        try:
            res = minimize(_nll, p0, args=(x,), method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
            if best is None or (res.fun < best.fun):
                best = res
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"[fhs] {label} GJR-GARCH MLE 失败")
    p = best.x
    s2, eps = gjr_filter(p, x)
    z = eps / np.sqrt(s2)                                 # 标准化残差(尺度无关)
    mu_x, omega, alpha, gamma, beta, nu = p
    persist = alpha + gamma / 2.0 + beta
    return {
        "label": label, "scale": s, "mu_real": mu_x * s,
        "omega": float(omega), "alpha": float(alpha), "gamma": float(gamma),
        "beta": float(beta), "nu": float(nu), "persistence": float(persist),
        "uncond_var_x": float(omega / max(1e-9, 1.0 - persist)),    # 缩放空间无条件方差
        "sigma2_x_max": float(s2.max()),                  # 历史最大条件方差(缩放空间) → 仿真 σ² 钳位界
        "z": z, "sigma_real": np.sqrt(s2) * s,            # 历史条件波动(还原原始量级)
        "sigma2_x_hist": s2,                              # 缩放空间历史条件方差(供初始 σ² 抽样)
        "nll": float(best.fun), "n": int(len(r)),
        "resid_std": float(z.std()), "resid_kurt": float(_kurtosis(z)),
        "resid_skew": float(_skew(z)),
    }


# ════════════════════════════════ FHS 联合自助生成 ════════════════════════════════
def fhs_generate(fits, idx_pool_len, N, L, seed=42, burn=256, clamp_histmax=1.0):
    """fits = [fit_sp, fit_dgs]; 返回 (N, 2, L) 原始量级路径。
    联合自助: 每 (path,step) 抽同一 τ, 两通道共享 → 保同期跨通道相关。
    稳态化(防近-IGARCH 仿真发散): ① 起始 σ²=无条件方差; ② burn 步预热后才记录(让路径
    扩散进稳态分布, 去除固定起点伪影); ③ 条件方差钳位 σ²_t ≤ clamp_histmax×【历史最大条件
    方差】—— 数据驱动上界(可重访最恶劣历史 regime 但不超越), 只杀"仅仿真"的波动失控级联,
    保全部真实 regime。clamp_histmax=None 关钳位。"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, idx_pool_len, size=(N, L + burn))   # 共享时间索引(联合自助核心)
    out = np.empty((N, 2, L), dtype=np.float64)
    for ch, f in enumerate(fits):
        s = f["scale"]; mu_x = f["mu_real"] / s
        omega, alpha, beta, gamma = f["omega"], f["alpha"], f["beta"], f["gamma"]
        zdraw = f["z"][idx]                               # (N, L+burn) 抽到的残差(共享 idx)
        cap = (clamp_histmax * f["sigma2_x_max"]) if clamp_histmax else np.inf
        s2_prev = np.full(N, f["uncond_var_x"], dtype=np.float64)   # 稳态起点(同一无条件方差)
        x = np.empty((N, L + burn), dtype=np.float64)
        eps_prev = np.sqrt(s2_prev) * zdraw[:, 0]
        x[:, 0] = mu_x + eps_prev
        for t in range(1, L + burn):
            e2 = eps_prev * eps_prev
            s2_t = omega + alpha * e2 + gamma * e2 * (eps_prev < 0.0) + beta * s2_prev
            if clamp_histmax:
                s2_t = np.minimum(s2_t, cap)
            eps_t = np.sqrt(s2_t) * zdraw[:, t]
            x[:, t] = mu_x + eps_t
            eps_prev = eps_t; s2_prev = s2_t
        out[:, ch, :] = x[:, burn:] * s                   # 丢 burn 预热段 + 还原原始量级
    return out


# ════════════════════════════════ I/O ════════════════════════════════
def load_real(csv_path):
    df = pd.read_csv(csv_path)
    return df["sp500"].values.astype(np.float64), df["DGS10"].values.astype(np.float64)


def export_wide(paths, output):
    """(N,2,L) → 宽表 CSV: sp500_0..sp500_{L-1}, dgs10_0..dgs10_{L-1} (对齐 load_changes)。"""
    N, _, L = paths.shape
    blocks, cols = [], []
    for ch, pre in enumerate(OUT_PREFIX):
        blocks.append(paths[:, ch, :])
        cols += [f"{pre}_{i}" for i in range(L)]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols).to_csv(output, index=False)
    return output


# ════════════════════════════════ 自检 ════════════════════════════════
def selftest(fits, real, gen):
    print("\n" + "=" * 74)
    print("  FHS 自检 — 拟合参数 / 残差 / 生成统计 vs 真实")
    print("=" * 74)
    names = ["sp500", "DGS10"]
    for f, nm in zip(fits, names):
        print(f"  [{nm}] μ={f['mu_real']:.2e} ω={f['omega']:.4f} α={f['alpha']:.4f} "
              f"γ={f['gamma']:.4f} β={f['beta']:.4f} ν={f['nu']:.2f}  persist={f['persistence']:.4f}")
        print(f"        残差: std={f['resid_std']:.3f}(应≈1) skew={f['resid_skew']:.3f} "
              f"exkurt={f['resid_kurt']:.2f}  NLL={f['nll']:.1f}")
    print("  " + "-" * 70)
    print(f"  {'统计量':<16}{'真实 sp500':>14}{'生成 sp500':>14}{'真实 DGS10':>14}{'生成 DGS10':>14}")
    for nm, fn in [("std", lambda a: a.std()), ("skew", lambda a: _skew(a.ravel())),
                   ("exkurt", lambda a: _kurtosis(a.ravel()))]:
        rs, gs = real[0], gen[:, 0, :]
        rd, gd = real[1], gen[:, 1, :]
        print(f"  {nm:<16}{fn(rs):>14.4f}{fn(gs):>14.4f}{fn(rd):>14.4f}{fn(gd):>14.4f}")
    print("=" * 74 + "\n")


# ════════════════════════════════ main ════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="E0 FHS/GJR-GARCH-t 零训练基线")
    ap.add_argument("--real", default=REAL_DEFAULT)
    ap.add_argument("--n", type=int, default=5120, help="生成路径条数")
    ap.add_argument("--L", type=int, default=2048, help="每条路径长度(FHS 无接缝, 任意 L)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--burn", type=int, default=256, help="仿真预热步数(稳态化)")
    ap.add_argument("--clamp-histmax", type=float, default=1.0,
                    help="条件方差钳位 = 该倍×历史最大条件方差(数据驱动稳态界; 0/None 关)")
    ap.add_argument("--persist-cap", type=float, default=0.99, help="GARCH 平稳性上限(防近-IGARCH)")
    ap.add_argument("--output", default="output/fhs_baseline_L2048.csv")
    ap.add_argument("--selftest", action="store_true", help="仅拟合+自检, 不导出")
    ap.add_argument("--params-out", default=None, help="拟合参数 JSON 落盘路径")
    args = ap.parse_args()

    t0 = time.time()
    sp, dg = load_real(args.real)
    print(f"[fhs] 真实数据 {args.real}: sp500 {len(sp)} 点 / DGS10 {len(dg)} 点")
    fit_sp = fit_gjr_t(sp, "sp500", persist_cap=args.persist_cap)
    fit_dg = fit_gjr_t(dg, "DGS10", persist_cap=args.persist_cap)
    fits = [fit_sp, fit_dg]
    clamp = args.clamp_histmax if args.clamp_histmax and args.clamp_histmax > 0 else None
    T = min(len(fit_sp["z"]), len(fit_dg["z"]))           # 残差池长度(联合自助索引上界)
    # 对齐残差池(两通道同长, 同期配对)
    fit_sp["z"] = fit_sp["z"][:T]; fit_dg["z"] = fit_dg["z"][:T]
    print(f"[fhs] 拟合完成 ({time.time()-t0:.1f}s); 残差池 T={T}")

    N_self = min(2048, args.n)
    gen_self = fhs_generate(fits, T, N_self, min(args.L, 2048), seed=args.seed,
                            burn=args.burn, clamp_histmax=clamp)
    selftest(fits, (sp, dg), gen_self)

    if args.params_out:
        dump = {f["label"]: {k: v for k, v in f.items()
                             if k not in ("z", "sigma_real", "sigma2_x_hist")} for f in fits}
        os.makedirs(os.path.dirname(os.path.abspath(args.params_out)), exist_ok=True)
        json.dump(dump, open(args.params_out, "w"), indent=2, ensure_ascii=False)
        print(f"[fhs] 参数 → {args.params_out}")

    if not args.selftest:
        paths = fhs_generate(fits, T, args.n, args.L, seed=args.seed,
                             burn=args.burn, clamp_histmax=clamp)
        export_wide(paths, args.output)
        print(f"[fhs] {args.n} 条 ×{args.L} (2通道) → {args.output} (总 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
