"""
eval/signature.py — 路径签名 Sig-MMD 假数据鉴别器 (自实现截断签名, 零依赖)

非自指: 在真实 vs candidate 的子窗"路径签名"特征上做 kernel-MMD 两样本检验。
路径签名是随机过程律的可微"指纹"; 这里实现 time-augmented depth-3 截断签名
(Chen identity), 不依赖 iisignature/signatory/sigkernel。

要点 (遵循 roadmap 警示):
  - 不对整条 L=2048 取签名 (会被净增量主导、对粗糙度盲视) → 随机裁剪短子窗;
  - 通道按真实数据全局 std 归一化 (保留波动幅度信息, 又使各通道可比);
  - MMD 用预计算核矩阵 + 标签置换得经验 p 值; 同源 (real-vs-real) 标定应不显著。

用法:
  conda run -n ts_diffusion python eval/signature.py \
    --real /home/u00134/data/train_sp500_us10y.csv \
    --fakes v9=output/deep_v9_20k.csv v10_sampling=output/deep_v10.csv v10_retrained=output/deep_v10_retrained.csv \
    --json eval/signature_results.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnostics import load_changes  # noqa: E402


# ──────────────────────────────────────────────
# 截断签名 (tensor-exp + Chen product)
# ──────────────────────────────────────────────
def _tensor_exp(dx: np.ndarray, depth: int):
    """单段增量 dx 的截断 tensor 指数: level_k = dx^{⊗k}/k!。返回 [1, lvl1, ..., lvl_depth]。"""
    terms = [np.array(1.0)]
    cur = np.array(1.0)
    for k in range(1, depth + 1):
        cur = np.tensordot(cur, dx, axes=0) / k
        terms.append(cur)
    return terms


def _chen(A, B, depth: int):
    """Chen 拼接: C_k = sum_{i=0}^k A_i ⊗ B_{k-i}。"""
    C = []
    for k in range(depth + 1):
        acc = None
        for i in range(k + 1):
            term = np.tensordot(A[i], B[k - i], axes=0)
            acc = term if acc is None else acc + term
        C.append(acc)
    return C


def signature(path: np.ndarray, depth: int) -> np.ndarray:
    """path: (n, d) → 截断签名 (level 1..depth, 拉平)。"""
    d = path.shape[1]
    dX = np.diff(path, axis=0)
    sig = [np.array(1.0)] + [np.zeros((d,) * k) for k in range(1, depth + 1)]
    for dx in dX:
        sig = _chen(sig, _tensor_exp(dx, depth), depth)
    return np.concatenate([sig[k].ravel() for k in range(1, depth + 1)])


# ──────────────────────────────────────────────
# 子窗签名特征
# ──────────────────────────────────────────────
def sig_features(windows, scale, n_windows, k_sub, l_sub, depth, seed):
    """从 windows (N,2,L) 采样子窗, time-augment + 全局 std 归一化, 算签名集合 (M, sigdim)。"""
    rng = np.random.default_rng(seed)
    N, _, L = windows.shape
    idx = rng.choice(N, size=min(n_windows, N), replace=False)
    t_col = (np.arange(l_sub) / (l_sub - 1)).reshape(-1, 1)
    out = []
    for wi in idx:
        sp = windows[wi, 0, :] / scale[0]
        dg = windows[wi, 1, :] / scale[1]
        for _ in range(k_sub):
            st = rng.integers(0, L - l_sub + 1)
            path = np.column_stack([sp[st:st + l_sub], dg[st:st + l_sub], t_col[:, 0]])
            out.append(signature(path, depth))
    return np.asarray(out)


# ──────────────────────────────────────────────
# RBF-MMD + 置换检验
# ──────────────────────────────────────────────
def _rbf_kernel(Z):
    sq = np.sum(Z * Z, 1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    iu = np.triu_indices(len(Z), 1)
    med = np.median(d2[iu]) + 1e-12        # median heuristic (距离平方中位数)
    return np.exp(-d2 / med)


def _mmd2(K, na):
    """无偏 MMD^2, 前 na 行为 A, 其余为 B。"""
    Kaa, Kbb, Kab = K[:na, :na], K[na:, na:], K[:na, na:]
    na_, nb_ = na, K.shape[0] - na
    saa = (Kaa.sum() - np.trace(Kaa)) / (na_ * (na_ - 1))
    sbb = (Kbb.sum() - np.trace(Kbb)) / (nb_ * (nb_ - 1))
    sab = Kab.mean()
    return saa + sbb - 2 * sab


def sig_mmd_test(A, B, n_perm=300, seed=0):
    """A=real sigs, B=candidate sigs。标准化→RBF→MMD^2 + 置换 p。"""
    Z = np.vstack([A, B])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)    # 标准化签名特征
    K = _rbf_kernel(Z)
    na = len(A)
    obs = _mmd2(K, na)
    rng = np.random.default_rng(seed)
    n = K.shape[0]
    ge = 1
    for _ in range(n_perm):
        p = rng.permutation(n)
        Kp = K[np.ix_(p, p)]
        if _mmd2(Kp, na) >= obs:
            ge += 1
    return {"mmd2": float(obs), "p_value": float(ge / (n_perm + 1)),
            "n_a": int(len(A)), "n_b": int(len(B))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="/home/u00134/data/train_sp500_us10y.csv")
    ap.add_argument("--fakes", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--json", default=None)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-windows", type=int, default=300)
    ap.add_argument("--k-sub", type=int, default=2)
    ap.add_argument("--l-sub", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=300)
    args = ap.parse_args()

    L = load_changes(args.fakes[0].split("=", 1)[1]).shape[-1]
    real = load_changes(args.real, target_seq_len=L)
    scale = np.array([real[:, 0, :].std(), real[:, 1, :].std()]) + 1e-12
    print(f"[sig] L={L}, real {real.shape[0]} 窗口, depth={args.depth}, "
          f"子窗 {args.k_sub}x{args.l_sub}, 全局 std scale={scale.round(5)}")

    # real 签名集合 (用两个不同 seed 切两份, 一份作 real-vs-real 标定)
    A_real = sig_features(real, scale, args.n_windows, args.k_sub, args.l_sub, args.depth, seed=1)
    A_real2 = sig_features(real, scale, args.n_windows, args.k_sub, args.l_sub, args.depth, seed=2)
    print(f"[sig] real 签名 {A_real.shape}")

    results = {"L": L, "depth": args.depth, "calibration": {}, "detect": {}}
    cal = sig_mmd_test(A_real, A_real2, n_perm=args.n_perm, seed=10)
    results["calibration"]["real_vs_real"] = cal

    rows = []
    for spec in args.fakes:
        label, path = spec.split("=", 1)
        fake = load_changes(path)
        B = sig_features(fake, scale, args.n_windows, args.k_sub, args.l_sub, args.depth, seed=3)
        r = sig_mmd_test(A_real, B, n_perm=args.n_perm, seed=11)
        results["detect"][label] = r
        rows.append((label, r))

    print(f"\n标定 real-vs-real: MMD^2={cal['mmd2']:.3e}  p={cal['p_value']:.3f}  (应不显著, p 大)")
    print(f"\n{'candidate':16s} {'Sig-MMD^2':>12s} {'p_value':>9s}   越低越真实 / p 越大越像真")
    print("-" * 62)
    for label, r in sorted(rows, key=lambda kv: kv[1]["mmd2"]):
        flag = "  <- 检出为假" if r["p_value"] < 0.05 else ""
        print(f"{label:16s} {r['mmd2']:12.3e} {r['p_value']:9.3f}{flag}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[json] 已保存 {args.json}")


if __name__ == "__main__":
    main()
