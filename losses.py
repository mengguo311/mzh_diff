"""
losses.py — Phase 2 (v10) 训练损失增强

针对 v9 诊断出的"过平滑 / 欠离散"问题 (生成路径比真实更平滑、波动聚集弱、
波动爆发不足)，在标准 ε-MSE 之外增加两类机制：

  1. min-SNR-γ 加权 (ε-prediction)：抑制低噪声步 (高 SNR) 对损失的过度主导，
     加速收敛、提升保真度 (Hang et al., 2023)。
  2. stylized-fact 辅助损失：由 ε̂ 反演 x̂₀，惩罚其与真实窗口在
       (a) |r| 自相关 (波动聚集) 与
       (b) 二阶差分能量 (roughness)
     上的差距 —— 直接攻击 ε-MSE 均值寻优忽略的高阶结构。

辅助损失仅在低噪声步 (ᾱ_t 较大、x̂₀ 可靠) 施加，并按 ᾱ_t 逐样本加权。
所有计算在标准化空间进行 (x̂₀ 与 x₀ 同空间)，与现有训练一致。
"""

import torch


def min_snr_weight(scheduler, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """
    ε-prediction 的 min-SNR-γ 逐样本权重: min(SNR_t, γ)/SNR_t = min(1, γ/SNR_t)。
    Returns: (B,)
    """
    abar = scheduler.alphas_cumprod.gather(0, t.long())          # (B,)
    snr = abar / (1.0 - abar).clamp(min=1e-8)                    # 信噪比 SNR_t = ᾱ/(1-ᾱ)
    gamma_t = torch.full_like(snr, gamma)
    return torch.minimum(snr, gamma_t) / snr.clamp(min=1e-8)


def recover_x0(scheduler, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """由噪声预测反演干净样本: x̂₀ = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t。xt/eps: (B,2,L)。"""
    sqrt_ab = scheduler._extract(scheduler.sqrt_alphas_cumprod, t, xt.shape)            # (B,1,1)
    sqrt_1m = scheduler._extract(scheduler.sqrt_one_minus_alphas_cumprod, t, xt.shape)  # (B,1,1)
    return (xt - sqrt_1m * eps) / sqrt_ab


def _abs_acf(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """|x| 的 lag 1..max_lag 自相关。x: (B,L) → (B,max_lag)。可微。"""
    a = x.abs()
    a = a - a.mean(dim=1, keepdim=True)
    var = (a * a).mean(dim=1).clamp(min=1e-8)                    # (B,)
    cols = []
    for lag in range(1, max_lag + 1):
        cov = (a[:, lag:] * a[:, :-lag]).mean(dim=1)            # (B,)
        cols.append(cov / var)
    return torch.stack(cols, dim=1)                             # (B, max_lag)


def _d2_energy(x: torch.Tensor) -> torch.Tensor:
    """二阶差分能量 (roughness 指标)。x: (B,L) → (B,)。"""
    d2 = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    return (d2 * d2).mean(dim=1)


def stylized_aux(x0_hat: torch.Tensor, x0: torch.Tensor, max_lag: int = 5):
    """
    逐样本 stylized-fact 差距。
      x0_hat, x0: (B, C, L) — 标准化空间
    Returns:
      acf_l1:  (B,)  两通道 |r| ACF 的 L1 差距 (无量纲)
      rough_l: (B,)  sp 通道二阶差分能量的相对差距 (无量纲)
    """
    C = x0.shape[1]
    acf = x0.new_zeros(x0.shape[0])
    for ch in range(C):
        acf = acf + (_abs_acf(x0_hat[:, ch, :], max_lag)
                     - _abs_acf(x0[:, ch, :], max_lag)).abs().mean(dim=1)
    acf = acf / C

    d2_hat = _d2_energy(x0_hat[:, 0, :])
    d2_real = _d2_energy(x0[:, 0, :])
    rough = (d2_hat - d2_real).abs() / (d2_real + 1e-6)         # 相对差距，尺度无关
    return acf, rough


# ──────────────────────────────────────────────
# v11: 可微 Sig-MMD 辅助损失 (depth-2 向量化截断签名 + 无偏 RBF-MMD)
# ──────────────────────────────────────────────
# 路径签名是随机过程"律"的可微指纹: 同一散度 D(P_real, Q) 既是生成器的训练损失,
# 又是 eval/signature.py 取证打分器的核。这里用 depth-2 截断签名 (便宜/可微/无 python
# 循环, 非移植 eval 的 numpy Chen 循环), 在随机短子窗上对 x̂₀ 与真实 x₀ 的签名分布做
# 无偏 MMD²。仅在低噪声步 (ᾱ_t 可靠、x̂₀ 反演可信) 施加 (门控在 train.py)。
# 可微性铁律: 损失路径内不得出现 .item()/numpy/data-dependent 整数索引/非带宽 detach。

def _path_signature_depth2(path: torch.Tensor) -> torch.Tensor:
    """
    depth-2 截断路径签名 (向量化, 无 python 循环, 可微)。
      path: (B, n, d) — d 维路径的 n 个采样点 (path 即"通道值序列", 内部再差分),
                        与 eval/signature.py 同一约定。
      return: (B, d + d*d) — [level-1 (d), level-2 (d*d 拉平)]。

    数学 (Chen identity 的左点 Riemann 离散, 与 eval/signature.py 一致, 但 cumsum/einsum 向量化):
      dX      = path[1:] - path[:-1]                 # 增量
      S1_i    = Σ_t dX_i(t)                          # level-1: 路径净增量
      prefix  = cumsum(dX) - dX                      # 排他前缀和 (当前步之前的累计增量)
      S2_{ij} = Σ_t prefix_i(t) · dX_j(t)            # level-2: 迭代积分 ∫∫_{s<t} dX_i dX_j
    """
    dX = path[:, 1:, :] - path[:, :-1, :]                 # (B, n-1, d)
    S1 = dX.sum(dim=1)                                     # (B, d)
    prefix = torch.cumsum(dX, dim=1) - dX                 # (B, n-1, d) 排他前缀和
    S2 = torch.einsum('bti,btj->bij', prefix, dX)         # (B, d, d)
    B = path.shape[0]
    return torch.cat([S1, S2.reshape(B, -1)], dim=1)      # (B, d + d^2)


def _rbf_mmd2(sig_real: torch.Tensor, sig_fake: torch.Tensor) -> torch.Tensor:
    """
    无偏 MMD² (RBF 核, median-heuristic 带宽)。
      sig_real: (Na, D) — 真实签名 (应已 detach, 作为目标常量)
      sig_fake: (Nb, D) — 假签名 (x̂₀ 的函数, 保留梯度)
    带宽用 .detach() 的距离平方中位数 (否则带宽随梯度漂移)。可微 wrt sig_fake。
    无偏估计可能略 < 0 (合法), 直接返回供 backward, 勿 clamp 以免截断梯度。
    """
    Z = torch.cat([sig_real, sig_fake], dim=0)                 # (M, D)
    sq = (Z * Z).sum(dim=1)                                    # (M,)
    d2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (Z @ Z.t())).clamp(min=0.0)  # (M,M)
    M = Z.shape[0]
    iu = torch.triu_indices(M, M, offset=1, device=Z.device)
    med = d2[iu[0], iu[1]].detach().median().clamp(min=1e-12)  # 带宽 detach (不随梯度漂)
    K = torch.exp(-d2 / med)
    na = sig_real.shape[0]
    nb = sig_fake.shape[0]
    Kaa, Kbb, Kab = K[:na, :na], K[na:, na:], K[:na, na:]
    saa = (Kaa.sum() - Kaa.diagonal().sum()) / (na * (na - 1))  # 排除对角 (无偏)
    sbb = (Kbb.sum() - Kbb.diagonal().sum()) / (nb * (nb - 1))
    sab = Kab.mean()
    return saa + sbb - 2.0 * sab


def sig_mmd_loss(
    x0_hat: torch.Tensor,
    x0: torch.Tensor,
    n_sub: int = 2,
    l_sub: int = 128,
    scale_sp: float = 1.0,
    scale_dg: float = 1.0,
) -> torch.Tensor:
    """
    可微 Sig-MMD 辅助损失: 在随机短子窗上, 对比 x̂₀ 与真实 x₀ 的 (time-augmented)
    depth-2 路径签名分布之差 (无偏 MMD²)。

      x0_hat, x0: (B, C, L) — 标准化(z-score)空间, C=2 (sp, dgs10)。

    通道处理: x0 已逐通道 z-score → 两通道天然同尺度 (≈unit std), 因此 scale_*=1.0 即等价于
      handoff 的 "按真实数据全局 std 归一化" (raw std 在 z 空间映射为 ≈1; 若改用 raw std
      0.011/0.069 反而会把两通道尺度拉到 6:1 失衡)。再拼接 time∈[0,1] 做 time-augmentation,
      以捕捉增量的时间不对称 (Lévy area)。
    签名特征按【真实签名逐维 (mean,std)】(detach) 标准化, 使异尺度的 level-1/level-2 在单带宽
      RBF 下可比, 同时保持对 x̂₀ 可微 (real 统计是常量)。
    子窗起点用 torch.randint 随机 (非数据依赖) → 切片对 x̂₀ 可微, 不破坏梯度。
    """
    B, C, L = x0_hat.shape
    device = x0_hat.device
    if L < l_sub + 1 or B < 2:
        return x0_hat.new_zeros(())

    tcol = torch.linspace(0.0, 1.0, l_sub, device=device)              # (l_sub,) 时间增广
    t_exp = tcol.unsqueeze(0).expand(B, -1)                            # (B, l_sub)
    # 随机(非数据依赖)起点; 在 CPU 上取整, 不引入 GPU 同步, 仅作常量切片索引
    starts = [int(s) for s in torch.randint(0, L - l_sub + 1, (n_sub,))]

    fake_sigs, real_sigs = [], []
    for s in starts:
        sl = slice(s, s + l_sub)
        path_f = torch.stack([x0_hat[:, 0, sl] / scale_sp,
                              x0_hat[:, 1, sl] / scale_dg, t_exp], dim=-1)  # (B, l_sub, 3)
        path_r = torch.stack([x0[:, 0, sl] / scale_sp,
                              x0[:, 1, sl] / scale_dg, t_exp], dim=-1)
        fake_sigs.append(_path_signature_depth2(path_f))
        real_sigs.append(_path_signature_depth2(path_r))

    sig_fake = torch.cat(fake_sigs, dim=0)                            # (B*n_sub, D)
    sig_real = torch.cat(real_sigs, dim=0).detach()                  # 真实作目标常量

    # 逐维标准化 (用真实统计, detach): 异尺度 level 在单带宽 RBF 下可比, 仍 wrt fake 可微
    mu = sig_real.mean(dim=0, keepdim=True)
    sd = sig_real.std(dim=0, keepdim=True).clamp(min=1e-8)
    sig_fake = (sig_fake - mu) / sd
    sig_real = (sig_real - mu) / sd

    return _rbf_mmd2(sig_real, sig_fake)
