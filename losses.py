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
