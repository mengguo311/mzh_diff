"""
metrics.py — 高级概率分布度量 (Wasserstein 距离 & MMD)
所有计算完全使用 PyTorch 在 GPU 上矢量化执行，严禁使用循环或 .cpu() 导致性能恶化。
"""

import torch

def interpolate_sorted(sorted_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    对已排序的 1D 张量进行线性插值，以匹配目标长度。
    
    Args:
        sorted_tensor: (N,) 已排序的一维张量
        target_len: 目标长度
    Returns:
        (target_len,) 插值对齐后的一维张量
    """
    N = len(sorted_tensor)
    if N == target_len:
        return sorted_tensor
    
    # 创建 [0, N-1] 之间的均匀分布索引
    indices = torch.linspace(0, N - 1, target_len, device=sorted_tensor.device)
    
    # 线性插值计算
    idx_low = indices.floor().long()
    idx_high = indices.ceil().long()
    weight = indices - idx_low
    
    return torch.lerp(sorted_tensor[idx_low], sorted_tensor[idx_high], weight)


def calculate_1d_wasserstein(x_real: torch.Tensor, x_fake: torch.Tensor) -> float:
    """
    计算两个 1D 分布之间的 Wasserstein-1 距离（L1 意义下）。
    输入可以是任意维度的张量，会在内部展平为一维分布。
    
    Args:
        x_real: 真实数据分布张量
        x_fake: 生成数据分布张量
    Returns:
        float: Wasserstein-1 距离值
    """
    # 展平为 1D
    r = x_real.flatten()
    f = x_fake.flatten()

    # 排序
    r_sorted = torch.sort(r)[0]
    f_sorted = torch.sort(f)[0]

    # 插值对齐样本大小
    if len(r_sorted) != len(f_sorted):
        target_len = max(len(r_sorted), len(f_sorted))
        r_sorted = interpolate_sorted(r_sorted, target_len)
        f_sorted = interpolate_sorted(f_sorted, target_len)

    # 计算 L1 均值差异
    w_dist = torch.mean(torch.abs(f_sorted - r_sorted))
    return w_dist.item()


def compute_pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两个特征矩阵 x (M, D) 和 y (N, D) 之间的成对平方欧氏距离。
    使用公式: ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * <x_i, y_j>
    
    Args:
        x: (M, D) 特征矩阵
        y: (N, D) 特征矩阵
    Returns:
        (M, N) 距离矩阵
    """
    x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)        # (M, 1)
    y_sq = torch.sum(y ** 2, dim=-1, keepdim=True).t()    # (1, N)
    xy = torch.matmul(x, y.t())                           # (M, N)
    dist = x_sq + y_sq - 2 * xy
    return torch.clamp(dist, min=0.0)


def calculate_mmd(x_real: torch.Tensor, x_fake: torch.Tensor) -> float:
    """
    计算真实样本与生成样本特征之间的最大均值差异 (MMD)，采用高斯 (RBF) 核。
    使用中位数启发式 (Median Heuristic) 自动估计 RBF 核的带宽 (Bandwidth)。
    
    Args:
        x_real: (M, D) 真实样本特征矩阵
        x_fake: (N, D) 生成样本特征矩阵
    Returns:
        float: MMD 距离值
    """
    device = x_real.device
    M = x_real.shape[0]
    N = x_fake.shape[0]

    # 合并样本用于中位数启发式估计核带宽
    combined = torch.cat([x_real, x_fake], dim=0)
    n_samples = combined.shape[0]
    
    # 限制样本数量在 1000 以内计算成对距离，防止 OOM
    if n_samples > 1000:
        indices = torch.randperm(n_samples, device=device)[:1000]
        combined_sub = combined[indices]
    else:
        combined_sub = combined

    # 计算子集内部的成对平方欧氏距离
    sq_dists = compute_pairwise_sq_dist(combined_sub, combined_sub) # (K, K)
    
    # 提取上三角元素（不包含自距离的 0 元素）
    triu_indices = torch.triu_indices(sq_dists.shape[0], sq_dists.shape[1], offset=1, device=device)
    pairwise_sq_vals = sq_dists[triu_indices[0], triu_indices[1]]
    
    # 中位数估计
    median_sq = torch.median(pairwise_sq_vals).item()
    if median_sq == 0:
        median_sq = torch.mean(pairwise_sq_vals).item()
    if median_sq == 0:
        median_sq = 1.0

    gamma = 1.0 / median_sq

    # 计算三组核矩阵
    d_xx = compute_pairwise_sq_dist(x_real, x_real)
    d_yy = compute_pairwise_sq_dist(x_fake, x_fake)
    d_xy = compute_pairwise_sq_dist(x_real, x_fake)

    k_xx = torch.exp(-gamma * d_xx)
    k_yy = torch.exp(-gamma * d_yy)
    k_xy = torch.exp(-gamma * d_xy)

    # 计算 MMD 平方值
    mmd_sq = torch.sum(k_xx) / (M * M) - 2 * torch.sum(k_xy) / (M * N) + torch.sum(k_yy) / (N * N)
    
    # 截断开方，防止浮点误差导致负值开根号
    mmd = torch.sqrt(torch.clamp(mmd_sq, min=0.0)).item()
    return mmd
