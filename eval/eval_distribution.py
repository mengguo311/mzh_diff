#!/usr/bin/env python3
"""
eval_distribution.py — 评估分数分布直方图绘制程序
将真实历史数据以 1260 天滑动窗口切分并打分，绘制真实基准分布，并支持叠加第三方生成数据的分数分布，直观展现生成质量的差异。
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 确保 src 目录在 python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from eval.score import FinancialScorer


@torch.no_grad()
def compute_individual_ddpm_mses(scorer: FinancialScorer, x_normalized: torch.Tensor, t_eval: int = 200, batch_size: int = 64) -> np.ndarray:
    """
    高效计算 Batch 中每一条路径单独的 DDPM MSE Loss。
    返回: (B,) 形状的 NumPy 数组，对应每条路径的 MSE。
    """
    scorer.model.eval()
    B = x_normalized.shape[0]
    L = x_normalized.shape[2]
    
    # 核心逻辑：确保输入序列长度是 8 的倍数，以配合 U-Net
    if L % 8 != 0:
        L_clean = (L // 8) * 8
        x_input = x_normalized[:, :, :L_clean]
    else:
        x_input = x_normalized
        
    mses = []
    for i in range(0, B, batch_size):
        x0 = x_input[i:i+batch_size].to(scorer.device)
        n_batch = x0.shape[0]
        
        t = torch.full((n_batch,), t_eval, device=scorer.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        
        c = x0[:, :, 0]
        xt = scorer.scheduler.q_sample(x0, t, noise)
        noise_pred = scorer.model(xt, t, c)
        
        # 计算每个样本独立的 MSE: (n_batch, 2, L_clean)
        sq_error = (noise_pred - noise) ** 2
        # 在通道和时间维度上求平均: (n_batch,)
        batch_mses = sq_error.mean(dim=(1, 2)).cpu().numpy()
        mses.extend(batch_mses)
        
    return np.array(mses)


def score_dataset_paths(scorer: FinancialScorer, x_norm: torch.Tensor, x_raw: np.ndarray, base_facts: dict, base_mse: float, t_eval: int = 200) -> list[float]:
    """
    对数据集中每一条路径独立进行打分。
    返回: 分数列表。
    """
    B = x_norm.shape[0]
    print(f"  Calculating path-wise DDPM MSE for {B} samples...")
    individual_mses = compute_individual_ddpm_mses(scorer, x_norm, t_eval=t_eval)
    
    print(f"  Calculating path-wise Stylized Facts for {B} samples...")
    scores = []
    start_time = time.time()
    
    for i in range(B):
        # 提取单条路径数据: (1, 2, L)
        path_raw = x_raw[i:i+1]
        path_mse = individual_mses[i]
        
        # 计算该条路径的金融典型事实
        path_facts = scorer.compute_stylized_facts(path_raw)
        
        # 打分
        score, _ = scorer.calculate_fidelity_score(base_facts, path_facts, base_mse, path_mse)
        scores.append(score)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    Processed {i + 1}/{B} paths (elapsed: {elapsed:.1f}s)")
            
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot score distribution for financial time series")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--scaler", type=str, required=True, help="Path to scaler state (.pt)")
    parser.add_argument("--real", type=str, required=True, help="Path to real data CSV")
    parser.add_argument("--fake", type=str, default=None, help="Path to fake/generated data CSV (optional)")
    parser.add_argument("--t-eval", type=int, default=200, help="Timestep for DDPM MSE evaluation (default: 200)")
    parser.add_argument("--stride", type=int, default=50, help="Stride to slice real historical data into 1260-day chunks (default: 50)")
    parser.add_argument("--num-fake-samples", type=int, default=200, help="Number of fake paths to sample and evaluate (default: 200)")
    parser.add_argument("--output", type=str, default="outputs/figures/score_distribution.png", help="Path to save output plot")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cpu)")
    
    args = parser.parse_args()
    
    # 1. 初始化评估打分器
    scorer = FinancialScorer(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        device=args.device
    )
    
    print("\n" + "="*60)
    print(" 1. 计算全局 Real Baseline")
    print("="*60)
    # 计算全局真实数据的基准指标，这里使用标准步长滑动窗口来获取更丰富的全局表示
    base_norm, base_raw = scorer.load_and_preprocess_data(args.real, target_seq_len=1260)
    base_mse = scorer.compute_ddpm_mse(base_norm, t_eval=args.t_eval)
    base_facts = scorer.compute_stylized_facts(base_raw)
    print(f"Global Real Baseline MSE: {base_mse:.6f}")
    print(f"Global Real Baseline Unconditional Corr: {base_facts['uncond_corr']:.4f}")
    
    print("\n" + "="*60)
    print(f" 2. 切分并评估真实数据 (每段 1260 天, 步长 {args.stride})")
    print("="*60)
    # 用较大的 stride 切分真实数据，以代表不同的历史区间
    # 为此，我们手动读取数据进行切片
    df_real = pd.read_csv(args.real, index_col=0)
    df_real = df_real[["sp500", "DGS10"]].ffill().bfill()
    real_raw_full = df_real.values.astype(np.float32)  # (N, 2)
    
    n_days = len(real_raw_full)
    real_chunks = []
    for start in range(0, n_days - 1260 + 1, args.stride):
        win = real_raw_full[start : start + 1260]  # (1260, 2)
        real_chunks.append(win.T)  # (2, 1260)
        
    x_real_chunks = np.stack(real_chunks, axis=0)  # (Num_Chunks, 2, 1260)
    x_real_tensor = torch.tensor(x_real_chunks, dtype=torch.float32)
    x_real_norm = scorer.scaler.transform(x_real_tensor)
    
    print(f"Total Real Chunks to evaluate: {len(x_real_chunks)}")
    real_scores = score_dataset_paths(
        scorer, x_real_norm, x_real_chunks, base_facts, base_mse, t_eval=args.t_eval
    )
    
    # 3. 如果提供了 fake 数据，进行采样并评估
    fake_scores = []
    if args.fake:
        print("\n" + "="*60)
        print(f" 3. 加载并采样评估生成数据 ({args.fake})")
        print("="*60)
        fake_norm, fake_raw = scorer.load_and_preprocess_data(args.fake)
        # fake_raw 形状为 (N_paths, 2, 1260)
        num_paths = fake_raw.shape[0]
        
        # 随机采样一部分进行评估以加快运行速度
        rng = np.random.RandomState(42)
        sample_size = min(args.num_fake_samples, num_paths)
        sampled_indices = rng.choice(num_paths, size=sample_size, replace=False)
        
        x_fake_norm_sampled = fake_norm[sampled_indices]
        x_fake_raw_sampled = fake_raw[sampled_indices]
        
        print(f"Sampled {sample_size}/{num_paths} generated paths for evaluation.")
        fake_scores = score_dataset_paths(
            scorer, x_fake_norm_sampled, x_fake_raw_sampled, base_facts, base_mse, t_eval=args.t_eval
        )
        
    # 4. 绘制分数直方图
    print("\n" + "="*60)
    print(" 4. 绘制保真度分数分布直方图")
    print("="*60)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
    # 绘制 Real Chunks 直方图
    plt.hist(real_scores, bins=15, density=True, alpha=0.4, color="#1f77b4", edgecolor="#1f77b4", label="Real Historical Chunks")
    
    # 绘制 Real Chunks KDE 曲线
    if len(real_scores) > 1:
        try:
            kde_real = gaussian_kde(real_scores)
            x_grid = np.linspace(min(real_scores) - 5, max(real_scores) + 5, 200)
            plt.plot(x_grid, kde_real(x_grid), color="#1f77b4", linewidth=2.5, label="Real Chunks Density (KDE)")
        except Exception as e:
            print(f"KDE calculation for Real failed: {e}")
            
    # 绘制 Fake Paths 直方图
    if args.fake and len(fake_scores) > 0:
        plt.hist(fake_scores, bins=15, density=True, alpha=0.4, color="#e377c2", edgecolor="#e377c2", label="Generated Paths (Evaluated Model)")
        try:
            kde_fake = gaussian_kde(fake_scores)
            x_grid = np.linspace(min(fake_scores) - 5, max(fake_scores) + 5, 200)
            plt.plot(x_grid, kde_fake(x_grid), color="#e377c2", linewidth=2.5, label="Generated Paths Density (KDE)")
        except Exception as e:
            print(f"KDE calculation for Fake failed: {e}")
            
    # 图表细节调整
    plt.title("Fidelity Score Distribution Comparison (1260-Day Windows)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Fidelity Score (Higher is closer to Real global baseline)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend(fontsize=10, loc="upper left")
    
    # 打印一些关键统计量
    print(f"\nScore Statistics:")
    print(f"  Real Chunks  | Mean Score: {np.mean(real_scores):.2f} | Std: {np.std(real_scores):.2f} | Min: {np.min(real_scores):.2f} | Max: {np.max(real_scores):.2f}")
    if args.fake:
        print(f"  Fake Paths   | Mean Score: {np.mean(fake_scores):.2f} | Std: {np.std(fake_scores):.2f} | Min: {np.min(fake_scores):.2f} | Max: {np.max(fake_scores):.2f}")
        
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n[Done] Score distribution plot successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
