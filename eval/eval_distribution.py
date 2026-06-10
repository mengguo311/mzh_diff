#!/usr/bin/env python3
"""
eval_distribution.py — 评估分数分布直方图绘制程序
将真实历史数据以 1260 天滑动窗口切分并打分，绘制真实基准分布，并支持叠加第三方生成数据的分数分布，直观展现生成质量的差异。

支持两种评分后端：
  --scorer ddpm   使用原始 DDPM MSE + Stylized Facts 打分 (需要 checkpoint + scaler)
  --scorer pca    使用 PCA-Wasserstein 548 特征打分 (无需 checkpoint，纯统计)
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 确保 src 目录在 python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
#  DDPM Scorer Backend (原始 score.py)
# ============================================================

def _init_ddpm_scorer(args):
    """初始化 DDPM 评分器（需要 checkpoint 和 scaler）。"""
    import torch
    import config
    from eval.score import FinancialScorer

    if not args.checkpoint or not args.scaler:
        raise ValueError("DDPM scorer requires --checkpoint and --scaler arguments")

    scorer = FinancialScorer(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        device=args.device
    )
    return scorer


@__import__('torch').no_grad()
def _ddpm_compute_individual_mses(scorer, x_normalized, t_eval=200, batch_size=64):
    """高效计算 Batch 中每一条路径单独的 DDPM MSE Loss。"""
    import torch
    scorer.model.eval()
    B = x_normalized.shape[0]
    L = x_normalized.shape[2]

    multiple = 2 ** len(scorer.channel_dims)
    if L % multiple != 0:
        L_clean = (L // multiple) * multiple
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

        sq_error = (noise_pred - noise) ** 2
        batch_mses = sq_error.mean(dim=(1, 2)).cpu().numpy()
        mses.extend(batch_mses)

    return np.array(mses)


def _ddpm_score_paths(scorer, x_norm, x_raw, base_facts, base_mse, t_eval=200):
    """对数据集中每一条路径使用 DDPM 后端独立打分。"""
    B = x_norm.shape[0]
    print(f"  [DDPM] Calculating path-wise DDPM MSE for {B} samples...")
    individual_mses = _ddpm_compute_individual_mses(scorer, x_norm, t_eval=t_eval)

    print(f"  [DDPM] Calculating path-wise Stylized Facts for {B} samples...")
    scores = []
    start_time = time.time()

    for i in range(B):
        path_raw = x_raw[i:i+1]
        path_mse = individual_mses[i]
        path_facts = scorer.compute_stylized_facts(path_raw)
        score, _ = scorer.calculate_fidelity_score(base_facts, path_facts, base_mse, path_mse)
        scores.append(score)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    Processed {i + 1}/{B} paths (elapsed: {elapsed:.1f}s)")

    return scores


def run_ddpm_evaluation(args):
    """使用 DDPM 后端执行完整评估流程。"""
    import torch

    scorer = _init_ddpm_scorer(args)

    print("\n" + "="*60)
    print(" 1. 计算全局 Real Baseline (DDPM)")
    print("="*60)
    base_norm, base_raw = scorer.load_and_preprocess_data(args.real, target_seq_len=1260)
    base_mse = scorer.compute_ddpm_mse(base_norm, t_eval=args.t_eval)
    base_facts = scorer.compute_stylized_facts(base_raw)
    print(f"Global Real Baseline MSE: {base_mse:.6f}")
    print(f"Global Real Baseline Unconditional Corr: {base_facts['uncond_corr']:.4f}")

    print("\n" + "="*60)
    print(f" 2. 切分并评估真实数据 (每段 1260 天, 步长 {args.stride})")
    print("="*60)
    df_real = pd.read_csv(args.real, index_col=0)
    df_real = df_real[["sp500", "DGS10"]].ffill().bfill()
    real_raw_full = df_real.values.astype(np.float32)

    n_days = len(real_raw_full)
    real_chunks = []
    for start in range(0, n_days - 1260 + 1, args.stride):
        win = real_raw_full[start : start + 1260]
        real_chunks.append(win.T)

    x_real_chunks = np.stack(real_chunks, axis=0)
    x_real_tensor = torch.tensor(x_real_chunks, dtype=torch.float32)
    x_real_norm = scorer.scaler.transform(x_real_tensor)

    print(f"Total Real Chunks to evaluate: {len(x_real_chunks)}")
    real_scores = _ddpm_score_paths(
        scorer, x_real_norm, x_real_chunks, base_facts, base_mse, t_eval=args.t_eval
    )

    fake_scores = []
    if args.fake:
        print("\n" + "="*60)
        print(f" 3. 加载并采样评估生成数据 ({args.fake})")
        print("="*60)
        fake_norm, fake_raw = scorer.load_and_preprocess_data(args.fake)
        num_paths = fake_raw.shape[0]

        rng = np.random.RandomState(42)
        sample_size = min(args.num_fake_samples, num_paths)
        sampled_indices = rng.choice(num_paths, size=sample_size, replace=False)

        x_fake_norm_sampled = fake_norm[sampled_indices]
        x_fake_raw_sampled = fake_raw[sampled_indices]

        print(f"Sampled {sample_size}/{num_paths} generated paths for evaluation.")
        fake_scores = _ddpm_score_paths(
            scorer, x_fake_norm_sampled, x_fake_raw_sampled, base_facts, base_mse, t_eval=args.t_eval
        )

    return real_scores, fake_scores


# ============================================================
#  PCA-Wasserstein Scorer Backend (score2/advanced_scorer.py)
# ============================================================

def run_pca_evaluation(args):
    """使用 PCA-Wasserstein 后端执行完整评估流程。"""
    from eval.score2.advanced_scorer import (
        PCAWassersteinScorer, load_real_data, load_fake_data
    )

    seq_len = args.seq_len if hasattr(args, 'seq_len') and args.seq_len else 1260
    variance = args.variance if hasattr(args, 'variance') and args.variance else 0.95

    print("\n" + "="*60)
    print(" 1. 加载数据 (PCA-Wasserstein 模式)")
    print("="*60)
    real_data = load_real_data(args.real, seq_len=seq_len, stride=args.stride)

    scorer = PCAWassersteinScorer(variance_threshold=variance)

    print("\n" + "="*60)
    print(" 2. 提取 548 维特征 → Real 数据")
    print("="*60)
    real_features = scorer.extract_features_batch(real_data)

    print("\n" + "="*60)
    print(" 3. PCA 正交化 (fit on Real)")
    print("="*60)
    scorer.fit(real_features)

    # ---  为 Real 数据中每条路径计算个体分数 ---
    print("\n" + "="*60)
    print(" 4. 计算 Real 路径个体分数")
    print("="*60)
    real_scores = _pca_per_path_scores(scorer, real_features, label="Real")

    # --- Fake 数据 ---
    fake_scores = []
    if args.fake:
        print("\n" + "="*60)
        print(f" 5. 加载并评估生成数据 ({args.fake})")
        print("="*60)
        fake_data = load_fake_data(args.fake, seq_len=seq_len)

        num_paths = fake_data.shape[0]
        sample_size = min(args.num_fake_samples, num_paths)
        if sample_size < num_paths:
            rng = np.random.RandomState(42)
            indices = rng.choice(num_paths, size=sample_size, replace=False)
            fake_data = fake_data[indices]
            print(f"  Sampled {sample_size}/{num_paths} generated paths for evaluation.")

        fake_features = scorer.extract_features_batch(fake_data)
        fake_scores = _pca_per_path_scores(scorer, fake_features, label="Fake")

    return real_scores, fake_scores


def _pca_per_path_scores(scorer, features: np.ndarray, label: str = "") -> list:
    """
    为每条路径计算独立的 PCA-Wasserstein 分数。

    方法：将单条路径的特征向量投影到 PCA 空间中，计算该投影点
    与 Real 数据 PCA 分布中心的加权马氏距离，再映射为 0-100 分数。

    这种"逐条评分"与 score() 的"群体对群体"Wasserstein 评分不同，
    但能为每条路径赋予一个合理的个体分数用于分布绘图。
    """
    from scipy.stats import wasserstein_distance

    N = features.shape[0]
    scaled = scorer.scaler.transform(features)
    projected = scorer.pca.transform(scaled)

    K = projected.shape[1]
    weights = scorer.pca.explained_variance_ratio_[:K]
    weights = weights / weights.sum()

    # Real 分布的中心和标准差（在 PCA 空间中）
    # 使用 fit 时的 Real 数据来获取参考分布
    real_scaled = scorer.scaler.transform(features) if label == "Real" else None
    # 我们需要全局 Real PCA 分布的统计量
    # scorer 在 fit 阶段已经保存了 PCA 模型；使用 PCA 的 mean_ 作为中心
    # 但更好的方式是：用每个 PC 上 Real 数据的分布来衡量偏离程度

    # 获取 Real 的 PCA 投影（已在 fit 中完成）
    # 我们在 fit 时保存它
    if not hasattr(scorer, '_real_pca_cache'):
        # 如果 Real 特征未缓存，使用当前特征作为参考（label=="Real" 时）
        scorer._real_pca_cache = projected if label == "Real" else None

    real_pca_ref = scorer._real_pca_cache
    if real_pca_ref is None:
        # Fallback: 使用标准正态作为参考
        pc_means = np.zeros(K)
        pc_stds = np.ones(K)
    else:
        pc_means = np.mean(real_pca_ref, axis=0)
        pc_stds = np.std(real_pca_ref, axis=0)
        pc_stds = np.maximum(pc_stds, 1e-6)

    # 计算 bootstrap self-distance 作为基准刻度
    if real_pca_ref is not None and real_pca_ref.shape[0] >= 6:
        rng = np.random.RandomState(42)
        self_scores_list = []
        for _ in range(20):
            perm = rng.permutation(real_pca_ref.shape[0])
            half = real_pca_ref.shape[0] // 2
            a_half = real_pca_ref[perm[:half]]
            # 对 a_half 中的每个样本计算到全体的距离
            for j in range(min(5, half)):
                d = np.sum(weights * np.abs(a_half[j] - pc_means) / pc_stds)
                self_scores_list.append(d)
        baseline_dist = float(np.median(self_scores_list))
    else:
        baseline_dist = 1.0
    decay_scale = 3.0 * max(baseline_dist, 0.01)

    scores = []
    t0 = time.time()
    for i in range(N):
        # 加权标准化距离: sum_k w_k * |z_k - mu_k| / sigma_k
        z = projected[i]
        dist = float(np.sum(weights * np.abs(z - pc_means) / pc_stds))
        excess = max(0.0, dist - baseline_dist)
        score = min(100.0, 100.0 * np.exp(-excess / decay_scale))
        scores.append(score)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    [{label}] Scored {i+1}/{N} paths ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  [{label}] Per-path scoring complete: {N} paths in {elapsed:.1f}s")
    return scores


# ============================================================
#  Plotting (共用)
# ============================================================

def plot_distribution(real_scores, fake_scores, output_path, scorer_name="", has_fake=False):
    """绘制保真度分数分布直方图。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    plt.figure(figsize=(12, 7))

    # 绘制 Real Chunks 直方图
    plt.hist(real_scores, bins=15, density=True, alpha=0.4, color="#1f77b4",
             edgecolor="#1f77b4", label="Real Historical Chunks")

    # 绘制 Real Chunks KDE 曲线
    if len(real_scores) > 1:
        try:
            kde_real = gaussian_kde(real_scores)
            x_grid = np.linspace(min(real_scores) - 5, max(real_scores) + 5, 200)
            plt.plot(x_grid, kde_real(x_grid), color="#1f77b4", linewidth=2.5,
                     label="Real Chunks Density (KDE)")
        except Exception as e:
            print(f"KDE calculation for Real failed: {e}")

    # 绘制 Fake Paths 直方图
    if has_fake and len(fake_scores) > 0:
        plt.hist(fake_scores, bins=15, density=True, alpha=0.4, color="#e377c2",
                 edgecolor="#e377c2", label="Generated Paths (Evaluated Model)")
        try:
            kde_fake = gaussian_kde(fake_scores)
            x_grid = np.linspace(min(fake_scores) - 5, max(fake_scores) + 5, 200)
            plt.plot(x_grid, kde_fake(x_grid), color="#e377c2", linewidth=2.5,
                     label="Generated Paths Density (KDE)")
        except Exception as e:
            print(f"KDE calculation for Fake failed: {e}")

    # 图表细节
    title_suffix = f" [{scorer_name}]" if scorer_name else ""
    plt.title(f"Fidelity Score Distribution Comparison (1260-Day Windows){title_suffix}",
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Fidelity Score (Higher is closer to Real global baseline)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend(fontsize=10, loc="upper left")

    # 打印关键统计量
    print(f"\nScore Statistics:")
    print(f"  Real Chunks  | Mean Score: {np.mean(real_scores):.2f} | Std: {np.std(real_scores):.2f} "
          f"| Min: {np.min(real_scores):.2f} | Max: {np.max(real_scores):.2f}")
    if has_fake and len(fake_scores) > 0:
        print(f"  Fake Paths   | Mean Score: {np.mean(fake_scores):.2f} | Std: {np.std(fake_scores):.2f} "
              f"| Min: {np.min(fake_scores):.2f} | Max: {np.max(fake_scores):.2f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[Done] Score distribution plot successfully saved to: {output_path}")


# ============================================================
#  Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot score distribution for financial time series"
    )
    # 评分后端选择
    parser.add_argument("--scorer", type=str, default="ddpm", choices=["ddpm", "pca"],
                        help="Scoring backend: 'ddpm' (DDPM MSE + Stylized Facts, default) "
                             "or 'pca' (PCA-Wasserstein 548 features, no checkpoint needed)")

    # 通用参数
    parser.add_argument("--real", type=str, required=True, help="Path to real data CSV")
    parser.add_argument("--fake", type=str, default=None, help="Path to fake/generated data CSV (optional)")
    parser.add_argument("--stride", type=int, default=50,
                        help="Stride to slice real historical data into 1260-day chunks (default: 50)")
    parser.add_argument("--num-fake-samples", type=int, default=200,
                        help="Number of fake paths to sample and evaluate (default: 200)")
    parser.add_argument("--output", type=str, default="outputs/figures/score_distribution.png",
                        help="Path to save output plot")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cpu)")

    # DDPM 专用参数
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[DDPM only] Path to trained checkpoint (.pt)")
    parser.add_argument("--scaler", type=str, default=None,
                        help="[DDPM only] Path to scaler state (.pt)")
    parser.add_argument("--t-eval", type=int, default=200,
                        help="[DDPM only] Timestep for DDPM MSE evaluation (default: 200)")

    # PCA 专用参数
    parser.add_argument("--seq-len", type=int, default=1260,
                        help="[PCA only] Window length for slicing (default: 1260)")
    parser.add_argument("--variance", type=float, default=0.95,
                        help="[PCA only] PCA variance threshold (default: 0.95)")

    args = parser.parse_args()

    print("=" * 60)
    print(f"  Score Distribution Evaluator  [Backend: {args.scorer.upper()}]")
    print("=" * 60)

    if args.scorer == "ddpm":
        real_scores, fake_scores = run_ddpm_evaluation(args)
        scorer_name = "DDPM"
    elif args.scorer == "pca":
        real_scores, fake_scores = run_pca_evaluation(args)
        scorer_name = "PCA-Wasserstein"
    else:
        raise ValueError(f"Unknown scorer: {args.scorer}")

    # 绘制分布图
    print("\n" + "="*60)
    print(" 绘制保真度分数分布直方图")
    print("="*60)

    plot_distribution(real_scores, fake_scores, args.output,
                      scorer_name=scorer_name, has_fake=bool(args.fake))


if __name__ == "__main__":
    main()

