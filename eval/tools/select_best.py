#!/usr/bin/env python3
"""
select_best.py — 基于 score.py v2 的保真度评分，从生成路径中评选 Top N 并绘图。
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.score import FinancialScorer


def wide_changes_to_levels(df, sp500_initial=500.0, dgs10_initial=2.0):
    """将宽表格式的日变化量转为价格/收益率水平值。"""
    cols = df.columns.tolist()
    sp500_cols = sorted([c for c in cols if c.startswith("sp500_") and c[6:].isdigit()],
                        key=lambda c: int(c[6:]))
    dgs10_cols = sorted([c for c in cols if c.startswith("dgs10_") and c[6:].isdigit()],
                        key=lambda c: int(c[6:]))

    seq_len = len(sp500_cols)
    sp500_returns = df[sp500_cols].values
    dgs10_diffs = df[dgs10_cols].values

    sp500_levels = sp500_initial * np.cumprod(1.0 + sp500_returns, axis=1)
    dgs10_levels = dgs10_initial + np.cumsum(dgs10_diffs, axis=1)

    sp_level_cols = [f"sp500_level_{i}" for i in range(seq_len)]
    dg_level_cols = [f"dgs10_level_{i}" for i in range(seq_len)]

    out = pd.DataFrame(
        np.concatenate([sp500_levels, dgs10_levels], axis=1),
        columns=sp_level_cols + dg_level_cols,
    )
    return out, sp500_levels, dgs10_levels


def plot_top_paths(sp500_levels, dgs10_levels, top_indices, scores, output_dir, prefix="top5"):
    """
    绘制 Top N 路径的 S&P 500 和 DGS10 水平图。
    """
    N = len(top_indices)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0',
              '#00BCD4', '#FF5722', '#795548'][:N]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # S&P 500
    ax = axes[0]
    for rank, idx in enumerate(top_indices):
        label = f"#{rank+1} Path {idx} (Score: {scores[idx]:.1f})"
        ax.plot(sp500_levels[idx], color=colors[rank], alpha=0.85, linewidth=1.2, label=label)
    ax.set_title(f"DDPM Generated S&P 500 Price Paths (Top {N})", fontsize=14, fontweight='bold')
    ax.set_ylabel("S&P 500 Index Level", fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # DGS10
    ax = axes[1]
    for rank, idx in enumerate(top_indices):
        label = f"#{rank+1} Path {idx} (Score: {scores[idx]:.1f})"
        ax.plot(dgs10_levels[idx], color=colors[rank], alpha=0.85, linewidth=1.2, label=label)
    ax.set_title(f"DDPM Generated 10Y Yield Paths (Top {N})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Trading Day (within window)", fontsize=12)
    ax.set_ylabel("10Y Treasury Yield Level", fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{prefix}_level.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def plot_top_diffs(fake_raw, top_indices, scores, output_dir, prefix="top5"):
    """
    绘制 Top N 路径的日收益率/差分时序图。
    """
    N = len(top_indices)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0',
              '#00BCD4', '#FF5722', '#795548'][:N]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # S&P 500 daily returns
    ax = axes[0]
    for rank, idx in enumerate(top_indices):
        label = f"#{rank+1} Path {idx}"
        ax.plot(fake_raw[idx, 0], color=colors[rank], alpha=0.6, linewidth=0.5, label=label)
    ax.set_title(f"DDPM Generated S&P 500 Daily Returns (Top {N})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Daily Return", fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # DGS10 daily diffs
    ax = axes[1]
    for rank, idx in enumerate(top_indices):
        label = f"#{rank+1} Path {idx}"
        ax.plot(fake_raw[idx, 1], color=colors[rank], alpha=0.6, linewidth=0.5, label=label)
    ax.set_title(f"DDPM Generated 10Y Yield Daily Changes (Top {N})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Trading Day (within window)", fontsize=12)
    ax.set_ylabel("Daily Change", fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{prefix}_diff.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def plot_score_distribution(per_path_scores, top_indices, output_dir, prefix="top5"):
    """绘制评分分布直方图，标注 Top N。"""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(per_path_scores, bins=60, color='#2196F3', alpha=0.75,
            edgecolor='white', linewidth=0.5, label='All paths')
    for rank, idx in enumerate(top_indices):
        ax.axvline(per_path_scores[idx], color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'#{rank+1} Path {idx} ({per_path_scores[idx]:.1f})' if rank < 3 else None)
    ax.set_xlabel("Fidelity Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution with Top 5 Highlighted", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{prefix}_score_dist.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Select top N generated paths based on Fidelity Score v2 and plot"
    )
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "dit-s", "dit-b", "dit-l"],
                        help="Backbone model: unet / dit-s / dit-b / dit-l (default: unet)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--scaler", type=str, required=True, help="Path to scaler state (.pt)")
    parser.add_argument("--real", type=str, required=True, help="Path to real baseline data CSV")
    parser.add_argument("--fake", type=str, required=True, help="Path to generated fake data CSV")
    parser.add_argument("--t-eval", type=int, default=200, help="Timestep for DDPM MSE evaluation")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top paths to select (default: 5)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="outputs/figures", help="Directory for output plots")

    args = parser.parse_args()

    # 1. Initialize FinancialScorer
    scorer = FinancialScorer(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        device=args.device,
        model_type=args.model
    )

    # 2. Calibrate from real data (auto-detects target seq_len from fake data)
    # First peek at fake data to get L_target
    df_fake_peek = pd.read_csv(args.fake, nrows=1)
    sp_cols_peek = [c for c in df_fake_peek.columns if c.startswith("sp500_") and c.split("_")[-1].isdigit()]
    L_target = len(sp_cols_peek) if len(sp_cols_peek) >= 2 else None

    print(f"\n[Step 1/4] Calibrating from real data...")
    scorer.calibrate_from_real(args.real, t_eval=args.t_eval, target_seq_len=L_target)

    # 3. Score all fake paths
    print(f"\n[Step 2/4] Scoring all fake paths...")
    result = scorer.score_batch(args.fake, t_eval=args.t_eval, target_seq_len=L_target)

    per_path_scores = result["per_path_scores"]
    fake_raw = result["fake_raw"]
    N_fake = len(per_path_scores)

    # 4. Get top N indices
    top_n = min(args.top_n, N_fake)
    sorted_indices = np.argsort(-per_path_scores)
    top_indices = sorted_indices[:top_n]

    print(f"\n{'='*65}")
    print(f"          TOP {top_n} GENERATED PATHS (Fidelity Score v2)")
    print(f"{'='*65}")
    for rank, idx in enumerate(top_indices, 1):
        comp = result["per_path_components"][idx]
        print(f"  #{rank} | Path {idx:>5d} | Score: {per_path_scores[idx]:.2f} | "
              f"DDPM: {comp['ddpm_mse']:.1f} | "
              f"Moments: {np.mean([comp['sp_skew'], comp['sp_kurt'], comp['dg_skew'], comp['dg_kurt']]):.1f} | "
              f"Corr: {comp['uncond_corr']:.1f} | "
              f"ACF: {np.mean([comp['sp_acf'], comp['dg_acf']]):.1f}")
    print(f"{'='*65}")
    print(f"  Mean score (all {N_fake} paths): {per_path_scores.mean():.2f} ± {per_path_scores.std():.2f}")
    print(f"  Score range: [{per_path_scores.min():.2f}, {per_path_scores.max():.2f}]")

    # 5. Convert to levels and plot
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"top{top_n}"

    print(f"\n[Step 3/4] Converting to price/yield levels...")
    df_fake = pd.read_csv(args.fake)
    _, sp500_levels, dgs10_levels = wide_changes_to_levels(df_fake)

    print(f"\n[Step 4/4] Generating plots...")

    # Plot level paths
    plot_top_paths(sp500_levels, dgs10_levels, top_indices, per_path_scores,
                   args.output_dir, prefix=prefix)

    # Plot diff paths
    plot_top_diffs(fake_raw, top_indices, per_path_scores,
                   args.output_dir, prefix=prefix)

    # Plot score distribution
    plot_score_distribution(per_path_scores, top_indices,
                            args.output_dir, prefix=prefix)

    print(f"\n[Done] All plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
