#!/usr/bin/env python3
"""
select_best.py — 基于保真度评分从生成路径中评选出 Top 3，并绘制 diff 和 level 图像。
"""

import os
import sys
import argparse
import subprocess
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure src directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.score import FinancialScorer
from eval.eval_distribution import compute_individual_ddpm_mses

def wide_changes_to_levels(df, sp500_initial=500.0, dgs10_initial=2.0):
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
    return out

def main():
    parser = argparse.ArgumentParser(description="Select top 3 generated paths based on Fidelity Score and plot")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--scaler", type=str, required=True, help="Path to scaler state (.pt)")
    parser.add_argument("--real", type=str, required=True, help="Path to real baseline data CSV")
    parser.add_argument("--fake", type=str, required=True, help="Path to generated fake data CSV")
    parser.add_argument("--t-eval", type=int, default=200, help="Timestep for DDPM MSE evaluation (default: 200)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="outputs/figures", help="Directory to save output plots")
    
    args = parser.parse_args()
    
    # 1. Initialize FinancialScorer
    scorer = FinancialScorer(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        device=args.device
    )
    
    # 2. Load Fake data to determine target sequence length
    print(f"\nLoading fake dataset: {args.fake}")
    fake_norm, fake_raw = scorer.load_and_preprocess_data(args.fake)
    B = fake_norm.shape[0]
    L_target = fake_raw.shape[2]
    print(f"Loaded {B} fake paths of length {L_target}")
    
    # 3. Load Real dataset with matching target sequence length
    print(f"\nLoading real baseline: {args.real}")
    base_norm, base_raw = scorer.load_and_preprocess_data(args.real, target_seq_len=L_target)
    base_mse = scorer.compute_ddpm_mse(base_norm, t_eval=args.t_eval)
    base_facts = scorer.compute_stylized_facts(base_raw)
    print(f"Baseline MSE: {base_mse:.6f}, Unconditional Corr: {base_facts['uncond_corr']:.4f}")
    
    # 4. Compute path-wise scores
    print(f"\nComputing Fidelity Score for all {B} generated paths...")
    individual_mses = compute_individual_ddpm_mses(scorer, fake_norm, t_eval=args.t_eval)
    
    scores = []
    start_time = time.time()
    for i in range(B):
        path_raw = fake_raw[i:i+1]
        path_mse = individual_mses[i]
        path_facts = scorer.compute_stylized_facts(path_raw)
        score, _ = scorer.calculate_fidelity_score(base_facts, path_facts, base_mse, path_mse)
        scores.append(score)
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Evaluated {i + 1}/{B} paths (elapsed: {elapsed:.1f}s)")
            
    scores = np.array(scores)
    
    # 5. Get top 3 indices
    top_3_indices = np.argsort(scores)[-3:][::-1]
    
    print("\n" + "="*60)
    print("                    TOP 3 GENERATED PATHS")
    print("="*60)
    for rank, idx in enumerate(top_3_indices, 1):
        print(f"Rank {rank} | Path Index: #{idx} | Fidelity Score: {scores[idx]:.2f}")
    print("="*60)
    
    # 6. Prepare data for plotting
    df_fake = pd.read_csv(args.fake)
    
    # Generate level CSV
    print("\nConverting changes/diff to price/yield levels...")
    df_levels = wide_changes_to_levels(df_fake)
    temp_levels_path = Path("output/temp_select_levels.csv")
    df_levels.to_csv(temp_levels_path, index=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 7. Call show1.py to plot diffs
    print("\n[Plotting] Generating Diff plot using show1.py...")
    paths_str = [str(idx) for idx in top_3_indices]
    
    try:
        subprocess.run([
            sys.executable, "eval/show1.py",
            "--csv", args.fake,
            "--paths"
        ] + paths_str + [
            "--num-figures", "1"
        ], check=True)
        
        # Copy output to final destination
        shutil.copy("outputs/figures/01_raw_time_series.png", os.path.join(args.output_dir, "top3_diff.png"))
        print(f"Saved: {os.path.join(args.output_dir, 'top3_diff.png')}")
    except Exception as e:
        print(f"Error plotting diffs: {e}")
        
    # 8. Call show1.py to plot levels
    print("\n[Plotting] Generating Level plot using show1.py...")
    try:
        subprocess.run([
            sys.executable, "eval/show1.py",
            "--csv", str(temp_levels_path),
            "--paths"
        ] + paths_str + [
            "--num-figures", "1"
        ], check=True)
        
        # Copy output to final destination
        shutil.copy("outputs/figures/01_raw_time_series.png", os.path.join(args.output_dir, "top3_level.png"))
        print(f"Saved: {os.path.join(args.output_dir, 'top3_level.png')}")
    except Exception as e:
        print(f"Error plotting levels: {e}")
    finally:
        # Clean up temporary file
        if temp_levels_path.exists():
            temp_levels_path.unlink()
            
    print("\n[Done] Process completed successfully.")

if __name__ == "__main__":
    main()
