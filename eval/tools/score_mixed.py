#!/usr/bin/env python3
"""
score_mixed.py — Mixed Data Evaluator using DDPM Scorer and PCA-Wasserstein Scorer.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Ensure src directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.score import FinancialScorer
from eval.tools.eval_distribution import _ddpm_compute_individual_mses, _pca_per_path_scores
from eval.score2.advanced_scorer import PCAWassersteinScorer, load_real_data


def load_mixed_data(brown_path, sabr_path, seq_len=1260):
    print(f"Loading mixed data:\n  Brownian: {brown_path}\n  SABR: {sabr_path}")
    df_brown = pd.read_csv(brown_path)
    df_sabr = pd.read_csv(sabr_path)
    
    if len(df_brown) < seq_len or len(df_sabr) < seq_len:
        raise ValueError(f"Mixed data rows ({len(df_brown)} or {len(df_sabr)}) is shorter than seq_len ({seq_len})")
        
    # Take first seq_len steps
    df_brown = df_brown.iloc[:seq_len]
    df_sabr = df_sabr.iloc[:seq_len]
    
    paths = []
    names = []
    
    # Extract columns: mask1_sp500, mask1_DGS10 ... mask5_sp500, mask5_DGS10
    # Shape: (10, 2, seq_len)
    for i in range(1, 6):
        sp_col = f"mask{i}_sp500"
        dg_col = f"mask{i}_DGS10"
        
        # Brownian paths
        p_sp_b = df_brown[sp_col].values.astype(np.float32)
        p_dg_b = df_brown[dg_col].values.astype(np.float32)
        paths.append(np.stack([p_sp_b, p_dg_b], axis=0))
        names.append(f"brown_{i}")
        
    for i in range(1, 6):
        sp_col = f"mask{i}_sp500"
        dg_col = f"mask{i}_DGS10"
        
        # SABR paths
        p_sp_s = df_sabr[sp_col].values.astype(np.float32)
        p_dg_s = df_sabr[dg_col].values.astype(np.float32)
        paths.append(np.stack([p_sp_s, p_dg_s], axis=0))
        names.append(f"sabr_{i}")
        
    return np.stack(paths, axis=0), names


def main():
    brown_path = "/home/u00134/src/data/mixed_data_1st/mixed_brown_masked.csv"
    sabr_path = "/home/u00134/src/data/mixed_data_1st/mixed_sabr_masked.csv"
    real_path = "/home/u00134/data/train_sp500_us10y.csv"
    checkpoint_path = "/home/u00134/src/logs/deep_training_v8_b32/checkpoint_final.pt"
    scaler_path = "/home/u00134/src/logs/deep_training_v8_b32/scaler.pt"
    output_dir = "/home/u00134/src/eval/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load mixed data
    seq_len = 1260
    mixed_data, path_names = load_mixed_data(brown_path, sabr_path, seq_len=seq_len)
    print(f"Successfully loaded {mixed_data.shape[0]} mixed paths with shape {mixed_data.shape}")
    
    # 2. Score under System 1: DDPM Scorer
    print("\n" + "="*50)
    print(" Evaluating using System 1: DDPM Scorer (Checkpoint V8)")
    print("="*50)
    ddpm_scorer = FinancialScorer(
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
        device="cpu"
    )
    
    # Get global real baseline facts & MSE
    print("Loading global real baseline for DDPM scorer...")
    base_norm, base_raw = ddpm_scorer.load_and_preprocess_data(real_path, target_seq_len=seq_len)
    base_mse = ddpm_scorer.compute_ddpm_mse(base_norm, t_eval=200)
    base_facts = ddpm_scorer.compute_stylized_facts(base_raw)
    print(f"DDPM Baseline MSE: {base_mse:.6f}")
    
    # Prepare mixed data for DDPM
    # mixed_data shape: (10, 2, 1260)
    x_mixed_tensor = torch.tensor(mixed_data, dtype=torch.float32)
    x_mixed_norm = ddpm_scorer.scaler.transform(x_mixed_tensor)
    
    # Compute individual MSEs & Scores
    print("Computing path-wise DDPM MSE and Fidelity Scores...")
    ddpm_individual_mses = _ddpm_compute_individual_mses(ddpm_scorer, x_mixed_norm, t_eval=200)
    
    ddpm_scores = []
    for i in range(len(path_names)):
        p_raw = mixed_data[i:i+1]
        p_mse = ddpm_individual_mses[i]
        p_facts = ddpm_scorer.compute_stylized_facts(p_raw)
        score, _ = ddpm_scorer.calculate_fidelity_score(base_facts, p_facts, base_mse, p_mse)
        ddpm_scores.append(score)
        print(f"  Path '{path_names[i]}' DDPM Score: {score:.4f} (MSE Diff: {abs(p_mse-base_mse):.6f})")
        
    # 3. Score under System 2: PCA-Wasserstein Scorer
    print("\n" + "="*50)
    print(" Evaluating using System 2: PCA-Wasserstein Scorer (548 features)")
    print("="*50)
    pca_scorer = PCAWassersteinScorer(variance_threshold=0.95)
    
    # Load and process real baseline features
    print("Loading real baseline features for PCA scorer...")
    real_data = load_real_data(real_path, seq_len=seq_len, stride=50)
    real_features = pca_scorer.extract_features_batch(real_data)
    pca_scorer.fit(real_features)
    
    # Run _pca_per_path_scores on real first to populate cache and calibrate
    _ = _pca_per_path_scores(pca_scorer, real_features, label="Real")
    
    # Extract features for mixed data
    print("Extracting features for mixed paths...")
    mixed_features = pca_scorer.extract_features_batch(mixed_data)
    
    # Compute scores
    print("Computing path-wise PCA-Wasserstein Scores...")
    pca_scores = _pca_per_path_scores(pca_scorer, mixed_features, label="Fake")
    for i in range(len(path_names)):
        print(f"  Path '{path_names[i]}' PCA Score: {pca_scores[i]:.4f}")
        
    # 4. Compile Rankings
    print("\n" + "="*50)
    print(" Ranking Summary")
    print("="*50)
    
    df_results = pd.DataFrame({
        "Path": path_names,
        "DDPM_Score": ddpm_scores,
        "PCA_Score": pca_scores
    })
    
    df_results["DDPM_Rank"] = df_results["DDPM_Score"].rank(ascending=False, method="min").astype(int)
    df_results["PCA_Rank"] = df_results["PCA_Score"].rank(ascending=False, method="min").astype(int)
    
    # Save CSV
    results_csv = os.path.join(output_dir, "mixed_data_scores.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"Saved score details to: {results_csv}")
    
    # Print tables sorted by each ranking
    print("\nSorted by DDPM Scorer Ranking:")
    print(df_results.sort_values(by="DDPM_Rank").to_string(index=False))
    
    print("\nSorted by PCA-Wasserstein Scorer Ranking:")
    print(df_results.sort_values(by="PCA_Rank").to_string(index=False))
    
    # 5. Plot Grouped Bar Chart
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(path_names))
    width = 0.35
    
    # Harmonious premium color palette
    # DDPM: Sleek Slate Blue
    # PCA: Modern Coral Pink
    color_ddpm = "#3f54b4"
    color_pca = "#e05d6f"
    
    rects1 = ax.bar(x - width/2, df_results["DDPM_Score"], width, label="DDPM Scorer", color=color_ddpm, alpha=0.9, edgecolor="none")
    rects2 = ax.bar(x + width/2, df_results["PCA_Score"], width, label="PCA-Wasserstein Scorer", color=color_pca, alpha=0.9, edgecolor="none")
    
    ax.set_ylabel("Fidelity Score (0 - 100)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Fidelity Score Comparison on Mixed Data (10 Paths)\nDDPM Scorer vs. PCA-Wasserstein Scorer", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(path_names, fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="upper right", frameon=True, shadow=False)
    
    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="semibold", color="#333333")
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "mixed_data_ranking_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully saved ranking comparison plot to: {plot_path}")
    
    # Copy to artifacts directory
    artifact_dir = "/home/u00134/.gemini/antigravity-ide/brain/69d80118-90b7-433e-96ad-c92f9602c7bb"
    if os.path.exists(artifact_dir):
        shutil.copy(plot_path, os.path.join(artifact_dir, "mixed_data_ranking_comparison.png"))
        print(f"Copied plot to artifact directory: {os.path.join(artifact_dir, 'mixed_data_ranking_comparison.png')}")


if __name__ == "__main__":
    main()
