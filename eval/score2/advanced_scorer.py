#!/usr/bin/env python3
"""
advanced_scorer.py — Scorer V3: 6 Core Financial Indicators Scorer
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
#  1. Feature Extraction Module
# ============================================================

def extract_six_features(x: np.ndarray) -> dict:
    """
    Extract 6 core financial indicators from batch paths.
    Args:
        x: np.ndarray of shape [Batch, 2, SEQ_LEN]
    Returns:
        dict containing 6 keys, each with shape [Batch]
    """
    B = x.shape[0]
    skew_sp500 = np.zeros(B)
    kurt_sp500 = np.zeros(B)
    vol_clustering = np.zeros(B)
    tail_corr = np.zeros(B)
    lead_lag = np.zeros(B)
    portfolio_var = np.zeros(B)
    
    for i in range(B):
        sp = x[i, 0]
        dg = x[i, 1]
        
        # 1. skew_sp500 (with clamp to prevent extreme outlier/NaN error)
        s = skew(sp)
        skew_sp500[i] = np.clip(s, -10.0, 10.0) if np.isfinite(s) else 0.0
        
        # 2. kurt_sp500 (with clamp)
        k = kurtosis(sp)
        kurt_sp500[i] = np.clip(k, -20.0, 100.0) if np.isfinite(k) else 0.0
        
        # 3. vol_clustering: average lag-1 to lag-5 ACF of absolute returns (or squared returns)
        # We use squared returns x^2 as requested by the prompt: (x^2) 的 Lag-1 到 Lag-5 自相关系数的平均值
        r2 = sp ** 2
        acfs = []
        for lag in range(1, 6):
            n = len(r2)
            if n > lag:
                mean_r2 = np.mean(r2)
                var_r2 = np.var(r2)
                if var_r2 > 1e-12:
                    cov = np.mean((r2[lag:] - mean_r2) * (r2[:-lag] - mean_r2))
                    acf_val = cov / var_r2
                else:
                    acf_val = 0.0
            else:
                acf_val = 0.0
            acfs.append(acf_val)
        vol_clustering[i] = np.mean(acfs)
        
        # 4. tail_corr: Channel 0 < mean - 1.5 * std, Pearson correlation between Channel 0 & Channel 1
        mean_sp = np.mean(sp)
        std_sp = np.std(sp)
        threshold = mean_sp - 1.5 * (std_sp if std_sp > 1e-8 else 1e-8)
        mask = sp < threshold
        if np.sum(mask) >= 2:
            c = np.corrcoef(sp[mask], dg[mask])[0, 1]
            tail_corr[i] = c if np.isfinite(c) else 0.0
        else:
            tail_corr[i] = 0.0
            
        # 5. lead_lag: Cross-Correlation between Channel 1 (t-1) and Channel 0 (t)
        if len(sp) > 1:
            c = np.corrcoef(dg[:-1], sp[1:])[0, 1]
            lead_lag[i] = c if np.isfinite(c) else 0.0
        else:
            lead_lag[i] = 0.0
            
        # 6. portfolio_var: 1st percentile of 0.6 * SP500 - 0.4 * DGS10
        port = 0.6 * sp - 0.4 * dg
        var_99 = np.percentile(port, 1.0)
        portfolio_var[i] = var_99 if np.isfinite(var_99) else 0.0
        
    return {
        "skew_sp500": skew_sp500,
        "kurt_sp500": kurt_sp500,
        "vol_clustering": vol_clustering,
        "tail_corr": tail_corr,
        "lead_lag": lead_lag,
        "portfolio_var": portfolio_var
    }


# ============================================================
#  2. Distribution Matcher & Scorer
# ============================================================

def compute_fidelity_scores(real_features: dict, fake_features: dict) -> tuple[float, dict]:
    """
    Compute W1 distance and score for each of the 6 indicators.
    """
    scores = {}
    details = {}
    
    keys = ["skew_sp500", "kurt_sp500", "vol_clustering", "tail_corr", "lead_lag", "portfolio_var"]
    for key in keys:
        u = real_features[key]
        v = fake_features[key]
        
        w1 = wasserstein_distance(u, v)
        sigma_real = np.std(u)
        
        # score = 100 * exp(-W_1 / (sigma_real + 1e-6))
        score = 100.0 * np.exp(-w1 / (sigma_real + 1e-6))
        
        scores[key] = float(score)
        details[key] = {
            "w1_distance": float(w1),
            "sigma_real": float(sigma_real),
            "score": float(score)
        }
        
    total_score = float(np.mean(list(scores.values())))
    return total_score, details


# ============================================================
#  3. Data Loading Utilities
# ============================================================

def load_real_data(csv_path: str, seq_len: int = 1260, stride: int = 50) -> np.ndarray:
    """Load and slice real data."""
    df = pd.read_csv(csv_path, index_col=0)
    col_pairs = [("sp500", "DGS10"), ("SP500", "DGS10")]
    col1, col2 = None, None
    for c1, c2 in col_pairs:
        if c1 in df.columns and c2 in df.columns:
            col1, col2 = c1, c2
            break
    if col1 is None:
        numeric = df.select_dtypes(include=[np.number]).columns[:2]
        col1, col2 = numeric[0], numeric[1]

    df_clean = df[[col1, col2]].ffill().bfill()

    # Check if level data (needs conversion to changes)
    if df_clean[col1].abs().mean() > 1.0:
        changes = pd.DataFrame(index=df_clean.index)
        changes[col1] = df_clean[col1].pct_change()
        changes[col2] = df_clean[col2].diff()
        changes = changes.dropna()
        raw = changes.values.astype(np.float32)
    else:
        raw = df_clean.values.astype(np.float32)

    n_days = len(raw)
    windows = []
    for start in range(0, n_days - seq_len + 1, stride):
        win = raw[start:start + seq_len]  # (seq_len, 2)
        windows.append(win.T)  # (2, seq_len)

    result = np.stack(windows, axis=0)
    print(f"  [Real Data] Loaded {result.shape[0]} windows of length {seq_len} (stride={stride})")
    return result


def load_fake_data(csv_path: str, seq_len: int = None) -> np.ndarray:
    """Load fake data in wide format or vertical masked format."""
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()

    sp_cols = sorted([c for c in cols if c.startswith("sp500_") and c.split("_")[-1].isdigit()],
                     key=lambda c: int(c.split("_")[-1]))
    dg_cols = sorted([c for c in cols if c.startswith("dgs10_") and c.split("_")[-1].isdigit()],
                     key=lambda c: int(c.split("_")[-1]))

    if len(sp_cols) >= 2 and len(dg_cols) >= 2:
        sp_data = df[sp_cols].values.astype(np.float32)
        dg_data = df[dg_cols].values.astype(np.float32)

        is_level = any("level" in c for c in sp_cols)
        if is_level:
            sp_changes = np.zeros_like(sp_data)
            dg_changes = np.zeros_like(dg_data)
            sp_changes[:, 0] = sp_data[:, 0] / 500.0 - 1.0
            dg_changes[:, 0] = dg_data[:, 0] - 2.0
            sp_changes[:, 1:] = sp_data[:, 1:] / sp_data[:, :-1] - 1.0
            dg_changes[:, 1:] = np.diff(dg_data, axis=1)
            x = np.stack([sp_changes, dg_changes], axis=1)
        else:
            x = np.stack([sp_data, dg_data], axis=1)

        if seq_len is not None and x.shape[2] > seq_len:
            x = x[:, :, :seq_len]

        print(f"  [Fake Data] Loaded {x.shape[0]} paths of length {x.shape[2]}")
        return x
    else:
        # Fallback to masked vertical columns format (e.g. mixed_brown_masked.csv)
        mask_sp_cols = sorted([c for c in cols if c.endswith("_sp500")])
        if len(mask_sp_cols) >= 1:
            paths = []
            for col in mask_sp_cols:
                prefix = col[:-6] # e.g. mask1
                dg_col = f"{prefix}_DGS10"
                if dg_col in cols:
                    p_sp = df[col].values.astype(np.float32)
                    p_dg = df[dg_col].values.astype(np.float32)
                    if seq_len is not None:
                        p_sp = p_sp[:seq_len]
                        p_dg = p_dg[:seq_len]
                    paths.append(np.stack([p_sp, p_dg], axis=0))
            if len(paths) >= 1:
                x = np.stack(paths, axis=0)
                print(f"  [Fake Data] Loaded {x.shape[0]} masked vertical paths of length {x.shape[2]}")
                return x

        raise ValueError(f"Cannot detect wide-format or masked vertical columns in {csv_path}")


# ============================================================
#  4. Top-level API
# ============================================================

def evaluate_model(real_csv: str, fake_csv: str, seq_len: int = 1260, stride: int = 50) -> float:
    print(f"\n[Scorer V3] Initializing evaluation flow...")
    real_data = load_real_data(real_csv, seq_len=seq_len, stride=stride)
    fake_data = load_fake_data(fake_csv, seq_len=seq_len)
    
    print(f"[Scorer V3] Extracting 6 core financial indicators...")
    real_features = extract_six_features(real_data)
    fake_features = extract_six_features(fake_data)
    
    total_score, details = compute_fidelity_scores(real_features, fake_features)
    
    print("\n" + "=" * 65)
    print("                MODEL FIDELITY REPORT (Scorer V3)")
    print("=" * 65)
    print("| Indicator | Wasserstein Distance (W1) | Real Std (sigma) | Score |")
    print("| :--- | :---: | :---: | :---: |")
    for key, info in details.items():
        print(f"| {key} | {info['w1_distance']:.6f} | {info['sigma_real']:.6f} | {info['score']:.2f} |")
    print(f"| **Total Fidelity Score** | - | - | **{total_score:.2f}** |")
    print("=" * 65 + "\n")
    
    return total_score


# ============================================================
#  5. Backward Compatibility Layer (PCAWassersteinScorer Wrapper)
# ============================================================

class PCAWassersteinScorer:
    """
    Compatibility class mapping to the 6 core features.
    """
    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_threshold)
        self._fitted = False
        
    def extract_features_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Extract the 6 core features for the compatibility interface.
        Returns:
            np.ndarray of shape [N, 6]
        """
        features_dict = extract_six_features(data)
        keys = ["skew_sp500", "kurt_sp500", "vol_clustering", "tail_corr", "lead_lag", "portfolio_var"]
        feats = np.stack([features_dict[k] for k in keys], axis=1)
        return feats
        
    def fit(self, real_features: np.ndarray):
        self.scaler.fit(real_features)
        real_scaled = self.scaler.transform(real_features)
        self.pca.fit(real_scaled)
        self._fitted = True


# ============================================================
#  6. Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scorer V3: 6 Core Financial Indicators Scorer"
    )
    parser.add_argument("--real", type=str, required=True, help="Path to real data CSV")
    parser.add_argument("--fake", type=str, required=True, help="Path to generated data CSV")
    parser.add_argument("--seq-len", type=int, default=1260, help="Window length for slicing")
    parser.add_argument("--stride", type=int, default=50, help="Stride for real data slicing")
    parser.add_argument("--variance", type=float, default=0.95, help="Unused, kept for compatibility")
    parser.add_argument("--num-fake", type=int, default=None, help="Unused, kept for compatibility")
    parser.add_argument("--json", type=str, default=None, help="Unused, kept for compatibility")
    parser.add_argument("--device", type=str, default="cpu", help="Unused, kept for compatibility")

    args = parser.parse_args()

    evaluate_model(args.real, args.fake, seq_len=args.seq_len, stride=args.stride)


if __name__ == "__main__":
    main()
