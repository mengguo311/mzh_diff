#!/usr/bin/env python3
"""
score.py — 双资产模拟生成数据体检报告打分系统 v2
使用训练好的一维扩散模型（1D-DDPM）结合量化金融 Stylized Facts（典型事实）
对第三方生成的数据进行多维度综合评分。

v2 修复:
  - DDPM MSE 改用分位数归一化 (Percentile Scoring)，修复因量级失配导致的 ≈ 0 分问题
  - Stylized Facts 的衰减尺度 σ 从硬编码改为自适应（从真实数据窗口统计量 std 计算）
  - 去除高维空间不稳定的 MMD 指标
  - 支持逐路径 (per-path) 评分
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import skew, kurtosis

# 确保 src 目录在 python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset import TimeSeriesScaler
from unet1d import UNet1d
from scheduler import DDPMScheduler
from eval.metrics import calculate_1d_wasserstein


# ============================================================
#  Stylized Facts 计算 (单条路径级别)
# ============================================================

def compute_path_stats(sp: np.ndarray, dg: np.ndarray, acf_max_lag: int = 10) -> dict:
    """
    对单条路径计算全部 Stylized Facts 指标。

    Args:
        sp: (L,) S&P 500 日收益率序列
        dg: (L,) DGS10 日差分序列
        acf_max_lag: ACF 最大滞后阶数
    Returns:
        dict: 各指标标量值
    """
    # 1. Moments
    sp_skew = float(skew(sp))
    sp_kurt = float(kurtosis(sp))  # excess kurtosis
    dg_skew = float(skew(dg))
    dg_kurt = float(kurtosis(dg))

    # 安全截断 (防止 NaN / Inf)
    sp_skew = np.clip(sp_skew, -20, 20) if np.isfinite(sp_skew) else 0.0
    sp_kurt = np.clip(sp_kurt, -20, 200) if np.isfinite(sp_kurt) else 0.0
    dg_skew = np.clip(dg_skew, -20, 20) if np.isfinite(dg_skew) else 0.0
    dg_kurt = np.clip(dg_kurt, -20, 200) if np.isfinite(dg_kurt) else 0.0

    # 2. Volatility Clustering — |r| 的 ACF (Lag 1 ~ max_lag)
    abs_sp = np.abs(sp)
    abs_dg = np.abs(dg)
    sp_acf = _compute_acf(abs_sp, acf_max_lag)
    dg_acf = _compute_acf(abs_dg, acf_max_lag)

    # 3. Unconditional Correlation
    if np.std(sp) > 1e-10 and np.std(dg) > 1e-10:
        uncond_corr = float(np.corrcoef(sp, dg)[0, 1])
        if not np.isfinite(uncond_corr):
            uncond_corr = 0.0
    else:
        uncond_corr = 0.0

    # 4. Tail Correlation (SP500 < mean - 1.5 * std)
    mu_sp = np.mean(sp)
    std_sp = np.std(sp)
    threshold = mu_sp - 1.5 * (std_sp if std_sp > 1e-8 else 1e-8)
    mask = sp < threshold
    if np.sum(mask) >= 5:
        tc = np.corrcoef(sp[mask], dg[mask])[0, 1]
        tail_corr = float(tc) if np.isfinite(tc) else 0.0
    else:
        tail_corr = 0.0

    return {
        "sp_skew": sp_skew,
        "sp_kurt": sp_kurt,
        "dg_skew": dg_skew,
        "dg_kurt": dg_kurt,
        "sp_acf": sp_acf,
        "dg_acf": dg_acf,
        "uncond_corr": uncond_corr,
        "tail_corr": tail_corr,
    }


def _compute_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """计算单条序列的 ACF (Lag 1 ~ max_lag)。"""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    acf = np.zeros(max_lag)
    if var < 1e-12:
        return acf
    for lag in range(1, max_lag + 1):
        if n > lag:
            cov = np.mean((series[lag:] - mean) * (series[:-lag] - mean))
            acf[lag - 1] = cov / var
    return acf


def compute_batch_stats(x_changes: np.ndarray, acf_max_lag: int = 10) -> list[dict]:
    """
    对一个 batch 的路径逐条计算 Stylized Facts。

    Args:
        x_changes: (Batch, 2, L) 日变化量数组
    Returns:
        list[dict]: 每条路径的指标字典
    """
    B = x_changes.shape[0]
    results = []
    for i in range(B):
        sp = x_changes[i, 0]
        dg = x_changes[i, 1]
        results.append(compute_path_stats(sp, dg, acf_max_lag))
    return results


# ============================================================
#  FinancialScorer
# ============================================================

class FinancialScorer:
    """
    1D-DDPM 基于隐式物理特征与量化金融典型事实的评估与打分系统 v2。

    评分流程:
        1. calibrate_from_real(): 加载真实数据，对每个滑动窗口计算指标 + DDPM MSE，
           得到各指标的 (baseline_mean, baseline_std) 作为自适应 σ。
        2. score_fake(): 加载假数据，计算同样的指标 + DDPM MSE，
           使用分位数归一化 (DDPM MSE) + 自适应指数衰减 (Stylized Facts) 评分。
    """

    # 权重配置
    WEIGHTS = {
        "ddpm_mse":     0.20,
        "sp_skew":      0.075,
        "sp_kurt":      0.075,
        "dg_skew":      0.05,
        "dg_kurt":      0.05,
        "sp_acf":       0.10,
        "dg_acf":       0.10,
        "uncond_corr":  0.15,
        "tail_corr":    0.10,
        "wasserstein":  0.10,
    }

    def __init__(self, checkpoint_path: str, scaler_path: str, device: str = None):
        # 1. 确定运行设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[FinancialScorer] Using device: {self.device}")

        # 2. 动态读取并应用训练配置
        self.checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        config_json_path = os.path.join(self.checkpoint_dir, "config.json")

        # 备份默认值
        self.seq_len = config.SEQ_LEN
        self.channels = config.CHANNELS
        self.channel_dims = config.CHANNEL_DIMS
        self.time_emb_dim = config.TIME_EMB_DIM
        self.T = config.T
        self.beta_start = config.BETA_START
        self.beta_end = config.BETA_END

        if os.path.exists(config_json_path):
            print(f"[FinancialScorer] Loading run configuration from: {config_json_path}")
            try:
                with open(config_json_path, "r") as f:
                    run_config = json.load(f)
                self.seq_len = run_config.get("seq_len", self.seq_len)
                self.channels = run_config.get("channels", self.channels)
                self.channel_dims = run_config.get("channel_dims", self.channel_dims)
                self.time_emb_dim = run_config.get("time_emb_dim", self.time_emb_dim)
                self.T = run_config.get("T", self.T)
                self.beta_start = run_config.get("beta_start", self.beta_start)
                self.beta_end = run_config.get("beta_end", self.beta_end)

                # 同步更新全局 config 模块中的值
                config.SEQ_LEN = self.seq_len
                config.CHANNELS = self.channels
                config.CHANNEL_DIMS = self.channel_dims
                config.TIME_EMB_DIM = self.time_emb_dim
                config.T = self.T
                config.BETA_START = self.beta_start
                config.BETA_END = self.beta_end

                print(f"  Applied: seq_len={self.seq_len}, channel_dims={self.channel_dims}, T={self.T}")
            except Exception as e:
                print(f"  [Warning] Failed to load config.json: {e}")

        # 3. 初始化并加载 1D U-Net 模型
        print("[FinancialScorer] Loading U-Net model...")
        self.model = UNet1d(
            in_channels=self.channels,
            channel_dims=self.channel_dims,
            time_emb_dim=self.time_emb_dim
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 优先读取 EMA 权重
        if "ema_state_dict" in checkpoint:
            ema_state = checkpoint["ema_state_dict"]
            model_state = self.model.state_dict()
            for name in ema_state:
                if name in model_state:
                    model_state[name] = ema_state[name]
            self.model.load_state_dict(model_state)
            print("  Loaded EMA model weights.")
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("  Loaded normal model weights (EMA not found).")
        else:
            self.model.load_state_dict(checkpoint)
            print("  Loaded raw checkpoint state dict.")

        self.model.eval()

        # 4. 初始化 DDPMScheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=self.T,
            beta_start=self.beta_start,
            beta_end=self.beta_end
        ).to(self.device)

        # 5. 加载 TimeSeriesScaler
        print("[FinancialScorer] Loading TimeSeriesScaler...")
        self.scaler = TimeSeriesScaler()
        self.scaler.load(scaler_path)

        # 6. 校准状态 (由 calibrate_from_real 填充)
        self._calibrated = False
        self.real_baselines = {}   # 各指标的 (mean, std)
        self.real_mse_dist = None  # 真实数据的逐窗口 DDPM MSE 分布
        self.real_stats_list = []  # 真实数据每个窗口的 stats 字典列表

    # ──────────────────────────────────────────────
    # 数据加载
    # ──────────────────────────────────────────────

    def load_and_preprocess_data(self, csv_path: str, target_seq_len: int = None) -> tuple[torch.Tensor, np.ndarray]:
        """
        加载并预处理数据。
        Returns:
            x_normalized: (Batch, 2, seq_len) PyTorch 张量（标准化空间）
            x_changes:    (Batch, 2, seq_len) NumPy 数组（真实量级）
        """
        df = pd.read_csv(csv_path)
        cols = df.columns.tolist()

        # 检测宽表格式 (sp500_0, sp500_1, ... / dgs10_0, ...)
        sp_cols = sorted(
            [c for c in cols if (c.startswith("sp500_") or c.startswith("sp500_level_")) and c.split("_")[-1].isdigit()],
            key=lambda c: int(c.split("_")[-1])
        )
        dg_cols = sorted(
            [c for c in cols if (c.startswith("dgs10_") or c.startswith("dgs10_level_")) and c.split("_")[-1].isdigit()],
            key=lambda c: int(c.split("_")[-1])
        )
        is_wide = len(sp_cols) >= 2 and len(dg_cols) >= 2

        if is_wide:
            is_level = any("level" in c for c in sp_cols)
            sp_data = df[sp_cols].values.astype(np.float32)
            dg_data = df[dg_cols].values.astype(np.float32)

            if is_level:
                sp500_initial = 500.0
                dgs10_initial = 2.0
                sp_changes = np.zeros_like(sp_data)
                dg_changes = np.zeros_like(dg_data)
                sp_changes[:, 0] = sp_data[:, 0] / sp500_initial - 1.0
                dg_changes[:, 0] = dg_data[:, 0] - dgs10_initial
                sp_changes[:, 1:] = sp_data[:, 1:] / sp_data[:, :-1] - 1.0
                dg_changes[:, 1:] = np.diff(dg_data, axis=1)
                print(f"  [Wide] Converted level → daily changes.")
            else:
                sp_changes = sp_data
                dg_changes = dg_data
                print(f"  [Wide] Loaded daily changes directly.")

            x_changes = np.stack([sp_changes, dg_changes], axis=1)

        else:
            # 长表格式
            col_pairs = [
                ("sp500", "DGS10"), ("SP500", "DGS10"),
                ("Asset_1_Return", "Asset_2_Return"),
                ("Asset_1_Level", "Asset_2_Level"),
                ("asset_1", "asset_2"), ("Asset1", "Asset2"),
            ]
            col1, col2 = None, None
            for c1, c2 in col_pairs:
                if c1 in df.columns and c2 in df.columns:
                    col1, col2 = c1, c2
                    break
            if col1 is None or col2 is None:
                numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                                if c.lower() not in {"day", "index", "date"}]
                if len(numeric_cols) < 2:
                    raise ValueError(f"CSV must have ≥2 numeric columns, got {list(df.columns)}")
                col1, col2 = numeric_cols[0], numeric_cols[1]

            df_clean = df[[col1, col2]].copy().ffill().bfill()

            # 判断是否为价格水平 (level) 数据
            is_level = df_clean[col1].abs().mean() > 1.0
            if is_level:
                df_changes = pd.DataFrame(index=df_clean.index)
                df_changes[col1] = df_clean[col1].pct_change()
                df_changes[col2] = df_clean[col2].diff()
                df_changes = df_changes.dropna()
                print(f"  [Long] Converted level → daily changes.")
            else:
                df_changes = df_clean
                print(f"  [Long] Loaded daily changes directly.")

            raw_data = df_changes.values.astype(np.float32)

            # 滑动窗口划分
            n_days = len(raw_data)
            stride = 5
            seq_len_to_use = target_seq_len if target_seq_len is not None else self.seq_len
            window_indices = list(range(0, n_days - seq_len_to_use + 1, stride))
            if len(window_indices) == 0:
                raise ValueError(f"Data length {n_days} < required seq_len {seq_len_to_use}")

            windows = []
            for start in window_indices:
                win = raw_data[start:start + seq_len_to_use]
                windows.append(win.T)

            x_changes = np.stack(windows, axis=0)
            print(f"  [Long] {len(x_changes)} windows of length {seq_len_to_use}.")

        # 标准化
        x_tensor = torch.tensor(x_changes, dtype=torch.float32)
        x_normalized = self.scaler.transform(x_tensor)

        return x_normalized, x_changes

    # ──────────────────────────────────────────────
    # DDPM MSE 计算 (逐路径)
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def compute_ddpm_mse_per_path(
        self, x_normalized: torch.Tensor, t_eval: int = 200, batch_size: int = 64
    ) -> np.ndarray:
        """
        逐路径计算 DDPM Noise Prediction MSE。

        Returns:
            np.ndarray: (N,) 每条路径的 MSE 值
        """
        self.model.eval()
        B = x_normalized.shape[0]
        L = x_normalized.shape[2]

        # U-Net 对齐
        multiple = 2 ** len(self.channel_dims)
        if L % multiple != 0:
            L_clean = (L // multiple) * multiple
            x_input = x_normalized[:, :, :L_clean]
        else:
            x_input = x_normalized

        mse_per_path = np.zeros(B)

        for i in range(0, B, batch_size):
            x0 = x_input[i:i + batch_size].to(self.device)
            n_batch = x0.shape[0]

            t = torch.full((n_batch,), t_eval, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0)

            # 前向加噪
            xt = self.scheduler.q_sample(x0, t, noise)

            # 提取初始条件
            c = x0[:, :, 0]

            # 模型预测噪声
            noise_pred = self.model(xt, t, c)

            # 逐路径 MSE (在 C 和 L 维度上取平均)
            per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none")
            per_sample_mse = per_sample_mse.mean(dim=(1, 2))  # (n_batch,)

            mse_per_path[i:i + n_batch] = per_sample_mse.cpu().numpy()

        return mse_per_path

    # ──────────────────────────────────────────────
    # 校准 (Calibration from Real Data)
    # ──────────────────────────────────────────────

    def calibrate_from_real(self, real_csv_path: str, t_eval: int = 200, target_seq_len: int = None):
        """
        用真实数据校准评分系统。

        对真实数据的每个滑动窗口:
          1. 计算 DDPM MSE → 得到 real_mse_distribution
          2. 计算 Stylized Facts → 得到各指标的 (mean, std) 作为自适应 σ

        校准后，self.real_baselines 和 self.real_mse_dist 会被填充。
        """
        print(f"\n{'='*65}")
        print(f"  [Calibration] Processing real data: {real_csv_path}")
        print(f"{'='*65}")

        real_norm, real_raw = self.load_and_preprocess_data(real_csv_path, target_seq_len=target_seq_len)
        N_real = real_raw.shape[0]
        print(f"  Real data: {N_real} windows, shape={real_raw.shape}")

        # 1. 逐窗口 DDPM MSE
        print(f"  Computing DDPM MSE for {N_real} real windows (t={t_eval})...")
        self.real_mse_dist = self.compute_ddpm_mse_per_path(real_norm, t_eval=t_eval)
        print(f"    Real MSE: mean={self.real_mse_dist.mean():.6f}, "
              f"std={self.real_mse_dist.std():.6f}, "
              f"range=[{self.real_mse_dist.min():.6f}, {self.real_mse_dist.max():.6f}]")

        # 2. 逐窗口 Stylized Facts
        print(f"  Computing Stylized Facts for {N_real} real windows...")
        self.real_stats_list = compute_batch_stats(real_raw)

        # 3. 汇总各指标的 baseline (mean, std)
        scalar_keys = ["sp_skew", "sp_kurt", "dg_skew", "dg_kurt", "uncond_corr", "tail_corr"]
        self.real_baselines = {}

        for key in scalar_keys:
            values = np.array([s[key] for s in self.real_stats_list])
            self.real_baselines[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        # ACF 向量: 取各窗口的平均 ACF 作为 baseline，std 从 MAE 分布计算
        for acf_key in ["sp_acf", "dg_acf"]:
            all_acf = np.array([s[acf_key] for s in self.real_stats_list])  # (N_real, max_lag)
            mean_acf = np.mean(all_acf, axis=0)

            # 每个窗口与全局均值的 MAE
            mae_per_window = np.mean(np.abs(all_acf - mean_acf), axis=1)  # (N_real,)
            self.real_baselines[acf_key] = {
                "mean_acf": mean_acf.tolist(),
                "mae_mean": float(np.mean(mae_per_window)),
                "mae_std": float(np.std(mae_per_window)),
            }

        # 保留标准化后的真实数据引用（用于 Wasserstein 计算）
        self._real_norm = real_norm
        self._real_raw = real_raw

        self._calibrated = True

        # 打印校准结果
        print(f"\n  {'─'*55}")
        print(f"  Calibration Results (Adaptive σ from real data):")
        print(f"  {'─'*55}")
        for key in scalar_keys:
            b = self.real_baselines[key]
            print(f"    {key:15s}: mean={b['mean']:+.4f}, σ={b['std']:.4f}")
        for acf_key in ["sp_acf", "dg_acf"]:
            b = self.real_baselines[acf_key]
            print(f"    {acf_key:15s}: MAE mean={b['mae_mean']:.4f}, σ={b['mae_std']:.4f}")
        print(f"    {'ddpm_mse':15s}: mean={self.real_mse_dist.mean():.6f}, "
              f"σ={self.real_mse_dist.std():.6f}")
        print(f"  {'─'*55}\n")

    # ──────────────────────────────────────────────
    # 评分
    # ──────────────────────────────────────────────

    def _score_ddpm_mse(self, fake_mse: float) -> float:
        """
        DDPM MSE 评分 — 双阶段混合评分:

        阶段 1 (在真实分布范围内): 分位数评分
          - fake_mse 在 real 分布中的排位 → 100 * (1 - percentile)

        阶段 2 (超出真实分布范围): 对数比率指数衰减
          - score = boundary_score * exp(-|log(fake_mse / real_boundary)|)
          - 使用 log-ratio 确保在 MSE 跨越数量级时仍有区分度

        这解决了 v1 中 fake MSE 天然比 real 高 10~100 倍导致的 0 分问题，
        同时在 fake MSE 接近 real 范围时给予合理的高分。
        """
        real_median = float(np.median(self.real_mse_dist))
        real_max = float(np.max(self.real_mse_dist))

        if fake_mse <= real_max:
            # 阶段 1: 在真实分布范围内，使用分位数
            rank = np.mean(self.real_mse_dist <= fake_mse)
            score = 100.0 * (1.0 - rank)
        else:
            # 阶段 2: 超出范围，使用 log-ratio 衰减
            # boundary_score: fake_mse 刚好等于 real_max 时的分数 (接近 0 分位)
            boundary_score = 5.0  # 给予 5 分作为刚出界的基础分
            log_ratio = np.log(fake_mse / real_max)
            # 衰减尺度 = 1.0 (每增加 e 倍 MSE，分数衰减约 63%)
            score = boundary_score * np.exp(-log_ratio / 1.0)

        return max(0.0, min(100.0, score))

    def _score_exponential(self, fake_val: float, baseline_mean: float, baseline_std: float) -> float:
        """
        自适应指数衰减评分:
          score = 100 * exp(-|fake - baseline_mean| / max(sigma, floor))

        sigma 使用真实数据的标准差。当 σ 过小时使用 floor 防止过度敏感。
        """
        diff = abs(fake_val - baseline_mean)
        # σ 下限: 防止 σ 过小导致微小偏差即 0 分
        sigma = max(baseline_std, abs(baseline_mean) * 0.1, 0.01)
        return max(0.0, 100.0 * np.exp(-diff / sigma))

    def _score_acf(self, fake_acf: np.ndarray, acf_key: str) -> float:
        """
        ACF 向量评分: MAE 与 baseline 对比。
        """
        baseline = self.real_baselines[acf_key]
        mean_acf = np.array(baseline["mean_acf"])
        mae = float(np.mean(np.abs(fake_acf - mean_acf)))

        sigma = max(baseline["mae_std"], baseline["mae_mean"] * 0.2, 0.005)
        return max(0.0, 100.0 * np.exp(-mae / sigma))

    def score_single_path(self, path_stats: dict, ddpm_mse: float) -> tuple[float, dict]:
        """
        对单条路径计算综合评分。

        Returns:
            (total_score, component_scores_dict)
        """
        scores = {}

        # 1. DDPM MSE (分位数)
        scores["ddpm_mse"] = self._score_ddpm_mse(ddpm_mse)

        # 2. Scalar Stylized Facts (自适应指数衰减)
        for key in ["sp_skew", "sp_kurt", "dg_skew", "dg_kurt", "uncond_corr", "tail_corr"]:
            b = self.real_baselines[key]
            scores[key] = self._score_exponential(path_stats[key], b["mean"], b["std"])

        # 3. ACF (向量 MAE)
        scores["sp_acf"] = self._score_acf(np.array(path_stats["sp_acf"]), "sp_acf")
        scores["dg_acf"] = self._score_acf(np.array(path_stats["dg_acf"]), "dg_acf")

        # Wasserstein 暂时置 0 (整体批次指标，在 generate_report 中计算)
        scores["wasserstein"] = 0.0

        total = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        return total, scores

    def score_batch(
        self,
        fake_csv_path: str,
        t_eval: int = 200,
        target_seq_len: int = None,
    ) -> dict:
        """
        对一个 fake 数据集进行批量评分。

        Returns:
            dict 包含:
              - per_path_scores: (N,) 每条路径的总分
              - per_path_components: list[dict] 每条路径的各指标分
              - batch_total: 批次总分
              - batch_components: 批次各指标平均分
              - wasserstein_score: 整体 Wasserstein 分
              - fake_norm, fake_raw, fake_stats, fake_mse: 原始数据
        """
        assert self._calibrated, "Must call calibrate_from_real() first!"

        print(f"\n[Scoring] Loading fake data: {fake_csv_path}")
        fake_norm, fake_raw = self.load_and_preprocess_data(fake_csv_path, target_seq_len=target_seq_len)
        N_fake = fake_raw.shape[0]
        L_target = fake_raw.shape[2]
        print(f"  Fake data: {N_fake} paths, shape={fake_raw.shape}")

        # 1. DDPM MSE
        print(f"  Computing DDPM MSE for {N_fake} fake paths (t={t_eval})...")
        fake_mse = self.compute_ddpm_mse_per_path(fake_norm, t_eval=t_eval)
        print(f"    Fake MSE: mean={fake_mse.mean():.6f}, "
              f"std={fake_mse.std():.6f}, "
              f"range=[{fake_mse.min():.6f}, {fake_mse.max():.6f}]")

        # 2. Stylized Facts
        print(f"  Computing Stylized Facts for {N_fake} fake paths...")
        fake_stats = compute_batch_stats(fake_raw)

        # 3. Wasserstein (整体批次)
        print(f"  Computing Joint Wasserstein distance...")
        wass_score = self._compute_wasserstein_score(fake_norm)

        # 4. 逐路径评分
        print(f"  Scoring {N_fake} paths...")
        per_path_scores = np.zeros(N_fake)
        per_path_components = []

        for i in range(N_fake):
            total, comp = self.score_single_path(fake_stats[i], fake_mse[i])
            # 加上 Wasserstein 的加权贡献 (整体分布指标，对所有路径统一)
            total += wass_score * self.WEIGHTS["wasserstein"]
            comp["wasserstein"] = wass_score
            per_path_scores[i] = total
            per_path_components.append(comp)

        # 5. 批次统计
        batch_components = {}
        for key in self.WEIGHTS:
            vals = [c[key] for c in per_path_components]
            batch_components[key] = float(np.mean(vals))

        batch_total = sum(batch_components[k] * self.WEIGHTS[k] for k in self.WEIGHTS)

        return {
            "per_path_scores": per_path_scores,
            "per_path_components": per_path_components,
            "batch_total": batch_total,
            "batch_components": batch_components,
            "wasserstein_score": wass_score,
            "fake_norm": fake_norm,
            "fake_raw": fake_raw,
            "fake_stats": fake_stats,
            "fake_mse": fake_mse,
            "L_target": L_target,
        }

    def _compute_wasserstein_score(self, fake_norm: torch.Tensor) -> float:
        """计算整体 Joint 1D Wasserstein 距离并转为评分。"""
        r_dev = self._real_norm.to(self.device)
        f_dev = fake_norm.to(self.device)
        joint_wass = calculate_1d_wasserstein(r_dev, f_dev)
        # 使用 σ=0.5 的指数衰减（Wasserstein 距离通常在 0~1 范围）
        score = max(0.0, 100.0 * np.exp(-joint_wass / 0.5))
        return score

    # ──────────────────────────────────────────────
    # 报告生成
    # ──────────────────────────────────────────────

    def generate_report(
        self,
        real_csv_path: str,
        fake_csv_path: str,
        t_eval: int = 200,
        json_output_path: str = None,
    ) -> dict:
        """
        完整评估流程: 校准 → 评分 → 报告。
        """
        # 1. 加载 Fake 数据确定序列长度
        print(f"\n[Phase 0] Detecting target sequence length from fake data...")
        df_fake_peek = pd.read_csv(fake_csv_path, nrows=1)
        sp_cols_peek = [c for c in df_fake_peek.columns if c.startswith("sp500_") and c.split("_")[-1].isdigit()]
        if len(sp_cols_peek) >= 2:
            L_target = len(sp_cols_peek)
            print(f"  Detected wide format: L_target = {L_target}")
        else:
            L_target = None
            print(f"  Long format detected, using model seq_len={self.seq_len}")

        # 2. 校准
        self.calibrate_from_real(real_csv_path, t_eval=t_eval, target_seq_len=L_target)

        # 3. 评分
        result = self.score_batch(fake_csv_path, t_eval=t_eval, target_seq_len=L_target)

        # 4. 真实数据自检分数（验证校准是否合理）
        print(f"\n[Self-Check] Computing real data self-scores...")
        real_self_scores = np.zeros(len(self.real_stats_list))
        for i in range(len(self.real_stats_list)):
            total, _ = self.score_single_path(self.real_stats_list[i], self.real_mse_dist[i])
            # 自检时 Wasserstein = 0（自身对自身距离为 0 → 得 100 分）
            total += 100.0 * self.WEIGHTS["wasserstein"]
            real_self_scores[i] = total

        print(f"  Real self-score: mean={real_self_scores.mean():.2f}, "
              f"std={real_self_scores.std():.2f}, "
              f"range=[{real_self_scores.min():.2f}, {real_self_scores.max():.2f}]")

        # 5. 组织报告
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "version": "v2",
            "t_eval": t_eval,
            "eval_seq_len": result["L_target"],
            "weights": self.WEIGHTS,
            "real_data": {
                "path": real_csv_path,
                "windows": len(self.real_stats_list),
                "ddpm_mse_mean": float(self.real_mse_dist.mean()),
                "ddpm_mse_std": float(self.real_mse_dist.std()),
                "self_score_mean": float(real_self_scores.mean()),
                "self_score_std": float(real_self_scores.std()),
                "baselines": self.real_baselines,
            },
            "fake_data": {
                "path": fake_csv_path,
                "paths": result["fake_raw"].shape[0],
                "ddpm_mse_mean": float(result["fake_mse"].mean()),
                "ddpm_mse_std": float(result["fake_mse"].std()),
            },
            "scores": {
                "batch_total": result["batch_total"],
                "batch_components": result["batch_components"],
                "per_path_mean": float(result["per_path_scores"].mean()),
                "per_path_std": float(result["per_path_scores"].std()),
                "per_path_min": float(result["per_path_scores"].min()),
                "per_path_max": float(result["per_path_scores"].max()),
                "wasserstein_score": result["wasserstein_score"],
            },
        }

        # 6. 打印报告
        self._print_markdown_report(report, result, real_self_scores)

        # 7. 保存 JSON
        if json_output_path:
            os.makedirs(os.path.dirname(os.path.abspath(json_output_path)), exist_ok=True)
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            print(f"\n[FinancialScorer] Report exported to: {json_output_path}")

        return report

    def _print_markdown_report(self, report: dict, result: dict, real_self_scores: np.ndarray):
        """打印格式化的 Markdown 报告。"""
        scores = report["scores"]
        batch_comp = scores["batch_components"]

        print(f"\n{'='*80}")
        print(f"          双资产模拟模型体检报告 v2 (Evaluation Report)")
        print(f"{'='*80}")
        print(f"评估时间:        {report['timestamp']}")
        print(f"评估序列长度:    {report['eval_seq_len']}")
        print(f"评估时间步 (t):  {report['t_eval']}")
        print(f"Real 数据:       {report['real_data']['path']} ({report['real_data']['windows']} 窗口)")
        print(f"Fake 数据:       {report['fake_data']['path']} ({report['fake_data']['paths']} 路径)")
        print(f"{'─'*80}")

        # 总分
        print(f"\n  【 综合保真度评分 (Fidelity Score) 】")
        print(f"    Fake 批次总分:          {scores['batch_total']:.2f} / 100.00")
        print(f"    Fake 逐路径平均:        {scores['per_path_mean']:.2f} ± {scores['per_path_std']:.2f}")
        print(f"    Fake 逐路径范围:        [{scores['per_path_min']:.2f}, {scores['per_path_max']:.2f}]")
        print(f"    Real 自检平均 (参考):   {report['real_data']['self_score_mean']:.2f} ± {report['real_data']['self_score_std']:.2f}")
        print()

        # 各指标表
        print(f"| 评估维度 | 指标 | 权重 | Baseline (μ±σ) | Fake 批次均分 | 单项评分 |")
        print(f"| :--- | :--- | :---: | :---: | :---: | :---: |")

        # DDPM MSE
        real_mse_mean = report['real_data']['ddpm_mse_mean']
        real_mse_std = report['real_data']['ddpm_mse_std']
        fake_mse_mean = report['fake_data']['ddpm_mse_mean']
        fake_mse_std = report['fake_data']['ddpm_mse_std']
        print(f"| 隐式物理分布 | DDPM MSE (t={report['t_eval']}) | {self.WEIGHTS['ddpm_mse']:.0%} | "
              f"{real_mse_mean:.4f}±{real_mse_std:.4f} | {fake_mse_mean:.4f}±{fake_mse_std:.4f} | "
              f"**{batch_comp['ddpm_mse']:.2f}** |")

        # Scalar indicators
        scalar_keys_labels = [
            ("sp_skew", "SP500 偏度 (Skewness)", "高阶矩匹配"),
            ("sp_kurt", "SP500 峰度 (Kurtosis)", "高阶矩匹配"),
            ("dg_skew", "DGS10 偏度 (Skewness)", "高阶矩匹配"),
            ("dg_kurt", "DGS10 峰度 (Kurtosis)", "高阶矩匹配"),
            ("uncond_corr", "无条件相关 (Corr)", "联合分布"),
            ("tail_corr", "尾部相关 (<-1.5σ)", "极端尾部"),
        ]
        for key, label, dim in scalar_keys_labels:
            b = self.real_baselines[key]
            # 计算 fake 的该指标均值
            fake_vals = np.array([s[key] for s in result["fake_stats"]])
            print(f"| {dim} | {label} | {self.WEIGHTS[key]:.1%} | "
                  f"{b['mean']:+.4f}±{b['std']:.4f} | "
                  f"{fake_vals.mean():+.4f}±{fake_vals.std():.4f} | "
                  f"**{batch_comp[key]:.2f}** |")

        # ACF
        for acf_key, label, dim in [
            ("sp_acf", "SP500 |r| ACF", "波动率聚集"),
            ("dg_acf", "DGS10 |r| ACF", "波动率聚集"),
        ]:
            b = self.real_baselines[acf_key]
            acf_str = ", ".join(f"{v:.3f}" for v in b["mean_acf"][:3]) + "..."
            print(f"| {dim} | {label} (Lag1-10) | {self.WEIGHTS[acf_key]:.0%} | "
                  f"MAE σ={b['mae_std']:.4f} | [{acf_str}] | "
                  f"**{batch_comp[acf_key]:.2f}** |")

        # Wasserstein
        print(f"| 分布距离 | Joint 1D Wasserstein | {self.WEIGHTS['wasserstein']:.0%} | "
              f"σ=0.5 (固定) | — | "
              f"**{batch_comp['wasserstein']:.2f}** |")

        print(f"{'='*80}")

        # Top 5 / Bottom 5
        sorted_idx = np.argsort(-result["per_path_scores"])
        print(f"\n  Top 5 路径:")
        for rank, idx in enumerate(sorted_idx[:5]):
            print(f"    #{rank+1}: Path {idx}, Score = {result['per_path_scores'][idx]:.2f}")
        print(f"\n  Bottom 5 路径:")
        for rank, idx in enumerate(sorted_idx[-5:]):
            print(f"    #{len(sorted_idx)-4+rank}: Path {idx}, Score = {result['per_path_scores'][idx]:.2f}")
        print()


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate simulated financial time series using 1D-DDPM and Stylized Facts (v2)"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--scaler", type=str, required=True, help="Path to scaler state (.pt)")
    parser.add_argument("--real", type=str, required=True, help="Path to real data CSV")
    parser.add_argument("--fake", type=str, required=True, help="Path to fake/generated data CSV")
    parser.add_argument("--t-eval", type=int, default=200, help="Timestep for DDPM MSE (default: 200)")
    parser.add_argument("--json", type=str, default=None, help="Path to save JSON report")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    scorer = FinancialScorer(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        device=args.device
    )

    scorer.generate_report(
        real_csv_path=args.real,
        fake_csv_path=args.fake,
        t_eval=args.t_eval,
        json_output_path=args.json
    )


if __name__ == "__main__":
    main()
