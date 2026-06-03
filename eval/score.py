#!/usr/bin/env python3
"""
score.py — 双资产模拟生成数据体检报告打分系统
使用训练好的一维扩散模型（1D-DDPM）结合量化金融 Stylized Facts（典型事实）对第三方生成的数据进行多维度综合评分。
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
from eval.metrics import calculate_1d_wasserstein, calculate_mmd


class FinancialScorer:
    """
    1D-DDPM 基于隐式物理特征与量化金融典型事实的评估与打分系统。
    """
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
                
                # 同步更新全局 config 模块中的值，以便其他内部层匹配
                config.SEQ_LEN = self.seq_len
                config.CHANNELS = self.channels
                config.CHANNEL_DIMS = self.channel_dims
                config.TIME_EMB_DIM = self.time_emb_dim
                config.T = self.T
                config.BETA_START = self.beta_start
                config.BETA_END = self.beta_end
                
                print(f"  Applied Model Config: seq_len={self.seq_len}, channel_dims={self.channel_dims}, T={self.T}")
            except Exception as e:
                print(f"  [Warning] Failed to load config.json, using default values. Error: {e}")
                
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
            print("  Loaded normal model weights (EMA weights not found).")
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

    def compute_distribution_metrics(self, real_norm: torch.Tensor, fake_norm: torch.Tensor) -> dict:
        """
        在 GPU 上计算 Wasserstein 距离和 MMD 距离。
        """
        r_dev = real_norm.to(self.device)
        f_dev = fake_norm.to(self.device)
        
        # 1D Wasserstein (单通道和联合通道)
        sp_wass = calculate_1d_wasserstein(r_dev[:, 0, :], f_dev[:, 0, :])
        dg_wass = calculate_1d_wasserstein(r_dev[:, 1, :], f_dev[:, 1, :])
        joint_wass = calculate_1d_wasserstein(r_dev, f_dev)
        
        # MMD (把每个路径展平为 C * L 的特征向量)
        B_r, C, L = r_dev.shape
        B_f = f_dev.shape[0]
        r_feat = r_dev.reshape(B_r, C * L)
        f_feat = f_dev.reshape(B_f, C * L)
        
        mmd_val = calculate_mmd(r_feat, f_feat)
        
        return {
            "sp_wasserstein": sp_wass,
            "dg_wasserstein": dg_wass,
            "joint_wasserstein": joint_wass,
            "mmd": mmd_val
        }

    def load_and_preprocess_data(self, csv_path: str, target_seq_len: int = None) -> tuple[torch.Tensor, np.ndarray]:
        """
        加载并预处理数据。
        返回:
            x_normalized: (Batch, 2, seq_len) PyTorch 张量
            x_changes:    (Batch, 2, seq_len) NumPy 数组 (真实量级)
        """
        df = pd.read_csv(csv_path)
        
        # 检测是否为宽表 (DDPM/SABR 生成格式)
        is_wide = False
        cols = df.columns.tolist()
        
        # 支持 sp500_0, sp500_level_0 等格式
        sp_cols = sorted([c for c in cols if (c.startswith("sp500_") or c.startswith("sp500_level_")) and c.split("_")[-1].isdigit()],
                         key=lambda c: int(c.split("_")[-1]))
        dg_cols = sorted([c for c in cols if (c.startswith("dgs10_") or c.startswith("dgs10_level_")) and c.split("_")[-1].isdigit()],
                         key=lambda c: int(c.split("_")[-1]))
                         
        if len(sp_cols) >= 2 and len(dg_cols) >= 2:
            is_wide = True
            
        if is_wide:
            # 检查是否为 level (价格/收益率绝对水平) 格式
            is_level = any("level" in c for c in sp_cols)
            sp_data = df[sp_cols].values.astype(np.float32)
            dg_data = df[dg_cols].values.astype(np.float32)
            
            if is_level:
                # 按照 price_table_converter.py 中的基准初始值还原
                sp500_initial = 500.0
                dgs10_initial = 2.0
                
                sp_changes = np.zeros_like(sp_data)
                dg_changes = np.zeros_like(dg_data)
                
                # t = 0
                sp_changes[:, 0] = sp_data[:, 0] / sp500_initial - 1.0
                dg_changes[:, 0] = dg_data[:, 0] - dgs10_initial
                
                # t > 0
                sp_changes[:, 1:] = sp_data[:, 1:] / sp_data[:, :-1] - 1.0
                dg_changes[:, 1:] = np.diff(dg_data, axis=1)
                
                print(f"  [Wide Format] Converted level columns to daily returns/differences.")
            else:
                sp_changes = sp_data
                dg_changes = dg_data
                print(f"  [Wide Format] Loaded returns/differences directly.")
                
            x_changes = np.stack([sp_changes, dg_changes], axis=1) # (N, 2, seq_len)
            
        else:
            # 长表格式 (例如真实的测试集历史数据)
            col_pairs = [
                ("sp500", "DGS10"),
                ("SP500", "DGS10"),
                ("Asset_1_Return", "Asset_2_Return"),
                ("Asset_1_Level", "Asset_2_Level"),
                ("asset_1", "asset_2"),
                ("Asset1", "Asset2"),
            ]
            col1, col2 = None, None
            for c1, c2 in col_pairs:
                if c1 in df.columns and c2 in df.columns:
                    col1, col2 = c1, c2
                    break
            if col1 is None or col2 is None:
                numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in {"day", "index", "date"}]
                if len(numeric_cols) < 2:
                    raise ValueError(f"CSV must contain at least 2 numeric asset columns, got {list(df.columns)}")
                col1, col2 = numeric_cols[0], numeric_cols[1]
                
            df_clean = df[[col1, col2]].copy().ffill().bfill()
            
            # 判断是否需要转换为 changes
            # 如果均值显著大于1.0，通常是指数水平价格/收益率水平
            is_level = df_clean[col1].abs().mean() > 1.0
            if is_level:
                df_changes = pd.DataFrame(index=df_clean.index)
                df_changes[col1] = df_clean[col1].pct_change()
                df_changes[col2] = df_clean[col2].diff()
                df_changes = df_changes.dropna()
                print(f"  [Long Format] Converted level columns to daily returns/differences.")
            else:
                df_changes = df_clean
                print(f"  [Long Format] Loaded returns/differences directly.")
                
            raw_data = df_changes.values.astype(np.float32) # (N, 2)
            
            # 滑动窗口划分 (与 TimeSeriesDataset 一致)
            n_days = len(raw_data)
            stride = 5
            seq_len_to_use = target_seq_len if target_seq_len is not None else self.seq_len
            window_indices = list(range(0, n_days - seq_len_to_use + 1, stride))
            if len(window_indices) == 0:
                raise ValueError(f"Data length {n_days} is less than required sequence length {seq_len_to_use}")
                
            windows = []
            for start in window_indices:
                win = raw_data[start : start + seq_len_to_use] # (seq_len_to_use, 2)
                windows.append(win.T) # (2, seq_len_to_use)
                
            x_changes = np.stack(windows, axis=0) # (Batch, 2, seq_len_to_use)
            print(f"  [Long Format] Divided historical series into {len(x_changes)} sliding windows of length {seq_len_to_use}.")
            
        # 转换并归一化
        x_tensor = torch.tensor(x_changes, dtype=torch.float32)
        x_normalized = self.scaler.transform(x_tensor)
        
        return x_normalized, x_changes

    @torch.no_grad()
    def compute_ddpm_mse(self, x_normalized: torch.Tensor, t_eval: int = 200, batch_size: int = 64) -> float:
        """
        计算 DDPM 隐式物理法则评分 (Noise Prediction MSE)。
        """
        self.model.eval()
        B = x_normalized.shape[0]
        L = x_normalized.shape[2]
        
        # 核心逻辑：确保输入序列长度是 2**d 的倍数，以配合 U-Net 对齐下采样/上采样
        multiple = 2 ** len(self.channel_dims)
        if L % multiple != 0:
            L_clean = (L // multiple) * multiple
            x_input = x_normalized[:, :, :L_clean]
            print(f"  [DDPM MSE] Input sequence length {L} is not a multiple of {multiple}. Slicing to {L_clean} for U-Net compatibility.")
        else:
            x_input = x_normalized
            
        total_loss = 0.0
        num_samples = 0
        
        for i in range(0, B, batch_size):
            x0 = x_input[i:i+batch_size].to(self.device)
            n_batch = x0.shape[0]
            
            t = torch.full((n_batch,), t_eval, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0)
            
            # 使用 DDPMScheduler 前向加噪
            xt = self.scheduler.q_sample(x0, t, noise)
            
            # 提取初始条件
            c = x0[:, :, 0]
            # 模型预测噪声
            noise_pred = self.model(xt, t, c)
            
            # 计算批次 MSE Loss (reduction='mean')
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            total_loss += loss.item() * n_batch
            num_samples += n_batch
            
        return total_loss / num_samples

    def compute_acf_vector(self, data: np.ndarray, max_lag: int = 10) -> np.ndarray:
        """
        计算每个路径的绝对收益序列的自相关系数 (ACF)，并取平均。
        """
        N, L = data.shape
        abs_data = np.abs(data)
        
        acf_all = np.zeros((N, max_lag))
        for i in range(N):
            path = abs_data[i]
            path_mean = np.mean(path)
            path_var = np.var(path)
            if path_var < 1e-8:
                continue
            for lag in range(1, max_lag + 1):
                cov = np.mean((path[lag:] - path_mean) * (path[:-lag] - path_mean))
                acf_all[i, lag-1] = cov / path_var
                
        return np.mean(acf_all, axis=0)

    def compute_stylized_facts(self, x_changes: np.ndarray) -> dict:
        """
        计算量化金融典型事实 (Stylized Facts) 指标。
        """
        # x_changes: (Batch, 2, seq_len)
        sp_data = x_changes[:, 0, :]  # (Batch, seq_len)
        dg_data = x_changes[:, 1, :]  # (Batch, seq_len)
        
        # 1. Moments (偏度和峰度) - 基于扁平化的全局序列
        sp_flat = sp_data.flatten()
        dg_flat = dg_data.flatten()
        
        sp_skew_val = skew(sp_flat)
        sp_kurt_val = kurtosis(sp_flat)  # excess kurtosis
        dg_skew_val = skew(dg_flat)
        dg_kurt_val = kurtosis(dg_flat)
        
        # 2. Volatility Clustering (Lag 1-10 ACF)
        sp_acf = self.compute_acf_vector(sp_data, max_lag=10)
        dg_acf = self.compute_acf_vector(dg_data, max_lag=10)
        
        # 3. Unconditional Correlation
        uncond_corr = np.corrcoef(sp_flat, dg_flat)[0, 1]
        if np.isnan(uncond_corr):
            uncond_corr = 0.0
            
        # 4. Tail Dependence (极端下跌下的条件相关系数)
        # 极端下跌定义为 S&P 500 低于其均值 -1.5 个标准差
        mu_sp = np.mean(sp_flat)
        std_sp = np.std(sp_flat)
        tail_threshold = mu_sp - 1.5 * std_sp
        
        tail_mask = sp_flat < tail_threshold
        if np.sum(tail_mask) >= 10:
            tail_corr = np.corrcoef(sp_flat[tail_mask], dg_flat[tail_mask])[0, 1]
            if np.isnan(tail_corr):
                tail_corr = 0.0
        else:
            tail_corr = 0.0
            
        return {
            "sp_skew": float(sp_skew_val),
            "sp_kurt": float(sp_kurt_val),
            "dg_skew": float(dg_skew_val),
            "dg_kurt": float(dg_kurt_val),
            "sp_acf": sp_acf.tolist(),
            "dg_acf": dg_acf.tolist(),
            "uncond_corr": float(uncond_corr),
            "tail_corr": float(tail_corr)
        }

    def calculate_fidelity_score(self, real_facts: dict, fake_facts: dict, real_mse: float, fake_mse: float, dist_metrics: dict = None) -> tuple[float, dict]:
        """
        计算模型各项指标的偏差，并采用指数衰减形式给出 0-100 的保真度打分 (Fidelity Score)。
        """
        scores = {}
        
        # 1. DDPM MSE 打分 (权重 20% / 10%)
        mse_diff = abs(fake_mse - real_mse)
        scores["ddpm_mse"] = max(0.0, 100.0 * np.exp(-mse_diff / (real_mse + 1e-8)))
        
        # 2. Moments 偏度与峰度打分 (每个指标占 5%)
        sp_skew_diff = abs(fake_facts["sp_skew"] - real_facts["sp_skew"])
        scores["sp_skew"] = max(0.0, 100.0 * np.exp(-sp_skew_diff / 0.5))
        
        sp_kurt_diff = abs(fake_facts["sp_kurt"] - real_facts["sp_kurt"])
        scores["sp_kurt"] = max(0.0, 100.0 * np.exp(-sp_kurt_diff / 1.0))
        
        dg_skew_diff = abs(fake_facts["dg_skew"] - real_facts["dg_skew"])
        scores["dg_skew"] = max(0.0, 100.0 * np.exp(-dg_skew_diff / 0.5))
        
        dg_kurt_diff = abs(fake_facts["dg_kurt"] - real_facts["dg_kurt"])
        scores["dg_kurt"] = max(0.0, 100.0 * np.exp(-dg_kurt_diff / 1.0))
        
        # 3. Volatility Clustering ACF 打分 (每个资产占 10% / 5%)
        real_sp_acf = np.array(real_facts["sp_acf"])
        fake_sp_acf = np.array(fake_facts["sp_acf"])
        sp_acf_mae = np.mean(np.abs(fake_sp_acf - real_sp_acf))
        scores["sp_acf"] = max(0.0, 100.0 * np.exp(-sp_acf_mae / 0.05))
        
        real_dg_acf = np.array(real_facts["dg_acf"])
        fake_dg_acf = np.array(fake_facts["dg_acf"])
        dg_acf_mae = np.mean(np.abs(fake_dg_acf - real_dg_acf))
        scores["dg_acf"] = max(0.0, 100.0 * np.exp(-dg_acf_mae / 0.05))
        
        # 4. Tail Dependence 尾部相关性打分 (权重 20% / 15%)
        tail_diff = abs(fake_facts["tail_corr"] - real_facts["tail_corr"])
        scores["tail_corr"] = max(0.0, 100.0 * np.exp(-tail_diff / 0.2))
        
        # 5. Unconditional Correlation 无条件相关性打分 (权重 20% / 15%)
        uncond_diff = abs(fake_facts["uncond_corr"] - real_facts["uncond_corr"])
        scores["uncond_corr"] = max(0.0, 100.0 * np.exp(-uncond_diff / 0.1))
        
        # 如果包含高级分布度量，进行加权整合
        if dist_metrics is not None:
            # Wasserstein 距离打分 (衰减尺度为 0.2)
            scores["joint_wasserstein"] = max(0.0, 100.0 * np.exp(-dist_metrics["joint_wasserstein"] / 0.2))
            # MMD 距离打分 (衰减尺度为 0.1)
            scores["mmd"] = max(0.0, 100.0 * np.exp(-dist_metrics["mmd"] / 0.1))
            
            # 使用重平衡的权重
            weights = {
                "ddpm_mse": 0.10,
                "sp_skew": 0.05,
                "sp_kurt": 0.05,
                "dg_skew": 0.05,
                "dg_kurt": 0.05,
                "sp_acf": 0.05,
                "dg_acf": 0.05,
                "tail_corr": 0.15,
                "uncond_corr": 0.15,
                "joint_wasserstein": 0.15,
                "mmd": 0.20
            }
        else:
            weights = {
                "ddpm_mse": 0.20,
                "sp_skew": 0.05,
                "sp_kurt": 0.05,
                "dg_skew": 0.05,
                "dg_kurt": 0.05,
                "sp_acf": 0.10,
                "dg_acf": 0.10,
                "tail_corr": 0.20,
                "uncond_corr": 0.20
            }
        
        total_score = sum(scores[key] * weights[key] for key in weights)
        return total_score, scores

    def generate_report(self, real_csv_path: str, fake_csv_path: str, t_eval: int = 200, json_output_path: str = None) -> dict:
        """
        加载真实与虚假数据，执行评估打分，并生成格式化报告。
        """
        # 1. 优先加载 Fake 数据，确定评估的目标序列长度 L
        print(f"\n[Phase 1] Processing Evaluated Data ({fake_csv_path})...")
        fake_norm, fake_raw = self.load_and_preprocess_data(fake_csv_path)
        L_target = fake_raw.shape[2] # 获取 fake 数据的实际长度 (如 1260)
        fake_mse = self.compute_ddpm_mse(fake_norm, t_eval=t_eval)
        fake_facts = self.compute_stylized_facts(fake_raw)
        
        # 2. 根据 L 加载 Real 数据，确保两者窗口长度完全一致
        print(f"\n[Phase 2] Processing Real Data Baseline ({real_csv_path})...")
        real_norm, real_raw = self.load_and_preprocess_data(real_csv_path, target_seq_len=L_target)
        real_mse = self.compute_ddpm_mse(real_norm, t_eval=t_eval)
        real_facts = self.compute_stylized_facts(real_raw)
        
        # 3. 计算 GPU 加速版本的高级概率分布度量
        print(f"\n[Phase 2.5] Computing Advanced Distribution Metrics on GPU...")
        dist_metrics = self.compute_distribution_metrics(real_norm, fake_norm)
        
        print("\n[Phase 3] Scoring Model Performance...")
        total_score, component_scores = self.calculate_fidelity_score(
            real_facts, fake_facts, real_mse, fake_mse, dist_metrics=dist_metrics
        )
        
        # 组织报告数据
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "t_eval": t_eval,
            "eval_seq_len": L_target,
            "real_data": {
                "path": real_csv_path,
                "samples": real_norm.shape[0],
                "ddpm_mse": real_mse,
                "stylized_facts": real_facts
            },
            "fake_data": {
                "path": fake_csv_path,
                "samples": fake_norm.shape[0],
                "ddpm_mse": fake_mse,
                "stylized_facts": fake_facts
            },
            "distribution_metrics": dist_metrics,
            "scores": {
                "total": total_score,
                "components": component_scores
            }
        }
        
        # 打印排版精美的 Markdown 表格
        self._print_markdown_report(report)
        
        # 保存为 JSON 报告
        if json_output_path:
            os.makedirs(os.path.dirname(os.path.abspath(json_output_path)), exist_ok=True)
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            print(f"\n[FinancialScorer] Report successfully exported to: {json_output_path}")
            
        return report

    def _print_markdown_report(self, report: dict):
        real_data = report["real_data"]
        fake_data = report["fake_data"]
        scores = report["scores"]
        
        real_facts = real_data["stylized_facts"]
        fake_facts = fake_data["stylized_facts"]
        comp_scores = scores["components"]
        
        # 格式化列表
        sp_real_acf_str = ", ".join(f"{v:.4f}" for v in real_facts["sp_acf"][:3]) + "..."
        sp_fake_acf_str = ", ".join(f"{v:.4f}" for v in fake_facts["sp_acf"][:3]) + "..."
        dg_real_acf_str = ", ".join(f"{v:.4f}" for v in real_facts["dg_acf"][:3]) + "..."
        dg_fake_acf_str = ", ".join(f"{v:.4f}" for v in fake_facts["dg_acf"][:3]) + "..."
        
        print("\n" + "=" * 80)
        print("                   双资产模拟模型体检报告 (Evaluation Report)                   ")
        print("=" * 80)
        print(f"评估时间:      {report['timestamp']}")
        print(f"评估时序长度:  {report['eval_seq_len']}")
        print(f"评估时间步 (t): {report['t_eval']}")
        print(f"Real 数据路径:  {real_data['path']} (样本数: {real_data['samples']})")
        print(f"Fake 数据路径:  {fake_data['path']} (样本数: {fake_data['samples']})")
        print("-" * 80)
        print(f"【 综合保真度评分 (Fidelity Score) 】  >>>  {scores['total']:.2f} / 100.00")
        print("-" * 80)
        
        # Markdown 表格
        print("| 评估维度 | 指标名称 (Metric) | 真实基准 (Real) | 评估模型 (Fake) | 绝对偏差 | 单项评分 (Score) |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: |")
        
        # 1. DDPM MSE
        mse_real = real_data["ddpm_mse"]
        mse_fake = fake_data["ddpm_mse"]
        mse_diff = abs(mse_fake - mse_real)
        print(f"| 隐式物理分布 | DDPM Noise MSE (t={report['t_eval']}) | {mse_real:.6f} | {mse_fake:.6f} | {mse_diff:.6f} | {comp_scores['ddpm_mse']:.2f} |")
        
        # 2. Moments
        print(f"| Moment Matching | S&P 500 Skewness | {real_facts['sp_skew']:.4f} | {fake_facts['sp_skew']:.4f} | {abs(fake_facts['sp_skew'] - real_facts['sp_skew']):.4f} | {comp_scores['sp_skew']:.2f} |")
        print(f"| Moment Matching | S&P 500 Kurtosis | {real_facts['sp_kurt']:.4f} | {fake_facts['sp_kurt']:.4f} | {abs(fake_facts['sp_kurt'] - real_facts['sp_kurt']):.4f} | {comp_scores['sp_kurt']:.2f} |")
        print(f"| Moment Matching | DGS10 Skewness | {real_facts['dg_skew']:.4f} | {fake_facts['dg_skew']:.4f} | {abs(fake_facts['dg_skew'] - real_facts['dg_skew']):.4f} | {comp_scores['dg_skew']:.2f} |")
        print(f"| Moment Matching | DGS10 Kurtosis | {real_facts['dg_kurt']:.4f} | {fake_facts['dg_kurt']:.4f} | {abs(fake_facts['dg_kurt'] - real_facts['dg_kurt']):.4f} | {comp_scores['dg_kurt']:.2f} |")
        
        # 3. Volatility Clustering (ACF)
        sp_acf_mae = np.mean(np.abs(np.array(fake_facts["sp_acf"]) - np.array(real_facts["sp_acf"])))
        dg_acf_mae = np.mean(np.abs(np.array(fake_facts["dg_acf"]) - np.array(real_facts["dg_acf"])))
        print(f"| 波动率聚集 | S&P 500 ACF (Lag 1-3) | {sp_real_acf_str} | {sp_fake_acf_str} | MAE: {sp_acf_mae:.4f} | {comp_scores['sp_acf']:.2f} |")
        print(f"| 波动率聚集 | DGS10 ACF (Lag 1-3) | {dg_real_acf_str} | {dg_fake_acf_str} | MAE: {dg_acf_mae:.4f} | {comp_scores['dg_acf']:.2f} |")
        
        # 4. Correlation & Tail Dependence
        print(f"| 联合分布关联 | Unconditional Corr | {real_facts['uncond_corr']:.4f} | {fake_facts['uncond_corr']:.4f} | {abs(fake_facts['uncond_corr'] - real_facts['uncond_corr']):.4f} | {comp_scores['uncond_corr']:.2f} |")
        print(f"| 极端尾部相关 | Tail Correlation (< -1.5σ) | {real_facts['tail_corr']:.4f} | {fake_facts['tail_corr']:.4f} | {abs(fake_facts['tail_corr'] - real_facts['tail_corr']):.4f} | {comp_scores['tail_corr']:.2f} |")
        
        # 5. Advanced Distribution Metrics
        if "distribution_metrics" in report:
            dm = report["distribution_metrics"]
            print(f"| 分布距离度量 | SP500 1D Wasserstein | 0.000000 | {dm['sp_wasserstein']:.6f} | {dm['sp_wasserstein']:.6f} | - |")
            print(f"| 分布距离度量 | DGS10 1D Wasserstein | 0.000000 | {dm['dg_wasserstein']:.6f} | {dm['dg_wasserstein']:.6f} | - |")
            print(f"| 分布距离度量 | Joint 1D Wasserstein | 0.000000 | {dm['joint_wasserstein']:.6f} | {dm['joint_wasserstein']:.6f} | {comp_scores['joint_wasserstein']:.2f} |")
            print(f"| 分布距离度量 | Path-Joint MMD (RBF) | 0.000000 | {dm['mmd']:.6f} | {dm['mmd']:.6f} | {comp_scores['mmd']:.2f} |")
            
        print("=" * 80)
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate simulated financial time series using 1D-DDPM and Stylized Facts")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--scaler", type=str, required=True, help="Path to scaler state (.pt)")
    parser.add_argument("--real", type=str, required=True, help="Path to real data CSV")
    parser.add_argument("--fake", type=str, required=True, help="Path to fake/generated data CSV")
    parser.add_argument("--t-eval", type=int, default=200, help="Timestep for DDPM MSE evaluation (default: 200)")
    parser.add_argument("--json", type=str, default=None, help="Path to save output JSON report")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cpu)")
    
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
