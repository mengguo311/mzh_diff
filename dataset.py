"""
dataset.py — 数据管道与 Z-score 归一化
阶段一：TimeSeriesScaler (Z-score + Clipping) + TimeSeriesDataset (滑动窗口)

关键设计:
  - 绝对禁止 MinMax Scaling（会抹杀金融的尖峰厚尾）
  - 双通道独立 Z-score 标准化 + ±5σ 硬截断
  - stride=5 滑动窗口，阻断窗口间高相关性
  - 输出张量形状: (Batch, 2, 128)
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


# ──────────────────────────────────────────────
# TimeSeriesScaler: Z-score + Clipping
# ──────────────────────────────────────────────

class TimeSeriesScaler:
    """
    双通道独立 Z-score 标准化器。
    
    transform:          x_norm = clamp((x - mean) / std, -clip, +clip)
    inverse_transform:  x_real = x * std + mean  (不反截断，允许生成超出 ±5σ)
    
    Attributes:
        mean: (2,) 双通道均值
        std:  (2,) 双通道标准差
        clip_range: float, 硬截断阈值
    """

    def __init__(self, clip_range: float = config.CLIP_RANGE):
        self.clip_range = clip_range
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, data: np.ndarray):
        """
        从原始数据计算双通道的均值和标准差。
        
        Args:
            data: numpy array, shape (N, 2), 原始金融时序
        """
        assert data.ndim == 2 and data.shape[1] == config.CHANNELS, \
            f"Expected shape (N, {config.CHANNELS}), got {data.shape}"

        self.mean = torch.tensor(data.mean(axis=0), dtype=torch.float32)  # (2,)
        self.std = torch.tensor(data.std(axis=0), dtype=torch.float32)    # (2,)
        
        # 防止除零（理论上不会发生，但做安全检查）
        self.std = torch.clamp(self.std, min=1e-8)

        print(f"[Scaler] Fitted on {data.shape[0]} samples")
        for i, name in enumerate(["sp500", "DGS10"]):
            print(f"  {name}: mean={self.mean[i]:.6f}, std={self.std[i]:.6f}")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Z-score 标准化 + 硬截断。
        
        Args:
            x: (N, 2) 或 (2, L) 或 (B, 2, L) — 支持多种形状
        Returns:
            标准化 + clipping 后的张量，值域 [-clip_range, +clip_range]
        """
        assert self.mean is not None, "Scaler not fitted! Call fit() first."
        
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        
        if x.ndim == 2 and x.shape[1] == config.CHANNELS:
            # (N, 2) — 逐列标准化
            x_norm = (x - mean.unsqueeze(0)) / std.unsqueeze(0)
        elif x.ndim == 2 and x.shape[0] == config.CHANNELS:
            # (2, L) — 逐通道标准化
            x_norm = (x - mean.unsqueeze(1)) / std.unsqueeze(1)
        elif x.ndim == 3:
            # (B, 2, L) — 逐通道标准化
            x_norm = (x - mean.view(1, -1, 1)) / std.view(1, -1, 1)
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        return torch.clamp(x_norm, -self.clip_range, self.clip_range)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        从标准化空间还原到真实金融量级。
        不做反截断——生成样本偶尔超出 ±5σ 是合理的。
        
        Args:
            x: (B, 2, L) 标准化后的张量
        Returns:
            还原后的张量，真实的收益率/差分值
        """
        assert self.mean is not None, "Scaler not fitted! Call fit() or load() first."
        
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        
        if x.ndim == 3:
            return x * std.view(1, -1, 1) + mean.view(1, -1, 1)
        elif x.ndim == 2 and x.shape[0] == config.CHANNELS:
            return x * std.unsqueeze(1) + mean.unsqueeze(1)
        elif x.ndim == 2:
            return x * std.unsqueeze(0) + mean.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

    def save(self, path: str):
        """保存 scaler 参数 (mean, std) 为 .pt 文件。"""
        torch.save({
            "mean": self.mean,
            "std": self.std,
            "clip_range": self.clip_range,
        }, path)
        print(f"[Scaler] Parameters saved to {path}")

    def load(self, path: str):
        """从 .pt 文件加载 scaler 参数。"""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.mean = state["mean"]
        self.std = state["std"]
        self.clip_range = state["clip_range"]
        print(f"[Scaler] Parameters loaded from {path}")
        for i, name in enumerate(["sp500", "DGS10"]):
            print(f"  {name}: mean={self.mean[i]:.6f}, std={self.std[i]:.6f}")


# ──────────────────────────────────────────────
# TimeSeriesDataset: 滑动窗口 + Z-score
# ──────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    金融时间序列数据集。
    
    处理流程:
        1. 读取 CSV → 提取 sp500, DGS10 两列
        2. 前向填充缺失值 (ffill + bfill)
        3. fit TimeSeriesScaler → Z-score + clip 全量数据
        4. 滑动窗口 (stride=5) 预计算所有窗口起始索引
        5. __getitem__ 返回 (2, 128) 张量
    
    Args:
        data_path: CSV 文件路径
        seq_len: 滑动窗口长度 (默认 128)
        stride: 滑动步长 (默认 5)
        scaler: 预先 fit 好的 Scaler（用于生成时保持一致），
                如果为 None 则自动 fit
    """

    def __init__(
        self,
        data_path: str = config.DATA_PATH,
        seq_len: int = config.SEQ_LEN,
        stride: int = config.STRIDE,
        scaler: Optional[TimeSeriesScaler] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride

        # ── 1. 读取数据 ──
        df = pd.read_csv(data_path, index_col=0)
        assert "sp500" in df.columns and "DGS10" in df.columns, \
            f"CSV must contain 'sp500' and 'DGS10' columns, got {list(df.columns)}"

        # ── 2. 前向填充缺失值 ──
        df = df[["sp500", "DGS10"]].ffill().bfill()
        raw_data = df.values.astype(np.float32)  # (N, 2)

        print(f"[Dataset] Loaded {len(raw_data)} rows from {data_path}")

        # ── 3. Scaler: fit + transform ──
        if scaler is None:
            self.scaler = TimeSeriesScaler()
            self.scaler.fit(raw_data)
        else:
            self.scaler = scaler

        # 转为张量并标准化: (N, 2)
        data_tensor = torch.tensor(raw_data, dtype=torch.float32)
        self.data = self.scaler.transform(data_tensor)  # (N, 2), 值域 [-5, 5]

        # ── 4. 预计算滑动窗口起始索引 ──
        n = len(self.data)
        self.indices = list(range(0, n - seq_len + 1, stride))

        print(f"[Dataset] {len(self.indices)} windows "
              f"(seq_len={seq_len}, stride={stride}, total_rows={n})")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        返回单个样本: (2, 128) — 通道优先。
        """
        start = self.indices[idx]
        window = self.data[start : start + self.seq_len]  # (128, 2)
        return window.T  # (2, 128) — 通道 × 序列长度

    def get_scaler(self) -> TimeSeriesScaler:
        """获取 scaler 实例（用于保存或传递给生成阶段）。"""
        return self.scaler
