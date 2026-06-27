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
        for i, name in enumerate(config.CHANNEL_COLS):
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
        assert len(self.mean) == len(config.CHANNEL_COLS), \
            f"scaler 通道数 {len(self.mean)} != CHANNEL_COLS {len(config.CHANNEL_COLS)} (加载了错误通道集的 scaler?)"
        for i, name in enumerate(config.CHANNEL_COLS):
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

        # ── 1. 读取数据 (多通道: 主CSV + 按需 FRED 辅助源, 按 CHANNEL_COLS 选列/定通道序) ──
        df_main = pd.read_csv(data_path, index_col=0)
        need_fred = any(config.CHANNEL_SOURCES.get(c) == "fred" for c in config.CHANNEL_COLS)
        if need_fred:
            df_fred = pd.read_csv(config.FRED_PATH, index_col=0)
            df = pd.concat([df_main, df_fred], axis=1)         # 两文件 index 已对齐项目交易日
        else:
            df = df_main
        miss = [c for c in config.CHANNEL_COLS if c not in df.columns]
        assert not miss, f"缺通道列 {miss}; 可选 {list(df.columns)}"

        # ── 2. 选通道(顺序=通道编号) + 截到所有通道非NaN起始 + 起点后内部缺口 ffill ──
        df = df[config.CHANNEL_COLS]
        valid = df.notna().all(axis=1)                          # 所有通道都有值的行
        first = valid.idxmax()                                  # 最早全通道非NaN起点 (利用 1976/1977/2003 边界)
        df = df.loc[first:].ffill().bfill()                     # 起点后补内部缺口(如 DGS30 缺口); 不跨序列首回填
        raw_data = df.values.astype(np.float32)                 # (N, C)

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

    @staticmethod
    def _ctx_stats(s: torch.Tensor) -> torch.Tensor:
        """v13 C1 — 单通道上下文富统计 (N_CTX_FEAT 维): std/|r|均值/均值/|r|-acf1/skew/kurt/末值/d2能量。"""
        a = s.abs()
        am = a - a.mean()
        var = (am * am).mean().clamp(min=1e-8)
        acf1 = (am[1:] * am[:-1]).mean() / var
        sc = s - s.mean()
        sd = sc.std().clamp(min=1e-8)
        z = sc / sd
        d2 = s[2:] - 2.0 * s[1:-1] + s[:-2]
        return torch.stack([s.std(), a.mean(), s.mean(), acf1,
                            (z ** 3).mean(), (z ** 4).mean() - 3.0, s[-1], (d2 * d2).mean()])

    def _cond_from(self, window: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        """根据开关返回条件向量: C1 富条件(前置上下文统计; 无前置→零) 或 原 2 维初值。"""
        if config.USE_CONTEXT_COND:
            if ctx is None:                                   # 起始窗/bootstrap 无前置上下文 → null
                return torch.zeros(config.CHANNELS * config.N_CTX_FEAT)
            return torch.cat([self._ctx_stats(ctx[:, ch]) for ch in range(config.CHANNELS)])
        return window[0]                                      # (C,) 原起点初值

    def _make_boot_window(self) -> torch.Tensor:
        """
        v13 A2 — on-the-fly moving-block bootstrap 增广窗 (标准化空间, (seq_len, 2))。
        取 (ceil(L/B)+1) 个随机【真实块】(各长 BLOCK_LEN) 首尾相接, 再【随机裁剪】出 L 长
        —— 整块搬运保块内 stylized fact, 随机裁剪让接缝位置逐窗不同(避免固定周期伪结构),
        只造新的宏观次序排列。块取自已标准化(±clip)的 self.data, 故无需再裁剪/归一化。
        """
        L, B = self.seq_len, config.BLOCK_LEN
        N = self.data.shape[0]
        n_blocks = (L + B - 1) // B + 1                      # 多取一块以便随机移接缝
        starts = [int(torch.randint(0, N - B + 1, (1,))) for _ in range(n_blocks)]
        cat = torch.cat([self.data[s:s + B] for s in starts], dim=0)   # (n_blocks*B, 2) > L
        off = int(torch.randint(0, cat.shape[0] - L + 1, (1,)))
        return cat[off:off + L]                              # (seq_len, 2)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回单个样本及其起点条件向量: (x, c)
        - x: (2, seq_len) 通道优先的时序数据
        - c: (2,) 序列起点的初始条件向量
        v13 A2: 若 USE_BLOCK_BOOTSTRAP, 以 BOOT_FRAC 概率改返回一个 on-the-fly bootstrap 增广窗。
        v13 C1: 若 USE_CONTEXT_COND, c 改为前置上下文窗的富统计向量 (bootstrap/起始窗→null)。
        """
        if config.USE_BLOCK_BOOTSTRAP and bool(torch.rand(1) < config.BOOT_FRAC):
            window = self._make_boot_window()                 # (seq_len, 2)
            ctx = None                                        # bootstrap 窗无真实前置上下文
        else:
            start = self.indices[idx]
            window = self.data[start : start + self.seq_len]  # (seq_len, 2)
            ctx = self.data[start - self.seq_len : start] if start >= self.seq_len else None
        x = window.T                       # (2, seq_len)
        c = self._cond_from(window, ctx)   # (2,) 或 (2*N_CTX_FEAT,)
        return x, c

    def get_scaler(self) -> TimeSeriesScaler:
        """获取 scaler 实例（用于保存或传递给生成阶段）。"""
        return self.scaler
