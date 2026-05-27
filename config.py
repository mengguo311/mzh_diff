"""
config.py — 全局超参数 & 设备探测
所有模块从此处导入配置，保持松耦合。
"""

import os
import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_PATH       = "/home/u00134/data/train_sp500_us10y.csv"
LOG_DIR         = os.path.expanduser("~/src/logs")
OUTPUT_DIR      = os.path.expanduser("~/src/output")

# ──────────────────────────────────────────────
# Data Pipeline
# ──────────────────────────────────────────────
SEQ_LEN         = 128       # 2^7, 完美适配 U-Net 逐级下采样
CHANNELS        = 2         # sp500 日收益率 + DGS10 日差分
STRIDE          = 5         # 按周滑动，阻断数据泄露

# ──────────────────────────────────────────────
# Z-score Scaler
# ──────────────────────────────────────────────
CLIP_RANGE      = 5.0       # 硬截断阈值 ±5σ

# ──────────────────────────────────────────────
# 1D U-Net
# ──────────────────────────────────────────────
CHANNEL_DIMS    = [256, 512, 1024]    # 编码器通道数序列
TIME_EMB_DIM    = 512               # 时间嵌入维度
NUM_GROUPS      = 8                 # GroupNorm 分组数

# ──────────────────────────────────────────────
# DDPM Scheduler
# ──────────────────────────────────────────────
T               = 1000       # 总扩散步数
BETA_START      = 1e-4      # β₁
BETA_END        = 0.02      # β_T

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE      = 64
NUM_EPOCHS      = 5000
LEARNING_RATE   = 2e-4
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0       # 梯度裁剪阈值
EMA_DECAY       = 0.995     # 指数移动平均衰减率
CHECKPOINT_EVERY = 100       # 每 N epoch 保存 checkpoint
SEED            = 42

# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
NUM_SIMULATIONS = 2560
GEN_BATCH_SIZE  = 256       # 分批生成，避免 OOM

# ──────────────────────────────────────────────
# Device Auto-Detection
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    _gpu_name = torch.cuda.get_device_name(0)
    _gpu_count = torch.cuda.device_count()
    print(f"[Config] CUDA detected: {_gpu_name} x{_gpu_count}")
else:
    DEVICE = torch.device("cpu")
    print("[Config] No CUDA available, using CPU")
