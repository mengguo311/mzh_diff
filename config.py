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
SEQ_LEN         = 2048       # 2^10, 完美适配 U-Net 逐级下采样
CHANNELS        = 2         # sp500 日收益率 + DGS10 日差分
STRIDE          = 5         # 按周滑动，阻断数据泄露

# ──────────────────────────────────────────────
# Z-score Scaler
# ──────────────────────────────────────────────
CLIP_RANGE      = 8.0       # 硬截断阈值 ±8σ (v10 Phase2: 由 5→8 放宽，保留驱动波动爆发的尾部)

# ──────────────────────────────────────────────
# 1D U-Net
# ──────────────────────────────────────────────
CHANNEL_DIMS    = [64, 128, 256, 512, 1024]  # 5层下采样通道序列，Receptive Field=64，捕捉宏观长程特征
TIME_EMB_DIM    = 256               # 时间嵌入维度
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
WEIGHT_DECAY    = 1e-3
GRAD_CLIP       = 1.0       # 梯度裁剪阈值
EMA_DECAY       = 0.995     # 指数移动平均衰减率
CHECKPOINT_EVERY = 2000       # 每 N epoch 保存 checkpoint (v11 验证训练: 2000, 12h 内多存档)
SEED            = 42

# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
NUM_SIMULATIONS = 5120
GEN_BATCH_SIZE  = 64        # 分批生成，避免 OOM

# ──────────────────────────────────────────────
# Generation — DDIM 采样 (v10 默认，零成本优化)
# ──────────────────────────────────────────────
# 由 Phase 1 采样扫描确定的最优工作点 (eta=1, steps=200, w=1, 总分 47.16→51.01)。
# 这些是 generate.py 的默认参考值，命令行可覆盖。
GEN_ETA              = 1.0   # DDIM 随机性 (0=确定性 偏平滑; 1≈DDPM 注入纹理/波动)
GEN_NUM_STEPS        = 200   # DDIM 采样步数 (拐点 ~200)
GEN_GUIDANCE_SCALE   = 1.0   # CFG 权重 (w=1 纯有条件，无放大)

# ──────────────────────────────────────────────
# Phase 2 (v10) 训练损失增强开关  —— 见 losses.py
# ──────────────────────────────────────────────
# 针对"过平滑/欠离散"病根。逐项可关闭以做消融。基线复现请全部置 False 且 CLIP_RANGE=5.0。
USE_MIN_SNR      = True      # P3: min-SNR-γ 加权 (ε-pred)
MIN_SNR_GAMMA    = 5.0

USE_AUX_LOSS     = True      # P1: stylized-fact 辅助损失 (波动聚集 + roughness)
AUX_ACF_WEIGHT   = 0.05      # |r| ACF L1 权重
AUX_ROUGH_WEIGHT = 0.05      # 二阶差分能量相对差距权重
AUX_ACF_MAX_LAG  = 5         # ACF 滞后阶数
AUX_ABAR_MIN     = 0.1       # 仅对 ᾱ_t > 此阈值 (低噪声步) 施加辅助损失，保证 x̂₀ 可靠

# ──────────────────────────────────────────────
# v11 训练损失增强: 可微 Sig-MMD 辅助损失  —— 见 losses.sig_mmd_loss
# ──────────────────────────────────────────────
# 路径签名是随机过程"律"的可微指纹: 同一散度既做训练损失又做假数据取证打分(签名脊梁)。
# 在随机短子窗上对 x̂₀ 与真实 x₀ 的 depth-2 time-augmented 签名分布做无偏 MMD²。
# 与 stylized acf+rough 互补, 复用 AUX_ABAR_MIN 门控 (仅低噪声步, x̂₀ 可靠)。
USE_SIG_MMD      = True      # 开/关 可微 Sig-MMD 辅助损失
SIG_MMD_WEIGHT   = 0.05      # MMD² 的损失权重 (从小起步)
SIG_DEPTH        = 2         # 截断签名深度 (向量化实现固定 = 2)
SIG_SUB_LEN      = 128       # 子窗长度 Lsub
SIG_N_SUB        = 2         # 每步随机取的子窗个数 K

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
