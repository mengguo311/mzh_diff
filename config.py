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
SEQ_LEN         = 512        # v13 A1 缩窗: 2048→512 (独立段 7.2→28.8, 直击数据稀缺记忆化; num_patches=512/16=32 自适应)
CHANNELS        = 2         # sp500 日收益率 + DGS10 日差分
STRIDE          = 2         # v13 A1: 缩窗后 stride 2 维持训练样本量 (~7100 窗)

# ── 多通道 (v14-fusion): 通道→源文件/列名 映射 + per-channel 量化/输出名 ──
# base 默认=原 2 通道 (sp500 + DGS10[本列已是日差分, mean≈0 std0.067]); 多通道由 configs/fusion.py 档案注入。
# 改这些【不影响】line1/line2 的 2 通道复现 (它们不设 CHANNEL_COLS → 用此默认)。
FRED_PATH       = "/home/u00134/data/fred_treasury.csv"   # 多通道辅助源 (DGS2_diff/DGS30_diff/...)
CHANNEL_COLS    = ["sp500", "DGS10"]                       # 通道顺序=训练通道编号 (决定 ch0/ch1/...)
CHANNEL_SOURCES = {"sp500": "main", "DGS10": "main",       # 每列来自主CSV('main')还是FRED辅助源('fred')
                   "DGS2_diff": "fred", "DGS30_diff": "fred",
                   "DFII10_diff": "fred", "T10YIE_diff": "fred"}
QUANTIZE_GRID   = {}                                       # per-channel 生成端量化网格 (利率→0.01, sp500永不); 空=回退 legacy DGS10_QUANTIZE
OUTPUT_PREFIX   = {"sp500": "sp500", "DGS10": "dgs10"}     # 生成CSV列名前缀 (保下游 eval 的 sp500_*/dgs10_* 兼容)

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
WEIGHT_DECAY    = 1e-2       # v12 抗记忆化: 1e-3→1e-2 (10x 正则, 配合 DiT-S 缩容对抗数据稀缺记忆化)
GRAD_CLIP       = 1.0       # 梯度裁剪阈值
EMA_DECAY       = 0.995     # 指数移动平均衰减率
CHECKPOINT_EVERY = 1000       # v12: 1000 密集存档, 供采样侧早停扫描 (用 memorization/novelty_rerank 选 ckpt)
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
USE_SIG_MMD      = False     # v12 根因消融: 关 Sig-MMD (v11 失败后默认关; 其余配方与 v11 一致)
SIG_MMD_WEIGHT   = 0.05      # MMD² 的损失权重 (从小起步)
SIG_DEPTH        = 2         # 截断签名深度 (向量化实现固定 = 2)
SIG_SUB_LEN      = 128       # 子窗长度 Lsub
SIG_N_SUB        = 2         # 每步随机取的子窗个数 K

# ──────────────────────────────────────────────
# v13 A2: moving-block bootstrap 数据增广  —— 见 dataset.py
# ──────────────────────────────────────────────
# 对【已标准化】真实序列做整块搬运的滑动块自助: 每个增广窗由若干随机真实块首尾相接、再随机裁剪而成
# —— 保块内 stylized fact(杠杆/峰度/|r|-ACF), 只造【新的宏观次序排列】→ 加独立训练信号; 不靠任何
# 凸组合/平均(后者必削尾), 不插值/不加噪。on-the-fly 每次新随机 → 无限变体、不可被记忆。默认关。
USE_BLOCK_BOOTSTRAP = False   # 开/关 block bootstrap 增广 (v13 A2)
BLOCK_LEN           = 192     # 块长 (装得下杠杆/波动聚集等块内结构)
BOOT_FRAC           = 0.3     # __getitem__ 中返回 bootstrap 窗的概率 (其余为真实窗)

# ──────────────────────────────────────────────
# v13 C1: context-conditioning 富条件  —— 见 dataset.py / dit1d.py / generate.py
# ──────────────────────────────────────────────
# 把条件从 c=window[0](2 维初值, 与波动相关≈0.02 形同无用)升级为【前置上下文窗的富统计向量】
# (每通道 N_CTX_FEAT 维: std/|r|均值/均值/|r|-acf1/skew/kurt/末值/d2能量), 让模型从"上下文状态"
# 泛化续写而非背诵; 缩窗 L=512 后配合【自回归拼接】重建 2048 长程(自回归采样器在 step3a 落地)。
# 必做消融(防 Sig-MMD 式 no-op): 比较 zero-ctx vs real-ctx 生成的 regime 分布, 无差异即停。默认关。
USE_CONTEXT_COND = True                                  # 开/关 富条件 (v13 C1)
N_CTX_FEAT       = 8                                       # 每通道上下文特征数
COND_DIM         = (CHANNELS * N_CTX_FEAT) if USE_CONTEXT_COND else CHANNELS  # 条件维度 (C*8 富条件 或 C 初值)

# ──────────────────────────────────────────────
# E4 (B1): 多资产跨市场预训练  —— 见 dataset.MultiAssetDataset / eval/fetch_multiasset.py
# ──────────────────────────────────────────────
# 注入【真·独立宏观窗】(US 外的同构对: 股指日 log 收益 + 本国 10Y 日差分), 把独立窗 ~29→~77(L512)。
# 阶段1: USE_MULTIASSET=True 池化多市场预训(per-market z-score, 窗不跨市场); 阶段2: 关多资产,
# 用 --init_from 载预训权重在 US SP500↔DGS10 微调。E0 实证 floor=数据稀缺 → 唯一治本方向。默认关。
USE_MULTIASSET     = False                                # 开/关 多资产池化预训 (E4)
MULTIASSET_DIR     = "/home/u00134/data/multiasset"       # 外部市场 CSV 目录 (绝不入主 CSV)
MULTIASSET_MARKETS = ["us", "jp", "uk", "eu"]             # 池化市场 (us=主CSV; jp/uk/eu=multiasset/*.csv)

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

# ──────────────────────────────────────────────
# 双线并行: 配置档案 (CONFIG_PROFILE=line1|line2 → configs/<profile>.py 的 OVERRIDES)
#   line1 = 真实度优先(外部 fool); line2 = 内部诚实/新颖(续 v13)。
#   详见 eval/docs/dual_track_structure.md。不设档案则用上面的 base 默认。
# ──────────────────────────────────────────────
LINE           = "base"                # 当前线 (base/line1/line2), 供 run 命名空间/记分牌识别
PRIMARY_METRIC = "internal_honest"     # 主记分牌 (external_fool / internal_honest)
DGS10_QUANTIZE = None                  # line1 特性: 生成端把 DGS10 吸附到该网格 (如 0.01); None=关
X0_CLAMP_SIGMA = None                  # ⑦ 生成端 x0 钳位(标准化 σ, 如 20); None=关, 治自回归罕见单窗发散
import os as _os
_PROFILE = _os.environ.get("CONFIG_PROFILE")
if _PROFILE:
    import importlib as _il
    _ov = _il.import_module(f"configs.{_PROFILE}").OVERRIDES
    for _k, _v in _ov.items():
        globals()[_k] = _v
    COND_DIM = (CHANNELS * N_CTX_FEAT) if USE_CONTEXT_COND else CHANNELS   # 应用档案后重算 (CHANNELS 已被档案更新)
    print(f"[Config] 档案 CONFIG_PROFILE={_PROFILE}: LINE={LINE} SEQ_LEN={SEQ_LEN} "
          f"CLIP_RANGE={CLIP_RANGE} AUX={USE_AUX_LOSS} CTX={USE_CONTEXT_COND} "
          f"COND_DIM={COND_DIM} DGS10_Q={DGS10_QUANTIZE} PRIMARY={PRIMARY_METRIC}")
