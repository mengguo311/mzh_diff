"""configs/e4_pretrain.py — E4(B1)阶段1: 多资产跨市场【预训练】配方。

= line2_clip11 配方(DiT-S/L512/CLIP11/富条件) + USE_MULTIASSET 池化 US+JP+UK+EU 同构对。
动机(research/E0_results.md): E0/E0b 证 floor=数据稀缺(C2ST 0.94 计量先验 vs 0.60 DiT),
唯一治本=注入真·独立窗。**只改一个变量(加跨市场真数据)** 对照既往从零 clip11(C2ST 0.601),
隔离"真数据能否破 floor"。阶段2 用 configs/e4_finetune + train.py --init_from 在 US 微调。

用法: CONFIG_PROFILE=e4_pretrain conda run -n ts_diffusion python train.py --model dit-s \
        --run_name e4/pretrain_multiasset
"""
from configs.line2 import OVERRIDES as _LINE2

OVERRIDES = {
    **_LINE2,
    "CLIP_RANGE": 11.0,            # line2 最优(峰度≈19)
    "LINE": "e4",
    "USE_MULTIASSET": True,        # 池化 us/jp/uk/eu(per-market z-score, 窗不跨市场)
    "NUM_EPOCHS": 1500,            # 池化数据 ~3x US → 较 line2 少; 中间 ckpt 供早停扫描
    "CHECKPOINT_EVERY": 500,       # 500/1000/1500 密集存档
}
