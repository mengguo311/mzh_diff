"""configs/e4_finetune.py — E4(B1)阶段2: US SP500↔DGS10【微调】配方。

= line2_clip11 配方(US 单源), 配合 train.py --init_from <阶段1预训ckpt> 仅载权重续训。
微调步数少(防过拟合 7 个 US 独立窗; 预训已学跨市场共享 stylized 律, 此处只贴 US 例)。
评估同口径 forensic_suite(US 标定); 防泄露: 微调后 memorization 对原 US bank 复制率须不升。

用法: CONFIG_PROFILE=e4_finetune conda run -n ts_diffusion python train.py --model dit-s \
        --init_from logs/e4/pretrain_multiasset/checkpoint_final.pt --run_name e4/finetune_us
"""
from configs.line2 import OVERRIDES as _LINE2

OVERRIDES = {
    **_LINE2,
    "CLIP_RANGE": 11.0,
    "LINE": "e4",
    "USE_MULTIASSET": False,       # 微调只用 US 主 CSV
    "NUM_EPOCHS": 800,             # 预训后轻微调(M1: 从零 ~1500 饱和; 微调更少)
    "CHECKPOINT_EVERY": 200,       # 200/400/600/800 供早停选点
}
