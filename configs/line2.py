"""configs/line2.py — 线2【内部诚实/新颖优先, 续 v13-c1 既往方向】配方。

C1 富条件(COND_DIM=16)+ 自回归 512→2048 + 攻厚尾(放宽 clip 破峰度天花板)。
主记分牌 = 内部诚实闸门(C2ST_新颖/复制率/SigP_新颖)。
依据: eval/docs/C1_improvement_analysis.md §7(M1 实证: 质量 ep~1500-2000 饱和; 峰度天花板=clip)。

用法: CONFIG_PROFILE=line2 conda run -n ts_diffusion python train.py --run_name line2/l2_<name>
"""
OVERRIDES = dict(
    LINE="line2",
    PRIMARY_METRIC="internal_honest", # 主记分牌 = C2ST_新颖/复制率/SigP_新颖(外部 fool 并报作参考)
    SEQ_LEN=512, STRIDE=2,
    CLIP_RANGE=15.0,                  # 攻厚尾(M1 实证峰度天花板由 clip=8 造成; numpy: clip15→真实峰度~18)
    USE_AUX_LOSS=True,
    USE_MIN_SNR=True,
    USE_CONTEXT_COND=True,            # C1 富条件 + 自回归
    USE_BLOCK_BOOTSTRAP=False,        # 可叠 A2: 设 True
    NUM_EPOCHS=2000,                  # M1 实证: 质量 ep~1500-2000 即饱和, 省半
)
