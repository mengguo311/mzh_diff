"""configs/line1.py — 线1【真实度优先 / 直追真实市场 / 同时增强 OUR 鉴别器】配方(修正版 2026-06-24)。

承 v9/v10 全窗(L=2048)基座, 容许复制(不缩窗、不罚记忆)。放开 CLIP 峰度天花板 + 开 stylized
辅助攻波动聚集 + DGS10 量化吸附补真实结构。主记分牌 = realism vector 否决式判定(eval/realism_board.py)。
**hw01_discriminator 不是基准, 其 fool 率仅作外部参照并报, 绝不优化**(追 fool=过平滑迎合=远离真实市场)。
依据: eval/docs/line1_realism_direction.md(§4)。numpy 实证: CLIP 是 ch0 峰度天花板(clip8→7.92 / clip15→12.90 / clip20→真实18.8)。

红线: 禁 cosine/zero-SNR(打崩尾部)、禁 latent mixup(削尾)、禁把任何鉴别信号回喂训练损失
(除非先过时间 held-out + 缓冲带闸, 见路线图 §5.2 防 GAN 泄露)。
"""
OVERRIDES = dict(
    LINE="line1",
    PRIMARY_METRIC="realism_vector",  # 否决式真实度向量(eval/realism_board.py), 非 external_fool
    SEQ_LEN=2048, STRIDE=5,           # 全窗, 承 v9/v10 主线, 容许复制不缩窗
    CLIP_RANGE=15.0,                  # 放开 ch0 峰度天花板(clip8→7.92 / clip15→12.90 逼近真实 18.8)
    #                                   ⚠️ 若实现 per-channel clip: ch0=15 / ch1=8(DGS10 clip12 即 plateau)
    USE_AUX_LOSS=True,                # 开 stylized |r|-ACF + 二阶差分能量, 攻波动聚集/regime
    #   配套(config.py 已有): AUX_ACF_WEIGHT/AUX_ROUGH_WEIGHT/AUX_ABAR_MIN 门控; 新增 kurt 项见路线图 §G3
    USE_MIN_SNR=True,
    USE_CONTEXT_COND=False,           # 走 v9/v10 路线, 不用富条件(那是 line2)
    USE_BLOCK_BOOTSTRAP=False,
    DGS10_QUANTIZE=0.01,              # 接线 generate.py(补真实量化结构); 非真实度排序主因
    GEN_NUM_STEPS=500,                # G4 实证: steps200→500 把 regime run_len 49.8→37.4 逼近真实(eta1), 不破坏峰度
    NUM_EPOCHS=2000,                  # M1 实证质量 ~1500-2000ep 饱和, 省半(旧 20000 充分过平滑无用)
    CHECKPOINT_EVERY=500,             # 密集存档: ep500 即查峰度是否随 clip15 抬升(否则clip非病立即停)
    # 容许复制: 不设复制率阈值、不进判定门; 但 realism_board 并报 all-vs-novel 污染 gap 防 RESTING ON COPIES
)
