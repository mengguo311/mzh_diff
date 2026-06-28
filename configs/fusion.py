"""configs/fusion.py — v14-fusion: 多通道(收益率曲线)融合线, 同时攻真实度+复制率。

新线(非 line1/line2), git tag **v14-fusion**。融合 3 根经实证的独立杠杆, 在锁死的
记忆化↔过平滑 Pareto 上把【两端】同时推到更好工作点(不声称解决数据稀缺):
  - 跨资产真实度: +DGS2_diff 通道引入 2s10s 收益率曲线斜率(经典 regime/衰退信号),
    直击 A1/C1 反复栽的 long-range/regime 短板(单通道物理上不可能有的新真实度维度)。
  - 厚尾真实度: CLIP_RANGE=20 (line1 实证解除峰度天花板, sp500 数据侧达原生 kurt 18.8;
    A3 验 clip20 > 全通道 max|z|=19.43 → 每通道命中原生峰度, 无削尾无过冲)。
  - 新颖/低复制: 联合 block-bootstrap(A2 验 copy_rate≈0% + 跨通道相关 0.799→0.810 保持 +
    块内 stylized 退化≤6.9%; 独立重采样对照崩到 0.001 → 实证【必须联合】) + context-cond 富条件
    (C1 实证修签名 SigP 0.003→0.106、长程 high_vol 0.27→0.49) + L=512 缩窗(独立段)。

诚实边界(务必记住): 多通道【本身不增独立时间窗】(同一日历, 3ch indep512 反 28.8→23.8),
不打穿 Pareto; 复制率改善全靠 joint-boot+context-cond+缩窗这三根独立杠杆, 不能归功于"加通道"。
T10YIE/DGS30 已剔除(2003边界砍62%行 / DGS2-DGS30 0.725冗余)。诚实闸门须对 3ch+L512 重标定,
绝不套旧 0.0%/0.496/0.116。DGS2_diff vs DGS10 真实 corr≈0.81(强联动, 但~34%独立方差=斜率信号)。
"""
OVERRIDES = dict(
    LINE="fusion",
    PRIMARY_METRIC="internal_honest",          # 北极星=诚实闸门(复制率/C2ST_新颖/SigP_新颖, 3ch 重标定)
    # ── 多通道 (收益率曲线) ──
    CHANNELS=3,
    CHANNEL_COLS=["sp500", "DGS10", "DGS2_diff"],
    QUANTIZE_GRID={"DGS10": 0.01, "DGS2_diff": 0.01},      # 两利率通道吸附 0.01 网格(真实量化指纹); sp500 永不
    OUTPUT_PREFIX={"sp500": "sp500", "DGS10": "dgs10", "DGS2_diff": "dgs2"},  # 保下游 sp500_*/dgs10_* 兼容
    # ── 缩窗 + 独立段 (承 v13 A1) ──
    SEQ_LEN=512,
    STRIDE=2,
    # ── 厚尾 (承 line1) ──
    CLIP_RANGE=20.0,
    USE_AUX_LOSS=True,                          # stylized-fact 辅助(|r|-ACF + roughness)
    # ── 低复制: 联合 block-bootstrap + 富条件 (承 v13 A2 + C1) ──
    USE_BLOCK_BOOTSTRAP=True,
    BLOCK_LEN=192,
    BOOT_FRAC=0.3,
    USE_CONTEXT_COND=True,
    N_CTX_FEAT=8,                               # COND_DIM = 3*8 = 24 (config 应用档案后自动重算)
    # ── 抗记忆化 + 训练 ──
    USE_SIG_MMD=False,                          # v12 实证 Sig-MMD = no-op (且其 loss 硬编码2通道, 多通道须关)
    WEIGHT_DECAY=1e-2,                          # 10x 正则对抗数据稀缺记忆化
    NUM_EPOCHS=2000,                            # M1 实证 ep~1000-2000 饱和(clip20 解锁厚尾后须复核轨迹)
    CHECKPOINT_EVERY=500,                       # 密集存档供 ckpt 选择
)
