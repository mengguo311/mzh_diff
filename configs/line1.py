"""configs/line1.py — 线1【真实度优先 / 最大化第三方交叉评分(fool)】配方。

承 v9/v10 过平滑基座: 让样本落进真实窗 TCN-embedding 质心(nn_k5 小)+ 低峰度 + L=2048
+ DGS10 量化指纹; 容许复制率(复制率与 fool 解耦)。主记分牌 = 队友 v5 鉴别器 fool/combined。
依据: thirdparty/THIRDPARTY_CLAUDE.md + 归因(nn_k5 vs fool −0.839; 厚尾有害; DGS10 量化未开发)。

用法: CONFIG_PROFILE=line1 conda run -n ts_diffusion python train.py --run_name line1/l1_<name>
"""
OVERRIDES = dict(
    LINE="line1",
    PRIMARY_METRIC="external_fool",   # 主记分牌 = 队友 v5 fool/combined(内部闸门仍并报作参考)
    SEQ_LEN=2048, STRIDE=5,           # ≥1260, 避 L=512 窗长伪影(fool 崩到 4.3%)
    CLIP_RANGE=8.0,                   # 过平滑/贴近真实质心, 不攻厚尾(厚尾被推向合成负类)
    USE_AUX_LOSS=False,               # 关 stylized 厚尾辅助(v10_retrained 被它推向合成簇而 fool 垫底)
    USE_MIN_SNR=True,
    USE_CONTEXT_COND=False,           # 走 v9/v10 路线, 不用富条件
    USE_BLOCK_BOOTSTRAP=False,
    NUM_EPOCHS=20000,                 # v9_20k 量级(充分过平滑); 可按需下调
    DGS10_QUANTIZE=0.01,              # ★头号杠杆: 生成端把 DGS10 吸附到 0.01 网格匹配真实量化指纹
)                                     #   (需在 generate*.py 接线; 亦可零训练后处理现有 CSV 先验证)
