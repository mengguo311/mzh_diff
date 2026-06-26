"""configs/line1_clip20_acf.py — line1 clip20 + 扩长 |r|-ACF 攻 regime(波动持续性过强)。

承 line1_clip20(厚尾已达成: 峰度20.15进带), 仅强化 stylized 辅助的 |r|-ACF 长程项:
clip20 模型 mean_run_len 63 vs 真33(纯采样修不动, 见 clip20_regime_scan), 须训练侧。
扩 AUX_ACF_MAX_LAG 5→20(教模型匹配真实波动持续性的长程衰减) + 权重 0.05→0.1。
仍是对 x̂₀ 的 stylized 正则(对比同 batch 真实 x0 的 |r|-ACF, 非 regime 指标本身 → 不 Goodhart)。
"""
from configs.line1_clip20 import OVERRIDES as _BASE

OVERRIDES = {**_BASE,
             "AUX_ACF_MAX_LAG": 20,    # 5→20: 捕捉长程波动持续性衰减(治 run_len 过长)
             "AUX_ACF_WEIGHT": 0.1,    # 0.05→0.1: 加压
             }
