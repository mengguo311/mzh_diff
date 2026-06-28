"""configs/line2_clip20.py — 线2 方向A【clip20】变体(逼近峰度≈真实19)。

收敛交叉验证: clip8→峰度8.06 / clip15→12.7(均欠真实19), 单调递增 → 需更高 clip。
numpy 上界: clip20→真实保留峰度 18.8 ≈ 真实。本档案 = line2 配方 + CLIP_RANGE=20。
注: 模型通常仅达 clip 天花板的~68%(clip15 天花板18.4 但模型只到12.7), clip20 若仍欠
    则下一步叠 dataset 极端窗上采样。生成端 x0_clamp 须 ≥25(允许 ±20σ 数据+余量)。
"""
from configs.line2 import OVERRIDES as _LINE2

OVERRIDES = {**_LINE2, "CLIP_RANGE": 20.0}
