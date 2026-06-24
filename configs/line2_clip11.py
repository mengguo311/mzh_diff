"""configs/line2_clip11.py — 线2 方向A【clip11】变体(与 line2_clip15 受控对比, 仅 CLIP_RANGE 不同)。

早检发现 clip15 把峰度从 8 推到 67.6(过冲, 真实~19); clip→峰度曲线锚定 clip8→8 / clip15→67.6,
目标峰度≈19 → clip≈10-11。本档案 = line2 配方 + CLIP_RANGE=11, 与 clip15 并行跑做交叉验证。
依据: eval/docs/C1_improvement_analysis.md(方向A) + 本会话 ep999 早检。
"""
from configs.line2 import OVERRIDES as _LINE2

OVERRIDES = {**_LINE2, "CLIP_RANGE": 11.0}
