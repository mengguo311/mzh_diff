"""configs/line2_clip11_boot.py — 线2 方向A clip11 + A2 block-bootstrap(攻弥散 C2ST floor)。

clip11 已是 line2 最优(复制率0% + C2ST_新颖~0.60 + 综合89.7), 但 0.60→0.495 残余是【弥散多特征
微弱分离】= 数据稀缺指纹(诊断: 无单特征 AUC>0.58)。唯一既不抬复制率又加独立信号的杠杆 =
moving-block bootstrap(BLOCK_LEN=192 真实块首尾相接造【新宏观次序】, 保块内 stylized fact;
boot 窗非复制 max|pearson|0.13-0.17 已验)。配方 = clip11 + USE_BLOCK_BOOTSTRAP。
判据(line2 北极星): 复制率仍≈0%(boot 不抬记忆)+ C2ST_新颖 是否破 0.60 平台。
"""
from configs.line2 import OVERRIDES as _LINE2

OVERRIDES = {**_LINE2, "CLIP_RANGE": 11.0, "USE_BLOCK_BOOTSTRAP": True}
