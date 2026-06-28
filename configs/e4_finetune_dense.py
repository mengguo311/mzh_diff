"""configs/e4_finetune_dense.py — E4 阶段2 微调【密集早存档】变体: 找 copy≤PASS × C2ST 甜点。

= e4_finetune 但 NUM_EPOCHS=250 + CHECKPOINT_EVERY=50(存 50/100/150/200/250)。
依据: ep199 native copy=0.028(刚过 PASS 线 0.02)、C2ST_新颖 0.548 PASS → 更早 ckpt 可能 copy≤0.02
(PASS)同时保住 C2ST(M1: copy 随训练单调升)。同 seed/数据/init → ep199 与既往一致, 早点内插。
"""
from configs.e4_finetune import OVERRIDES as _FT

OVERRIDES = {**_FT, "NUM_EPOCHS": 250, "CHECKPOINT_EVERY": 50}
