"""configs/line1_clip20.py — line1 真实度 recipe 的 clip20 变体(攻 tail_kurt 临门一脚)。

承 line1(v9/v10 全窗 + AUX + DGS10量化 + steps500), 仅 CLIP_RANGE 15→20:
numpy 实证 clip20→真实数据峰度 18.8(vs clip15 12.9), 有望把模型峰度 13.3→~17-18 进 realism_board
带 [17.5,20.6] → tail_kurt 转 PASS。ch1(DGS10)diff 极少超 12σ, clip20 对其几乎无影响(可接受微损)。

用法: CONFIG_PROFILE=line1_clip20 conda run -n ts_diffusion python train.py --model dit-b --run_name line1/l1_clip20
"""
from configs.line1 import OVERRIDES as _BASE

OVERRIDES = {**_BASE, "CLIP_RANGE": 20.0}
