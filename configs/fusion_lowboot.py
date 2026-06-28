"""configs/fusion_lowboot.py — v14-fusion 降 bootstrap 变体, 修长程签名。

承 fusion 全配置(3通道收益率曲线 + clip20厚尾 + 富条件 + 自回归), 仅把
BOOT_FRAC 0.3→0.15: 终裁实证 joint-bootstrap(0.3)为换 0.0% 复制率牺牲了长程签名
(30% 训练窗块乱序、无长程结构 → 模型未学到 2048 尺度连贯路径 → AR SigP 塌 0.003)。
降到 0.15 保大部分抗记忆(预期 copy 仍 <10%), 恢复长程连贯 → 目标 AR SigP 回升~0.1(如 C1)。
ctx 反馈钳位修复(generate_autoregressive --ctx_clamp_pct)已就位, 新模型 AR 直接可用。
"""
from configs.fusion import OVERRIDES as _BASE

OVERRIDES = {**_BASE,
             "BOOT_FRAC": 0.15,    # 0.3→0.15: 减块乱序占比, 恢复长程签名 (留部分抗记忆)
             }
