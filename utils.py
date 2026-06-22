"""
utils.py — 辅助工具
包含：EMA、随机种子、Checkpoint 管理、设备获取。
"""

import os
import copy
import json
import random
import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    """全局设置随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Utils] Random seed set to {seed}")


# ──────────────────────────────────────────────
# Exponential Moving Average (EMA)
# ──────────────────────────────────────────────

class EMA:
    """
    指数移动平均 (Exponential Moving Average)。
    维护模型参数的滑动平均副本，用于推理时获得更稳定的输出。
    
    用法:
        ema = EMA(model, decay=0.995)
        # 训练循环中:
        ema.update(model)
        # 推理时:
        ema.apply(model)
        output = model(x)
        ema.restore(model)
    """

    def __init__(self, model: nn.Module, decay: float = config.EMA_DECAY):
        self.decay = decay
        # 深拷贝所有可训练参数的值
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    def update(self, model: nn.Module):
        """用当前模型参数更新 shadow 参数。"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module):
        """将 EMA 权重应用到模型（保存当前权重以便恢复）。"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """恢复 apply() 之前的原始权重。"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """返回 EMA shadow 参数的字典（用于 checkpoint 保存）。"""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict):
        """从 checkpoint 恢复 EMA shadow 参数。"""
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ──────────────────────────────────────────────
# Run Directory & Checkpoint Management
# ──────────────────────────────────────────────

def create_run_dir(run_name: Optional[str] = None, base_dir: str = config.LOG_DIR) -> str:
    """
    在 logs/ 下创建一个带时间戳的运行目录。
    
    Returns:
        run_dir: 创建的目录路径，如 ~/src/logs/run_20260527_164400/
    """
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Utils] Run directory: {run_dir}")
    return run_dir


def save_config_snapshot(run_dir: str):
    """将当前超参数快照保存为 JSON，便于溯源。"""
    snapshot = {
        "data_path":        config.DATA_PATH,
        "seq_len":          config.SEQ_LEN,
        "channels":         config.CHANNELS,
        "cond_dim":         config.COND_DIM,
        "stride":           config.STRIDE,
        "clip_range":       config.CLIP_RANGE,
        "channel_dims":     config.CHANNEL_DIMS,
        "time_emb_dim":     config.TIME_EMB_DIM,
        "T":                config.T,
        "beta_start":       config.BETA_START,
        "beta_end":         config.BETA_END,
        "batch_size":       config.BATCH_SIZE,
        "num_epochs":       config.NUM_EPOCHS,
        "learning_rate":    config.LEARNING_RATE,
        "weight_decay":     config.WEIGHT_DECAY,
        "grad_clip":        config.GRAD_CLIP,
        "ema_decay":        config.EMA_DECAY,
        "seed":             config.SEED,
        "device":           str(config.DEVICE),
    }
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[Utils] Config snapshot saved to {path}")


def save_checkpoint(
    run_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    loss_history: list,
    is_final: bool = False,
):
    """
    保存训练 checkpoint。
    
    包含: model_state_dict, ema_state_dict, optimizer_state_dict, epoch, loss_history
    """
    tag = "final" if is_final else f"epoch_{epoch:04d}"
    path = os.path.join(run_dir, f"checkpoint_{tag}.pt")
    
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "ema_state_dict":       ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history":         loss_history,
    }, path)
    print(f"[Utils] Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str, model: nn.Module, optimizer=None, ema=None, device=None):
    """
    加载 checkpoint 并恢复模型/优化器/EMA 状态。
    
    Returns:
        epoch: 恢复的 epoch 编号
        loss_history: 训练损失历史
    """
    if device is None:
        device = config.DEVICE
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    loss_history = checkpoint.get("loss_history", [])
    
    print(f"[Utils] Checkpoint loaded from {path} (epoch {epoch})")
    return epoch, loss_history
