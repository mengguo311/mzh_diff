"""
train.py — DDPM 训练流水线
阶段四：完整的训练入口，包含 EMA、梯度裁剪、Checkpoint 管理。

用法:
    conda run -n ts_diffusion python ~/src/train.py
    conda run -n ts_diffusion python ~/src/train.py --epochs 100 --run_name my_exp
    conda run -n ts_diffusion python ~/src/train.py --resume ~/src/logs/run_xxx/checkpoint_epoch_0100.pt

所有超参数从 config.py 导入，命令行参数可覆盖部分配置。
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# 确保 ~/src 在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from dataset import TimeSeriesDataset
from unet1d import UNet1d
from scheduler import DDPMScheduler
from utils import (
    set_seed,
    EMA,
    create_run_dir,
    save_config_snapshot,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="1D-DDPM Training for Financial Time Series")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help=f"训练轮数 (default: {config.NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help=f"批大小 (default: {config.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help=f"学习率 (default: {config.LEARNING_RATE})")
    parser.add_argument("--run_name", type=str, default=None,
                        help="运行名称（默认自动生成时间戳）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练（提供 .pt 路径）")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"随机种子 (default: {config.SEED})")
    return parser.parse_args()


def train():
    args = parse_args()
    device = config.DEVICE

    # ── 可复现性 ──
    set_seed(args.seed)

    print("=" * 60)
    print("  1D-DDPM Training Pipeline")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  T:          {config.T}")
    print("=" * 60)

    # ── 1. 数据管道 ──
    print("\n[Phase 1] Loading data...")
    dataset = TimeSeriesDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    scaler = dataset.get_scaler()

    print(f"  Samples:      {len(dataset)}")
    print(f"  Batches/epoch: {len(dataloader)}")

    # ── 2. 模型 ──
    print("\n[Phase 2] Building model...")
    model = UNet1d().to(device)
    scheduler = DDPMScheduler().to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  U-Net parameters: {num_params:,}")

    # ── 3. 优化器 & 调度器 ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # ── 4. EMA ──
    ema = EMA(model, decay=config.EMA_DECAY)

    # ── 5. 运行目录 ──
    run_dir = create_run_dir(run_name=args.run_name)
    save_config_snapshot(run_dir)

    # 保存 scaler 参数
    scaler_path = os.path.join(run_dir, "scaler.pt")
    scaler.save(scaler_path)

    # ── 6. 恢复训练 ──
    start_epoch = 0
    loss_history = []

    if args.resume is not None:
        print(f"\n[Resume] Loading checkpoint from {args.resume}")
        start_epoch, loss_history = load_checkpoint(
            args.resume, model, optimizer, ema, device
        )
        start_epoch += 1  # 从下一个 epoch 开始
        print(f"  Resuming from epoch {start_epoch}")

    # ── 7. 训练循环 ──
    print(f"\n[Phase 3] Training starts (epoch {start_epoch} → {args.epochs - 1})...")
    print("-" * 60)

    model.train()
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, x0 in enumerate(dataloader):
            # ── 强制上设备 ──
            x0 = x0.to(device)  # (B, 2, 128)

            # ── 随机采样时间步 ──
            t = torch.randint(
                0, config.T, (x0.shape[0],), device=device, dtype=torch.long
            )

            # ── 生成随机高斯噪声 ──
            noise = torch.randn_like(x0)  # 在 device 上

            # ── 前向加噪 ──
            xt = scheduler.q_sample(x0, t, noise)

            # ── 预测噪声 ──
            noise_pred = model(xt, t)

            # ── MSE Loss ──
            loss = F.mse_loss(noise_pred, noise)

            # ── 反向传播 ──
            optimizer.zero_grad()
            loss.backward()

            # ── 梯度裁剪 ──
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.GRAD_CLIP
            )

            # ── 更新参数 ──
            optimizer.step()

            # ── 更新 EMA ──
            ema.update(model)

            epoch_loss += loss.item()

        # ── 学习率衰减 ──
        lr_scheduler.step()

        # ── Epoch 统计 ──
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ── 日志输出 ──
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"  Epoch {epoch:>4d}/{args.epochs}  |  "
                f"Loss: {avg_loss:.6f}  |  "
                f"LR: {current_lr:.2e}  |  "
                f"Time: {epoch_time:.2f}s"
            )

        # ── Checkpoint ──
        if (epoch + 1) % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(run_dir, epoch, model, optimizer, ema, loss_history)

    # ── 8. 保存最终 checkpoint ──
    save_checkpoint(
        run_dir, args.epochs - 1, model, optimizer, ema, loss_history, is_final=True
    )

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"\n[Done] Training complete!")
    print(f"  Total time:   {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final loss:   {loss_history[-1]:.6f}")
    print(f"  Best loss:    {min(loss_history):.6f} (epoch {loss_history.index(min(loss_history))})")
    print(f"  Run dir:      {run_dir}")
    print(f"  Scaler:       {scaler_path}")

    # ── 9. 保存 loss 历史 ──
    loss_path = os.path.join(run_dir, "loss_history.csv")
    with open(loss_path, "w") as f:
        f.write("epoch,loss\n")
        for i, loss_val in enumerate(loss_history):
            f.write(f"{i},{loss_val:.8f}\n")
    print(f"  Loss history: {loss_path}")


if __name__ == "__main__":
    train()
