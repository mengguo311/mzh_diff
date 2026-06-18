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
from dit1d import DiT1D, DiT1D_S, DiT1D_B, DiT1D_L
from scheduler import DDPMScheduler
from utils import (
    set_seed,
    EMA,
    create_run_dir,
    save_config_snapshot,
    save_checkpoint,
    load_checkpoint,
)
import losses


def parse_args():
    parser = argparse.ArgumentParser(description="1D-DDPM Training for Financial Time Series")
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "dit-s", "dit-b", "dit-l"],
                        help="骨干网络: unet / dit-s / dit-b / dit-l (default: unet)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help=f"训练轮数 (default: {config.NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help=f"批大小 (default: {config.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help=f"学习率 (default: {config.LEARNING_RATE})")
    parser.add_argument("--warmup", type=int, default=0,
                        help="学习率 Warmup 步数 (default: 0, DiT 建议 200)")
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
    print(f"  Backbone:   {args.model}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Warmup:     {args.warmup} epochs")
    print(f"  T:          {config.T}")
    print(f"  CLIP_RANGE: {config.CLIP_RANGE}")
    print(f"  min-SNR:    {config.USE_MIN_SNR} (γ={config.MIN_SNR_GAMMA})")
    print(f"  aux-loss:   {config.USE_AUX_LOSS} (acf={config.AUX_ACF_WEIGHT}, rough={config.AUX_ROUGH_WEIGHT}, lag={config.AUX_ACF_MAX_LAG})")
    print(f"  sig-mmd:    {config.USE_SIG_MMD} (w={config.SIG_MMD_WEIGHT}, depth={config.SIG_DEPTH}, Lsub={config.SIG_SUB_LEN}, Nsub={config.SIG_N_SUB})")
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
    model_builders = {
        "unet": lambda: UNet1d(),
        "dit-s": lambda: DiT1D_S(),
        "dit-b": lambda: DiT1D_B(),
        "dit-l": lambda: DiT1D_L(),
    }
    model = model_builders[args.model]().to(device)
    scheduler = DDPMScheduler().to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  {args.model.upper()} parameters: {num_params:,}")

    # ── 3. 优化器 & 调度器 ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup),
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
        epoch_mse = 0.0
        epoch_acf = 0.0
        epoch_rough = 0.0
        epoch_sig = 0.0
        epoch_start = time.time()

        for batch_idx, (x0, c) in enumerate(dataloader):
            # ── 强制上设备 ──
            x0 = x0.to(device)  # (B, 2, seq_len)
            c = c.to(device)    # (B, cond_dim)

            # ── CFG 条件丢弃 (Null Token 替换) ──
            mask = (torch.rand(c.shape[0], 1, device=device) < 0.15)
            c_masked = torch.where(mask, torch.zeros_like(c), c)

            # ── 随机采样时间步 ──
            t = torch.randint(
                0, config.T, (x0.shape[0],), device=device, dtype=torch.long
            )

            # ── 生成随机高斯噪声 ──
            noise = torch.randn_like(x0)  # 在 device 上

            # ── 前向加噪 ──
            xt = scheduler.q_sample(x0, t, noise)

            # ── 预测噪声 ──
            noise_pred = model(xt, t, c_masked)

            # ── 主损失: (min-SNR 加权的) ε-MSE ──
            mse_ps = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=[1, 2])  # (B,)
            if config.USE_MIN_SNR:
                w_snr = losses.min_snr_weight(scheduler, t, config.MIN_SNR_GAMMA)
                loss_mse = (w_snr * mse_ps).mean()
            else:
                loss_mse = mse_ps.mean()

            # ── 辅助损失: stylized-fact (波动聚集 + roughness) + 可微 Sig-MMD，
            #    仅低噪声步 (ᾱ_t 可靠), x̂₀ 反演一次供两者共用 ──
            loss_acf = xt.new_zeros(())
            loss_rough = xt.new_zeros(())
            loss_sig = xt.new_zeros(())
            if config.USE_AUX_LOSS or config.USE_SIG_MMD:
                x0_hat = losses.recover_x0(scheduler, xt, t, noise_pred)
                abar = scheduler.alphas_cumprod.gather(0, t.long())          # (B,) 可靠度
                reliable = abar > config.AUX_ABAR_MIN
                if reliable.any():
                    clip = config.CLIP_RANGE * 3.0
                    x0h = x0_hat[reliable].clamp(-clip, clip)
                    x0r = x0[reliable]
                    if config.USE_AUX_LOSS:
                        acf_l, rough_l = losses.stylized_aux(x0h, x0r, config.AUX_ACF_MAX_LAG)
                        wabar = abar[reliable]
                        loss_acf = (wabar * acf_l).mean() * config.AUX_ACF_WEIGHT
                        loss_rough = (wabar * rough_l).mean() * config.AUX_ROUGH_WEIGHT
                    if config.USE_SIG_MMD and int(reliable.sum()) >= 4:
                        mmd2 = losses.sig_mmd_loss(
                            x0h, x0r,
                            n_sub=config.SIG_N_SUB, l_sub=config.SIG_SUB_LEN,
                        )
                        loss_sig = config.SIG_MMD_WEIGHT * mmd2

            loss = loss_mse + loss_acf + loss_rough + loss_sig

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
            epoch_mse += loss_mse.item()
            epoch_acf += float(loss_acf)
            epoch_rough += float(loss_rough)
            epoch_sig += float(loss_sig)

        # ── 学习率调度 (含 Warmup) ──
        if epoch < args.warmup:
            # 线性 Warmup: lr 从 0 线性增长到 target lr
            warmup_factor = (epoch + 1) / args.warmup
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_factor
        else:
            lr_scheduler.step()

        # ── Epoch 统计 ──
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ── 日志输出 ──
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            nb = len(dataloader)
            print(
                f"  Epoch {epoch:>4d}/{args.epochs}  |  "
                f"Loss: {avg_loss:.6f}  "
                f"(mse {epoch_mse/nb:.5f} acf {epoch_acf/nb:.5f} rgh {epoch_rough/nb:.5f} sig {epoch_sig/nb:.5f})  |  "
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
