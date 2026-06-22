"""
generate.py — DDPM 生成流水线
阶段四：加载训练好的模型，从纯噪声生成金融时间序列。

用法:
    conda run -n ts_diffusion python ~/src/generate.py \
        --checkpoint ~/src/logs/run_xxx/checkpoint_final.pt \
        --scaler ~/src/logs/run_xxx/scaler.pt \
        --num_samples 10000

生成流程:
    1. 加载 EMA 权重 → model.to(device)
    2. 加载 scaler.pt → TimeSeriesScaler
    3. 分批生成 (每批 GEN_BATCH_SIZE 个):
       x_T ~ N(0,I) → 200步逆向去噪 → x_0 → inverse_transform → 真实量级
    4. 保存为 CSV: sp500_0...127, dgs10_0...127 (共256列)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from dataset import TimeSeriesDataset, TimeSeriesScaler
from unet1d import UNet1d
from dit1d import DiT1D_S, DiT1D_B, DiT1D_L
from scheduler import DDPMScheduler
from utils import set_seed, EMA


def parse_args():
    parser = argparse.ArgumentParser(description="1D-DDPM Generation for Financial Time Series")
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "dit-s", "dit-b", "dit-l"],
                        help="骨干网络: unet / dit-s / dit-b / dit-l (default: unet)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练 checkpoint 路径 (.pt)")
    parser.add_argument("--scaler", type=str, required=True,
                        help="Scaler 参数路径 (.pt)")
    parser.add_argument("--num_samples", type=int, default=config.NUM_SIMULATIONS,
                        help=f"生成路径数量 (default: {config.NUM_SIMULATIONS})")
    parser.add_argument("--batch_size", type=int, default=config.GEN_BATCH_SIZE,
                        help=f"生成批大小 (default: {config.GEN_BATCH_SIZE})")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 路径 (default: ~/src/output/generated_paths.csv)")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"随机种子 (default: {config.SEED})")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="使用 EMA 权重生成 (default: True)")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false",
                        help="使用原始模型权重生成")
    parser.add_argument("--num_inference_steps", type=int, default=config.GEN_NUM_STEPS,
                        help=f"DDIM 快速采样步数 (default: {config.GEN_NUM_STEPS})")
    parser.add_argument("--guidance_scale", "-w", type=float, default=config.GEN_GUIDANCE_SCALE,
                        help=f"Classifier-Free Guidance 引导权重 w (default: {config.GEN_GUIDANCE_SCALE})")
    parser.add_argument("--eta", type=float, default=config.GEN_ETA,
                        help=f"DDIM 随机性 eta (0=确定性, 1≈DDPM; 注入纹理/波动, default: {config.GEN_ETA})")
    parser.add_argument("--cond_mode", type=str, default="dataset", choices=["dataset", "zero"],
                        help="条件生成模式 (default: dataset)")
    return parser.parse_args()


def generate():
    args = parse_args()
    device = config.DEVICE

    set_seed(args.seed)

    # ── 输出路径 ──
    if args.output is None:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(config.OUTPUT_DIR, "generated_paths.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        output_path = args.output

    print("=" * 60)
    print("  1D-DDPM Generation Pipeline")
    print("=" * 60)
    print(f"  Device:       {device}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Scaler:       {args.scaler}")
    print(f"  Num samples:  {args.num_samples}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Use EMA:      {args.use_ema}")
    print(f"  Output:       {output_path}")
    print("=" * 60)

    # ── 1. 加载 Scaler ──
    print("\n[Step 1] Loading scaler...")
    scaler = TimeSeriesScaler()
    scaler.load(args.scaler)

    # ── 2. 加载和解析运行配置 (如果存在) ──
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    config_json_path = os.path.join(checkpoint_dir, "config.json")
    
    # 默认值使用全局 config 中的值
    seq_len = config.SEQ_LEN
    channels = config.CHANNELS
    cond_dim = config.COND_DIM
    channel_dims = config.CHANNEL_DIMS
    time_emb_dim = config.TIME_EMB_DIM
    T = config.T
    beta_start = config.BETA_START
    beta_end = config.BETA_END

    if os.path.exists(config_json_path):
        print(f"  Found run configuration: {config_json_path}")
        try:
            with open(config_json_path, "r") as f:
                run_config = json.load(f)
            seq_len = run_config.get("seq_len", seq_len)
            channels = run_config.get("channels", channels)
            cond_dim = run_config.get("cond_dim", cond_dim)
            channel_dims = run_config.get("channel_dims", channel_dims)
            time_emb_dim = run_config.get("time_emb_dim", time_emb_dim)
            T = run_config.get("T", T)
            beta_start = run_config.get("beta_start", beta_start)
            beta_end = run_config.get("beta_end", beta_end)
            print(f"  Loaded model config: channel_dims={channel_dims}, time_emb_dim={time_emb_dim}, T={T}")
        except Exception as e:
            print(f"  [Warning] Failed to load config.json, using defaults. Error: {e}")

    # ── 3. 加载模型 ──
    print("\n[Step 2] Loading model...")
    if args.model == "unet":
        model = UNet1d(
            in_channels=channels,
            channel_dims=channel_dims,
            time_emb_dim=time_emb_dim
        ).to(device)
    else:
        model_builders = {
            "dit-s": DiT1D_S,
            "dit-b": DiT1D_B,
            "dit-l": DiT1D_L,
        }
        model = model_builders[args.model](
            in_channels=channels,
            seq_len=seq_len,
            cond_dim=cond_dim,
        ).to(device)
    
    scheduler = DDPMScheduler(
        num_timesteps=T,
        beta_start=beta_start,
        beta_end=beta_end
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.use_ema and "ema_state_dict" in checkpoint:
        # 使用 EMA 权重
        ema_state = checkpoint["ema_state_dict"]
        # EMA state dict 的 key 与 model.named_parameters() 的 key 一致
        model_state = model.state_dict()
        for name in ema_state:
            if name in model_state:
                model_state[name] = ema_state[name]
        model.load_state_dict(model_state)
        print("  Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("  Loaded model weights (no EMA)")

    epoch = checkpoint.get("epoch", "unknown")
    print(f"  Trained epoch: {epoch}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  {args.model.upper()} parameters: {num_params:,}")

    # ── 3.5 准备初始条件 ──
    if args.cond_mode == "dataset":
        print("\n[Step 2.5] Loading dataset for conditional generation initial states...")
        dataset = TimeSeriesDataset(scaler=scaler)
        print(f"  Dataset loaded. Number of available conditions: {len(dataset)}")
        rng = np.random.default_rng(args.seed)
        sampled_indices = rng.choice(len(dataset), size=args.num_samples, replace=True)
        sampled_conditions = [dataset[idx][1] for idx in sampled_indices]
        sampled_conditions = torch.stack(sampled_conditions).to(device)  # (num_samples, cond_dim)
    else:
        print("\n[Step 2.5] Using zero vector as unconditional/null conditions...")
        sampled_conditions = torch.zeros(args.num_samples, cond_dim, device=device)

    # ── 4. 分批生成 ──
    print(f"\n[Step 3] Generating {args.num_samples} paths...")
    model.eval()

    all_samples = []
    num_remaining = args.num_samples
    batch_idx = 0
    gen_start = time.time()

    while num_remaining > 0:
        batch = min(args.batch_size, num_remaining)
        batch_start_idx = args.num_samples - num_remaining
        batch_idx += 1

        print(f"\n  --- Batch {batch_idx} ({batch} samples) ---")

        # 初始化纯高斯噪声
        x_T = torch.randn(batch, channels, seq_len, device=device)
        
        # 提取当前 batch 的条件向量
        c_batch = sampled_conditions[batch_start_idx : batch_start_idx + batch]

        # 确定性 DDIM 逆向去噪
        with torch.no_grad():
            x_0 = scheduler.ddim_sample_loop(
                model=model,
                c=c_batch,
                x_T=x_T,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
                verbose=True
            )

        # 还原真实金融量级
        x_real = scaler.inverse_transform(x_0)  # (batch, 2, seq_len)

        all_samples.append(x_real.cpu())
        num_remaining -= batch

        print(f"  Generated: {args.num_samples - num_remaining}/{args.num_samples}")

    gen_time = time.time() - gen_start

    # ── 5. 拼接并保存 CSV ──
    print(f"\n[Step 4] Saving to CSV...")
    all_samples = torch.cat(all_samples, dim=0)  # (N, 2, seq_len)

    # 拆分双通道
    sp500_data = all_samples[:, 0, :].numpy()  # (N, seq_len)
    dgs10_data = all_samples[:, 1, :].numpy()  # (N, seq_len)

    # 拼接为宽表: (N, 2 * seq_len)
    combined = np.concatenate([sp500_data, dgs10_data], axis=1)

    # 列名: sp500_0...sp500_{seq_len-1}, dgs10_0...dgs10_{seq_len-1}
    columns = (
        [f"sp500_{i}" for i in range(seq_len)]
        + [f"dgs10_{i}" for i in range(seq_len)]
    )

    df = pd.DataFrame(combined, columns=columns)
    df.to_csv(output_path, index=False)

    # ── 6. 生成质量报告 ──
    print("\n" + "=" * 60)
    print("  Generation Complete!")
    print("=" * 60)
    print(f"  Output:         {output_path}")
    print(f"  Shape:          {df.shape}")
    print(f"  Total time:     {gen_time:.1f}s")
    print(f"  Time per sample: {gen_time/args.num_samples*1000:.2f}ms")
    print()
    print("  ── SP500 日收益率统计 ──")
    print(f"    Mean:  {sp500_data.mean():.6f}")
    print(f"    Std:   {sp500_data.std():.6f}")
    print(f"    Min:   {sp500_data.min():.6f}")
    print(f"    Max:   {sp500_data.max():.6f}")
    print()
    print("  ── DGS10 日差分统计 ──")
    print(f"    Mean:  {dgs10_data.mean():.6f}")
    print(f"    Std:   {dgs10_data.std():.6f}")
    print(f"    Min:   {dgs10_data.min():.6f}")
    print(f"    Max:   {dgs10_data.max():.6f}")


if __name__ == "__main__":
    generate()
