#!/usr/bin/env python3
"""
generate_autoregressive.py — v13 C1 自回归采样器: 用 context-conditioned 模型把短窗(512)
链式拼接成长样本(默认 2048), 重建缩窗丢失的长程结构。

机制: 逐窗生成, 每个新窗以【前一窗的富统计上下文向量】(与训练 dataset._ctx_stats 完全同口径) 为条件,
故跨窗 regime/波动状态可延续。窗 0 用采样的真实上下文(或 null)起步。朴素首尾拼接(接缝处由 context
延续保证连贯; overlap-averaging 平滑为后续可选 refinement, 严禁真实数据填缝)。

仅适用于 cond_dim>2 的 C1 模型 (USE_CONTEXT_COND 训练出的)。
用法:
  conda run -n ts_diffusion python generate_autoregressive.py --model dit-s \
    --checkpoint logs/<c1_run>/checkpoint_final.pt --scaler logs/<c1_run>/scaler.pt \
    --num_samples 5120 --k 4 --seed_ctx real --output output/<c1>_ar2048.csv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import config
from dataset import TimeSeriesScaler, TimeSeriesDataset
from dit1d import DiT1D_S, DiT1D_B, DiT1D_L
from scheduler import DDPMScheduler


def ctx_features(x: torch.Tensor) -> torch.Tensor:
    """(B,2,L) 标准化窗 → (B, 2*8) 富上下文向量。**逐项与 dataset._ctx_stats 同口径同顺序。**"""
    feats = []
    for ch in range(2):
        s = x[:, ch, :]                                   # (B,L)
        a = s.abs(); am = a - a.mean(1, keepdim=True)
        var = (am * am).mean(1).clamp(min=1e-8)
        acf1 = (am[:, 1:] * am[:, :-1]).mean(1) / var
        sc = s - s.mean(1, keepdim=True)
        sd = sc.std(1).clamp(min=1e-8)
        z = sc / sd.unsqueeze(1)
        d2 = s[:, 2:] - 2.0 * s[:, 1:-1] + s[:, :-2]
        feats += [s.std(1), a.mean(1), s.mean(1), acf1,
                  (z ** 3).mean(1), (z ** 4).mean(1) - 3.0, s[:, -1], (d2 * d2).mean(1)]
    return torch.stack(feats, dim=1)                      # (B,16)


def autoregressive_generate(model, scheduler, scaler, num_samples, seq_len, k,
                            steps, w, eta, device, seed_ctx="null", seed_bank=None,
                            batch_size=64, force_null=False, x0_clamp=None, verbose=False):
    """链式生成 num_samples 条 (2, k*seq_len) 长样本 (标准化空间逆变换后真实量级)。
    force_null=True: 所有窗条件恒为零 (消融控制臂, 验证 context 是否真被用上)。"""
    cond_dim = model.c_embedder.mlp[0].in_features
    out = []
    done = 0
    while done < num_samples:
        B = min(batch_size, num_samples - done)
        # 窗 0 的条件
        if force_null or seed_ctx == "null" or seed_bank is None:
            c = torch.zeros(B, cond_dim, device=device)
        else:
            sel = torch.randint(0, seed_bank.shape[0], (B,))
            c = seed_bank[sel].to(device)
        wins = []
        for j in range(k):
            x_T = torch.randn(B, 2, seq_len, device=device)
            with torch.no_grad():
                x0 = scheduler.ddim_sample_loop(model=model, c=c, x_T=x_T,
                                                num_inference_steps=steps,
                                                guidance_scale=w, eta=eta,
                                                x0_clamp=x0_clamp, verbose=False)
            wins.append(x0)
            c = torch.zeros(B, cond_dim, device=device) if force_null else ctx_features(x0).detach()
        full = torch.cat(wins, dim=2)                     # (B,2,k*seq_len)
        out.append(scaler.inverse_transform(full).cpu())
        done += B
        if verbose:
            print(f"  自回归 {done}/{num_samples}")
    return torch.cat(out, dim=0)                           # (N,2,k*seq_len)


def _load_model(model_name, checkpoint, device):
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    seq_len, cond_dim = config.SEQ_LEN, config.COND_DIM
    cj = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cj):
        rc = json.load(open(cj))
        seq_len = rc.get("seq_len", seq_len); cond_dim = rc.get("cond_dim", cond_dim)
    builder = {"dit-s": DiT1D_S, "dit-b": DiT1D_B, "dit-l": DiT1D_L}[model_name]
    model = builder(in_channels=config.CHANNELS, seq_len=seq_len, cond_dim=cond_dim).to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ck.get("ema_state_dict") or ck.get("model_state_dict") or ck
    ms = model.state_dict()
    for n in state:
        if n in ms:
            ms[n] = state[n]
    model.load_state_dict(ms); model.eval()
    return model, seq_len, cond_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dit-s", choices=["dit-s", "dit-b", "dit-l"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--num_samples", type=int, default=config.NUM_SIMULATIONS)
    ap.add_argument("--k", type=int, default=4, help="链式窗数 (512*4=2048)")
    ap.add_argument("--num_inference_steps", type=int, default=200)
    ap.add_argument("--guidance_scale", "-w", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--seed_ctx", default="real", choices=["real", "null"])
    ap.add_argument("--force_null", action="store_true", help="所有窗条件恒零 (C1 消融控制臂)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--x0_clamp", type=float, default=None,
                    help="⑦ x0 钳位(标准化 σ, 如 20 治自回归发散); 不传则用 config.X0_CLAMP_SIGMA")
    args = ap.parse_args()

    device = config.DEVICE
    scaler = TimeSeriesScaler(); scaler.load(args.scaler)
    model, seq_len, cond_dim = _load_model(args.model, args.checkpoint, device)
    print(f"[ar] model {args.model} seq_len={seq_len} cond_dim={cond_dim}; 链式 k={args.k} → {args.k*seq_len}")
    if cond_dim <= 2:
        raise SystemExit("[ar] cond_dim<=2: 该 checkpoint 非 C1 富条件模型, 自回归无意义。")

    seed_bank = None
    if args.seed_ctx == "real" and not args.force_null:
        _saved = config.USE_CONTEXT_COND
        config.USE_CONTEXT_COND = True                    # 让 dataset 出富条件做种子
        ds = TimeSeriesDataset(scaler=scaler)
        valid = [i for i, s in enumerate(ds.indices) if s >= seq_len]
        seed_bank = torch.stack([ds[i][1] for i in valid])
        config.USE_CONTEXT_COND = _saved
        print(f"[ar] 真实种子上下文库 {tuple(seed_bank.shape)}")

    scheduler = DDPMScheduler().to(device)
    t0 = time.time()
    x0c = args.x0_clamp if args.x0_clamp is not None else getattr(config, "X0_CLAMP_SIGMA", None)
    if x0c is not None:
        print(f"[ar] ⑦ x0 钳位启用: ±{x0c}σ (治自回归罕见单窗发散)")
    samples = autoregressive_generate(model, scheduler, scaler, args.num_samples, seq_len,
                                      args.k, args.num_inference_steps, args.guidance_scale,
                                      args.eta, device, args.seed_ctx, seed_bank,
                                      args.batch_size, force_null=args.force_null,
                                      x0_clamp=x0c, verbose=True)
    L = samples.shape[2]
    sp = samples[:, 0, :].numpy(); dg = samples[:, 1, :].numpy()
    _q = getattr(config, "DGS10_QUANTIZE", None)        # line1: DGS10 量化吸附到 0.01 网格
    if _q:
        dg = np.round(dg / _q) * _q
        print(f"[ar] [DGS10量化] 吸附到 {_q} 网格")
    cols = [f"sp500_{i}" for i in range(L)] + [f"dgs10_{i}" for i in range(L)]
    pd.DataFrame(np.concatenate([sp, dg], axis=1), columns=cols).to_csv(args.output, index=False)
    print(f"[ar] {samples.shape[0]} 条 x{L} 写入 {args.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
