"""
unet1d.py — 1D U-Net 去噪骨干网络
阶段二：专为时间序列设计的 1D U-Net，禁止 Conv2d。

架构:
    输入: (B, 2, 128)
    编码器: 128 → 64 → 32 → 16  (3级下采样)
    瓶颈:   16                    (2个 ResBlock)
    解码器: 16 → 32 → 64 → 128  (3级上采样 + skip connections)
    输出: (B, 2, 128)

通道数路径: 2 → 64 → [64, 128, 256] → 256(瓶颈) → [128, 64, 64] → 2
"""

import math
import torch
import torch.nn as nn

import config


# ──────────────────────────────────────────────
# Sinusoidal Positional Embedding
# ──────────────────────────────────────────────

class SinusoidalPositionalEmbedding(nn.Module):
    """
    将标量时间步 t 编码为高维正弦/余弦嵌入。
    与 Transformer 位置编码相同的数学公式。
    
    输入:  t (B,)  — 整数时间步
    输出:  emb (B, dim) — 高维嵌入向量
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        
        # 频率序列: exp(-ln(10000) * i / (d/2))
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        
        # 外积: (B, 1) × (1, d/2) → (B, d/2)
        emb = t[:, None].float() * emb[None, :]
        
        # 拼接 sin 和 cos: (B, d)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


# ──────────────────────────────────────────────
# Residual Block for 1D Convolutions
# ──────────────────────────────────────────────

class ResidualBlock1d(nn.Module):
    """
    带残差连接和条件引导 (adaGN / adaLN) 嵌入注入的 1D 卷积块。
    
    数据流:
        x (B, C_in, L) + emb (t_emb + c_emb) (B, emb_dim)
        │
        ├─ Conv1d → GroupNorm (affine=False) → Scale/Shift (predicted from emb) → GELU
        ├─ Conv1d → GroupNorm (affine=False) → Scale/Shift (predicted from emb) → GELU
        │
        └─ skip: Conv1d(1x1) if C_in ≠ C_out else Identity
        
        output = block(x) + skip(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = config.TIME_EMB_DIM,
        num_groups: int = config.NUM_GROUPS,
    ):
        super().__init__()

        # 第一组: Conv → Norm (非仿射，用于手动注入自适应 Scale & Shift) → Act
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels, affine=False)
        self.act1 = nn.GELU()
        
        # 正则化：Dropout 层防过拟合
        self.dropout = nn.Dropout(0.1)

        # 自适应 Group Norm (adaGN) 投影: (B, emb_dim) -> (B, out_channels * 4)
        # 一次性预测出两个 Norm 层的 scale1, shift1, scale2, shift2
        self.adaln_proj = nn.Linear(time_emb_dim, out_channels * 4)

        # 第二组: Conv → Norm → Act
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels, affine=False)
        self.act2 = nn.GELU()

        # 残差连接: 若通道数改变则用 1x1 Conv 对齐
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C_in, L)
            t_emb: (B, emb_dim)
            c_emb: (B, emb_dim)
        Returns:
            (B, C_out, L)
        """
        # 条件与时间融合嵌入
        emb = t_emb + c_emb  # (B, emb_dim)
        
        # 预测自适应归一化参数
        proj = self.adaln_proj(emb)  # (B, out_channels * 4)
        scale1, shift1, scale2, shift2 = torch.chunk(proj, chunks=4, dim=-1)
        
        # 第一层卷积与自适应 GN1
        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + scale1.unsqueeze(-1)) + shift1.unsqueeze(-1)
        h = self.act1(h)
        h = self.dropout(h)

        # 第二层卷积与自适应 GN2
        h = self.conv2(h)
        h = self.norm2(h)
        h = h * (1.0 + scale2.unsqueeze(-1)) + shift2.unsqueeze(-1)
        h = self.act2(h)

        # 残差连接
        return h + self.skip(x)


# ──────────────────────────────────────────────
# Downsample / Upsample
# ──────────────────────────────────────────────

class Downsample1d(nn.Module):
    """1D 下采样: 序列长度严格减半。使用 stride=2 的卷积。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D 上采样: 序列长度严格翻倍。使用 stride=2 的转置卷积。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ──────────────────────────────────────────────
# 1D U-Net
# ──────────────────────────────────────────────

class UNet1d(nn.Module):
    """
    1D U-Net 去噪网络，支持 Classifier-Free Guidance (CFG)，并且能够动态适应任意层数的 channel_dims。
    
    输入: (B, 2, L) — 带噪的双通道时间序列
    条件: (B, cond_dim) — 初始状态向量
    输出: (B, 2, L) — 预测的噪声 ε̂
    """

    def __init__(
        self,
        in_channels: int = config.CHANNELS,
        channel_dims: list = None,
        time_emb_dim: int = config.TIME_EMB_DIM,
        cond_dim: int = config.CHANNELS,
    ):
        super().__init__()
        
        if channel_dims is None:
            channel_dims = config.CHANNEL_DIMS  # [64, 128, 256, 512, 1024]

        num_resolutions = len(channel_dims)

        # ── 时间嵌入 MLP ──
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ── 条件嵌入 MLP ──
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── 初始卷积: 2 → channel_dims[0] ──
        self.init_conv = nn.Conv1d(
            in_channels, channel_dims[0], kernel_size=7, padding=3
        )

        # ── 动态构建编码器 (nn.ModuleList) ──
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(num_resolutions):
            in_c = channel_dims[0] if i == 0 else channel_dims[i - 1]
            out_c = channel_dims[i]
            self.enc_blocks.append(ResidualBlock1d(in_c, out_c, time_emb_dim))
            self.downs.append(Downsample1d(out_c))

        # ── 瓶颈 (2个 ResBlock) ──
        mid_c = channel_dims[-1]
        self.mid_block1 = ResidualBlock1d(mid_c, mid_c, time_emb_dim)
        self.mid_block2 = ResidualBlock1d(mid_c, mid_c, time_emb_dim)

        # ── 动态构建解码器 (nn.ModuleList, 与编码器对称) ──
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(num_resolutions)):
            self.ups.append(Upsample1d(channel_dims[i]))
            in_c = channel_dims[i] * 2
            out_c = channel_dims[0] if i == 0 else channel_dims[i - 1]
            self.dec_blocks.append(ResidualBlock1d(in_c, out_c, time_emb_dim))

        # ── 最终输出 ──
        self.final_conv = nn.Sequential(
            nn.GroupNorm(config.NUM_GROUPS, channel_dims[0]),
            nn.GELU(),
            nn.Conv1d(channel_dims[0], in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: (B, 2, L) — 带噪声的时间序列
            t: (B,)      — 扩散时间步
            c: (B, cond_dim) — 初始状态条件
        Returns:
            (B, 2, L)    — 预测的噪声 ε̂
        """
        # 时间嵌入与条件嵌入
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)
        c_emb = self.cond_mlp(c)  # (B, time_emb_dim)

        # 初始卷积
        x = self.init_conv(x)

        # ── 编码器 ──
        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x, t_emb, c_emb)
            skips.append(x)
            x = down(x)

        # ── 瓶颈 ──
        x = self.mid_block1(x, t_emb, c_emb)
        x = self.mid_block2(x, t_emb, c_emb)

        # ── 解码器 ──
        for up, block in zip(self.ups, self.dec_blocks):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb, c_emb)

        # ── 输出 ──
        x = self.final_conv(x)

        return x
