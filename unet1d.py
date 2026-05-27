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
    带残差连接和时间嵌入注入的 1D 卷积块。
    
    数据流:
        x (B, C_in, L) + t_emb (B, emb_dim)
        │
        ├─ Conv1d → GroupNorm → GELU
        ├─ + time_projection (broadcast add)
        ├─ Conv1d → GroupNorm → GELU
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

        # 第一组: Conv → Norm → Act
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.GELU()

        # 时间嵌入投影: (B, emb_dim) → (B, out_channels)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        # 第二组: Conv → Norm → Act
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.GELU()

        # 残差连接: 若通道数改变则用 1x1 Conv 对齐
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C_in, L)
            t_emb: (B, emb_dim)
        Returns:
            (B, C_out, L)
        """
        # 第一组卷积
        h = self.act1(self.norm1(self.conv1(x)))

        # 注入时间嵌入: (B, C_out) → (B, C_out, 1) → broadcast add
        t = self.time_proj(t_emb).unsqueeze(-1)  # (B, C_out, 1)
        h = h + t

        # 第二组卷积
        h = self.act2(self.norm2(self.conv2(h)))

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
    1D U-Net 去噪网络。
    
    输入: (B, 2, 128) — 带噪的双通道时间序列
    输出: (B, 2, 128) — 预测的噪声 ε̂
    
    Architecture:
        Encoder:  128 → 64 → 32 → 16
        Bottleneck: 16 (256 channels, 2x ResBlock)
        Decoder:  16 → 32 → 64 → 128  (with skip connections)
    """

    def __init__(
        self,
        in_channels: int = config.CHANNELS,
        channel_dims: list = None,
        time_emb_dim: int = config.TIME_EMB_DIM,
    ):
        super().__init__()
        
        if channel_dims is None:
            channel_dims = config.CHANNEL_DIMS  # [64, 128, 256]

        # ── 时间嵌入 MLP ──
        # t (B,) → SinEmb (B, dim) → MLP → (B, dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ── 初始卷积: 2 → 64 ──
        self.init_conv = nn.Conv1d(
            in_channels, channel_dims[0], kernel_size=7, padding=3
        )

        # ── 编码器 (3级) ──
        # Level 1: 64→64,  128→64
        self.enc_block1 = ResidualBlock1d(channel_dims[0], channel_dims[0], time_emb_dim)
        self.down1 = Downsample1d(channel_dims[0])

        # Level 2: 64→128, 64→32
        self.enc_block2 = ResidualBlock1d(channel_dims[0], channel_dims[1], time_emb_dim)
        self.down2 = Downsample1d(channel_dims[1])

        # Level 3: 128→256, 32→16
        self.enc_block3 = ResidualBlock1d(channel_dims[1], channel_dims[2], time_emb_dim)
        self.down3 = Downsample1d(channel_dims[2])

        # ── 瓶颈 (2个 ResBlock) ──
        self.mid_block1 = ResidualBlock1d(channel_dims[2], channel_dims[2], time_emb_dim)
        self.mid_block2 = ResidualBlock1d(channel_dims[2], channel_dims[2], time_emb_dim)

        # ── 解码器 (3级, 与编码器对称) ──
        # Level 3: upsample 256(16→32), cat skip3(256) → 512, ResBlock → 128
        self.up3 = Upsample1d(channel_dims[2])
        self.dec_block3 = ResidualBlock1d(
            channel_dims[2] * 2, channel_dims[1], time_emb_dim
        )

        # Level 2: upsample 128(32→64), cat skip2(128) → 256, ResBlock → 64
        self.up2 = Upsample1d(channel_dims[1])
        self.dec_block2 = ResidualBlock1d(
            channel_dims[1] * 2, channel_dims[0], time_emb_dim
        )

        # Level 1: upsample 64(64→128), cat skip1(64) → 128, ResBlock → 64
        self.up1 = Upsample1d(channel_dims[0])
        self.dec_block1 = ResidualBlock1d(
            channel_dims[0] * 2, channel_dims[0], time_emb_dim
        )

        # ── 最终输出: 64 → 2 ──
        self.final_conv = nn.Sequential(
            nn.GroupNorm(config.NUM_GROUPS, channel_dims[0]),
            nn.GELU(),
            nn.Conv1d(channel_dims[0], in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: (B, 2, 128) — 带噪声的时间序列
            t: (B,)        — 扩散时间步（整数）
        Returns:
            (B, 2, 128)    — 预测的噪声 ε̂
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)

        # 初始卷积
        x = self.init_conv(x)     # (B, 64, 128)

        # ── 编码器 ──
        h1 = self.enc_block1(x, t_emb)     # (B, 64,  128) ← skip₁
        x = self.down1(h1)                  # (B, 64,   64)

        h2 = self.enc_block2(x, t_emb)     # (B, 128,  64) ← skip₂
        x = self.down2(h2)                  # (B, 128,  32)

        h3 = self.enc_block3(x, t_emb)     # (B, 256,  32) ← skip₃
        x = self.down3(h3)                  # (B, 256,  16)

        # ── 瓶颈 ──
        x = self.mid_block1(x, t_emb)      # (B, 256,  16)
        x = self.mid_block2(x, t_emb)      # (B, 256,  16)

        # ── 解码器 ──
        x = self.up3(x)                    # (B, 256,  32)
        x = torch.cat([x, h3], dim=1)      # (B, 512,  32)
        x = self.dec_block3(x, t_emb)      # (B, 128,  32)

        x = self.up2(x)                    # (B, 128,  64)
        x = torch.cat([x, h2], dim=1)      # (B, 256,  64)
        x = self.dec_block2(x, t_emb)      # (B, 64,   64)

        x = self.up1(x)                    # (B, 64,  128)
        x = torch.cat([x, h1], dim=1)      # (B, 128, 128)
        x = self.dec_block1(x, t_emb)      # (B, 64,  128)

        # ── 输出 ──
        x = self.final_conv(x)             # (B, 2,   128)

        return x
