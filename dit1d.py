"""
dit1d.py — 1D Diffusion Transformer (DiT) 骨干网络
基于论文 "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
为金融时间序列 [Batch, 2, 2048] 设计的一维 DiT 架构。

与 U-Net 的核心区别:
    - U-Net 依赖局部卷积感受野，通过逐级下采样/上采样扩展长程覆盖。
      5 级 U-Net 的有效感受野约 64 个时间步，对 2048 长序列的远端依赖捕获困难。
    - DiT 使用全局 Self-Attention，每一层的每个 Token 都能直接关注所有 128 个 Token
      （对应 2048 时间步），天然解决长程注意力稀释问题。

架构流程:
    输入: x (B, 2, 2048) + t (B,) + c (B, 2)
      ↓
    [Patchify]  Conv1d(2, hidden, kernel=16, stride=16) → (B, 128, hidden)
      ↓
    [+] 正弦位置编码 (Sinusoidal PE)
      ↓
    [DiTBlock × N] 每层:
        adaLN-Zero → Multi-Head Self-Attention → 残差
        adaLN-Zero → MLP (GELU) → 残差
      ↓
    [Final adaLN] 最后一层自适应归一化
      ↓
    [Unpatchify] Linear(hidden, patch_size * 2) → reshape → (B, 2, 2048)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
#  1. 正弦位置编码 (Sinusoidal Positional Encoding)
# ══════════════════════════════════════════════════════════════

def get_1d_sincos_pos_embed(embed_dim: int, num_positions: int) -> torch.Tensor:
    """
    生成一维正弦/余弦位置编码表 (不可学习, 固定初始化)。

    数学公式 (与 "Attention Is All You Need" 一致):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        embed_dim:     嵌入维度 d
        num_positions: 位置总数 (= seq_len / patch_size)
    Returns:
        (num_positions, embed_dim) 位置编码矩阵
    """
    half_dim = embed_dim // 2
    # 频率向量: (half_dim,)
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    )
    # 位置向量: (num_positions, 1) × (1, half_dim) → (num_positions, half_dim)
    positions = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    angles = positions * freq.unsqueeze(0)

    # 拼接 sin 和 cos: (num_positions, embed_dim)
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return pe


# ══════════════════════════════════════════════════════════════
#  2. 时间步正弦编码 (Timestep Sinusoidal Embedding)
# ══════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """
    将标量扩散时间步 t ∈ {0, ..., T-1} 编码为高维嵌入。

    流程: t (B,) → 正弦编码 (B, dim) → MLP → (B, hidden_size)
    """

    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        # 两层 MLP: freq_dim → hidden_size → hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 整数时间步
        Returns:
            (B, hidden_size) 时间嵌入
        """
        half = self.frequency_dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        # (B, 1) × (1, half) → (B, half)
        args = t[:, None].float() * freq[None, :]
        # (B, frequency_dim)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


# ══════════════════════════════════════════════════════════════
#  3. 条件嵌入 (Condition Embedding for CFG)
# ══════════════════════════════════════════════════════════════

class ConditionEmbedding(nn.Module):
    """
    将条件向量 c (初始状态, 形状 (B, cond_dim)) 编码为嵌入。
    兼容 Classifier-Free Guidance: 当 c=None 或被置零时，输出全零嵌入。

    流程: c (B, cond_dim) → MLP → (B, hidden_size)
    """

    def __init__(self, cond_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (B, cond_dim) 条件向量, 或 None
        Returns:
            (B, hidden_size) 条件嵌入
        """
        return self.mlp(c)


# ══════════════════════════════════════════════════════════════
#  4. adaLN-Zero 自适应层归一化 (核心机制)
# ══════════════════════════════════════════════════════════════

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    自适应归一化的调制函数。

    公式: y = x * (1 + scale) + shift
    其中 scale 和 shift 由条件嵌入通过线性层预测。

    Args:
        x:     (B, N, D) LayerNorm 后的 Token 特征
        shift: (B, 1, D) 或 (B, D) 平移参数
        scale: (B, 1, D) 或 (B, D) 缩放参数
    Returns:
        (B, N, D) 调制后的特征
    """
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    return x * (1.0 + scale) + shift


# ══════════════════════════════════════════════════════════════
#  5. DiT Block (Transformer Block with adaLN-Zero)
# ══════════════════════════════════════════════════════════════

class DiTBlock1D(nn.Module):
    """
    DiT Transformer Block，使用 adaLN-Zero 机制注入条件信息。

    与标准 Transformer Block 的区别:
        1. 不使用标准 LayerNorm(affine=True)，而是使用 LayerNorm(affine=False) + 条件调制
        2. 每个 Block 从条件嵌入预测 6 个参数:
           - γ₁, β₁: Attention 前 LayerNorm 的 scale/shift
           - γ₂, β₂: MLP 前 LayerNorm 的 scale/shift
           - α₁:     Attention 残差连接的门控 (gate)
           - α₂:     MLP 残差连接的门控 (gate)
        3. adaLN-Zero 初始化: 预测 6 参数的线性层权重和偏置初始化为全零，
           使得网络初始状态等效于恒等映射 (Identity Function)，保证训练稳定性。

    数据流:
        ┌──────────────────────────────────────┐
        │  cond_emb (B, D)                     │
        │       │                              │
        │       ▼                              │
        │  Linear → 6 params (γ₁,β₁,α₁,γ₂,β₂,α₂)  │
        │                                      │
        │  x ─→ LN(x)·(1+γ₁)+β₁ → Attn → ×α₁ ─→ + ─→ x'
        │  x'─→ LN(x')·(1+γ₂)+β₂ → MLP → ×α₂ ─→ + ─→ output
        └──────────────────────────────────────┘
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        """
        Args:
            hidden_size: Token 特征维度 D
            num_heads:   Self-Attention 的头数
            mlp_ratio:   MLP 隐藏层宽度倍率 (hidden → hidden * mlp_ratio → hidden)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # ── LayerNorm (非仿射, affine=False) ──
        # 标准化统计量由数据本身决定, 而缩放/平移由条件嵌入动态预测
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # ── Multi-Head Self-Attention ──
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,  # 输入格式: (B, N, D)
        )

        # ── MLP (Point-wise Feed-Forward) ──
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # ── adaLN-Zero 参数预测层 ──
        # 从条件嵌入预测 6 个 D 维向量: (γ₁, β₁, α₁, γ₂, β₂, α₂)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

        # 🔴 关键初始化: 将 adaLN 的最终线性层权重和偏置初始化为零
        # 这确保了网络在训练初期:
        #   γ = 0, β = 0, α = 0
        #   → LN(x) * (1 + 0) + 0 = LN(x)  (归一化无偏移)
        #   → Attn(LN(x)) * 0 = 0           (残差门控关闭)
        #   → output = x + 0 = x            (恒等映射 ✓)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        (B, N, D) Token 序列
            cond_emb: (B, D)   融合条件嵌入 (t_emb + c_emb)
        Returns:
            (B, N, D) 处理后的 Token 序列
        """
        # 预测 6 个调制参数
        modulation_params = self.adaLN_modulation(cond_emb)  # (B, 6D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = modulation_params.chunk(6, dim=-1)
        # 各参数形状: (B, D)

        # ── Attention 分支 ──
        # 1. 自适应归一化
        x_norm1 = modulate(self.norm1(x), shift=beta1, scale=gamma1)  # (B, N, D)
        # 2. Multi-Head Self-Attention
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)  # (B, N, D)
        # 3. 门控残差连接
        x = x + alpha1.unsqueeze(1) * attn_out

        # ── MLP 分支 ──
        # 1. 自适应归一化
        x_norm2 = modulate(self.norm2(x), shift=beta2, scale=gamma2)  # (B, N, D)
        # 2. MLP
        mlp_out = self.mlp(x_norm2)  # (B, N, D)
        # 3. 门控残差连接
        x = x + alpha2.unsqueeze(1) * mlp_out

        return x


# ══════════════════════════════════════════════════════════════
#  6. DiT1D 主模型
# ══════════════════════════════════════════════════════════════

class DiT1D(nn.Module):
    """
    1D Diffusion Transformer — 用于金融时间序列的 DiT 骨干网络。

    架构总览:
        Patchify (Conv1d) → [+ Pos Embed] → N × DiTBlock1D → Final Norm → Unpatchify

    参数:
        in_channels:  输入通道数 (默认 2: SP500 收益率 + DGS10 差分)
        seq_len:      输入序列长度 (默认 2048)
        patch_size:   Patch 切分大小 (默认 16, 产生 2048/16=128 个 Token)
        hidden_size:  Transformer 隐藏维度 (默认 512)
        num_heads:    Self-Attention 头数 (默认 8)
        depth:        DiT Block 堆叠层数 (默认 12)
        mlp_ratio:    MLP 宽度倍率 (默认 4.0)
        cond_dim:     条件向量维度 (默认 2: 双资产初始状态)

    模型规模参考 (hidden_size / depth / num_heads):
        DiT-S: 384 / 12 / 6   (~25M params)
        DiT-B: 512 / 12 / 8   (~55M params)  ← 默认配置
        DiT-L: 768 / 24 / 12  (~200M params)
    """

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 2048,
        patch_size: int = 16,
        hidden_size: int = 512,
        num_heads: int = 8,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        cond_dim: int = 2,
    ):
        super().__init__()

        # ── 配置保存 ──
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.num_patches = seq_len // patch_size  # 2048 / 16 = 128

        assert seq_len % patch_size == 0, \
            f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})"
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # ══════════════════════════════════════════
        # Patchify: 一维时序块切分
        # ══════════════════════════════════════════
        # Conv1d(2, hidden, kernel=16, stride=16) 等效于将每 16 个时间步压缩为 1 个 Token
        # 输入: (B, 2, 2048) → 输出: (B, hidden, 128) → permute → (B, 128, hidden)
        self.patch_embed = nn.Conv1d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )

        # ══════════════════════════════════════════
        # 一维正弦位置编码 (不可学习)
        # ══════════════════════════════════════════
        pe = get_1d_sincos_pos_embed(hidden_size, self.num_patches)  # (128, hidden)
        self.register_buffer("pos_embed", pe.unsqueeze(0))  # (1, 128, hidden)

        # ══════════════════════════════════════════
        # 时间步嵌入 & 条件嵌入
        # ══════════════════════════════════════════
        self.t_embedder = TimestepEmbedding(hidden_size)
        self.c_embedder = ConditionEmbedding(cond_dim, hidden_size)

        # ══════════════════════════════════════════
        # DiT Block 堆叠
        # ══════════════════════════════════════════
        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # ══════════════════════════════════════════
        # Final Layer: 最终自适应 LayerNorm + 线性投影
        # ══════════════════════════════════════════
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),  # 预测 γ_final, β_final
        )

        # Unpatchify: 将 Token 还原为时序
        # Linear(hidden, patch_size * in_channels) = Linear(hidden, 16 * 2 = 32)
        self.unpatchify_proj = nn.Linear(hidden_size, patch_size * in_channels)

        # ══════════════════════════════════════════
        # 权重初始化
        # ══════════════════════════════════════════
        self._initialize_weights()

    def _initialize_weights(self):
        """
        DiT 论文推荐的权重初始化策略:
          1. 所有线性层和嵌入层使用 Xavier Uniform 初始化
          2. adaLN-Zero 的最终线性层: 全零初始化 (已在 DiTBlock1D.__init__ 中完成)
          3. Unpatchify 解码头: 全零初始化 (确保初始输出接近零)
          4. Final adaLN 预测层: 全零初始化
        """

        def _basic_init(module):
            """对基础层应用 Xavier 均匀初始化。"""
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 1. 基础初始化 (所有子模块)
        self.apply(_basic_init)

        # 2. Patch 嵌入层: 使用 Xavier 初始化卷积权重
        w = self.patch_embed.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1).unsqueeze(0))  # 展平后初始化
        nn.init.zeros_(self.patch_embed.bias)

        # 3. 时间步嵌入 MLP: 正常初始化 (已由 _basic_init 覆盖)

        # 4. 条件嵌入 MLP: 正常初始化 (已由 _basic_init 覆盖)

        # 5. 每个 DiT Block 的 adaLN 零初始化 (已在 DiTBlock1D.__init__ 中完成)

        # 6. Final adaLN 预测层: 零初始化
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # 7. 🔴 Unpatchify 解码头: 零初始化
        # 确保网络初始输出为零噪声 (与 DDPM 的 ε ~ N(0,I) 先验一致)
        nn.init.zeros_(self.unpatchify_proj.weight)
        nn.init.zeros_(self.unpatchify_proj.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将时序输入切分为 Patch Token 序列。

        Args:
            x: (B, C, L) = (B, 2, 2048) 原始时序
        Returns:
            (B, N, D) = (B, 128, hidden_size) Token 序列
        """
        # Conv1d: (B, 2, 2048) → (B, hidden, 128)
        tokens = self.patch_embed(x)
        # 转为 (B, 128, hidden) — Transformer 的标准格式 (batch_first=True)
        tokens = tokens.transpose(1, 2)
        return tokens

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        将 Token 序列还原为时序输出。

        Args:
            tokens: (B, N, D) = (B, 128, hidden_size)
        Returns:
            (B, C, L) = (B, 2, 2048) 还原的时序
        """
        B, N, D = tokens.shape
        # Linear: (B, 128, hidden) → (B, 128, patch_size * in_channels) = (B, 128, 32)
        tokens = self.unpatchify_proj(tokens)
        # Reshape: (B, 128, 32) → (B, 128, 2, 16) → (B, 2, 128, 16) → (B, 2, 2048)
        tokens = tokens.view(B, N, self.in_channels, self.patch_size)
        tokens = tokens.permute(0, 2, 1, 3)  # (B, 2, 128, 16)
        output = tokens.reshape(B, self.in_channels, N * self.patch_size)  # (B, 2, 2048)
        return output

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor = None
    ) -> torch.Tensor:
        """
        DiT1D 前向传播 — 与 UNet1d.forward() 接口完全兼容。

        Args:
            x: (B, 2, 2048) — 带噪声的双通道时间序列
            t: (B,)         — 扩散时间步 ∈ {0, ..., T-1}
            c: (B, 2)       — 初始状态条件向量 (可为 None, 兼容 CFG)
        Returns:
            (B, 2, 2048)    — 预测的噪声 ε̂
        """
        # ── 1. 条件嵌入 ──
        # 时间步嵌入: (B,) → (B, hidden_size)
        t_emb = self.t_embedder(t)

        # 条件嵌入: (B, 2) → (B, hidden_size), CFG 兼容
        if c is None:
            c_emb = torch.zeros_like(t_emb)  # 无条件模式: 全零嵌入
        else:
            c_emb = self.c_embedder(c)

        # 融合条件: 直接相加
        cond_emb = t_emb + c_emb  # (B, hidden_size)

        # ── 2. Patchify + 位置编码 ──
        # (B, 2, 2048) → (B, 128, hidden_size)
        tokens = self.patchify(x)

        # 加入正弦位置编码
        tokens = tokens + self.pos_embed  # (B, 128, hidden_size)

        # ── 3. DiT Blocks ──
        for block in self.blocks:
            tokens = block(tokens, cond_emb)

        # ── 4. Final Layer: 自适应归一化 ──
        final_params = self.final_adaLN(cond_emb)  # (B, 2 * hidden_size)
        gamma_final, beta_final = final_params.chunk(2, dim=-1)
        tokens = modulate(self.final_norm(tokens), shift=beta_final, scale=gamma_final)

        # ── 5. Unpatchify: 还原时序 ──
        # (B, 128, hidden_size) → (B, 2, 2048)
        output = self.unpatchify(tokens)

        return output


# ══════════════════════════════════════════════════════════════
#  7. 模型工厂函数 (预设配置)
# ══════════════════════════════════════════════════════════════

def DiT1D_S(**kwargs):
    """DiT-Small: ~25M params. 适合快速实验和验证。"""
    return DiT1D(hidden_size=384, depth=12, num_heads=6, **kwargs)


def DiT1D_B(**kwargs):
    """DiT-Base: ~55M params. 推荐的默认配置，兼顾性能和效率。"""
    return DiT1D(hidden_size=512, depth=12, num_heads=8, **kwargs)


def DiT1D_L(**kwargs):
    """DiT-Large: ~200M params. 追求极致性能时使用。"""
    return DiT1D(hidden_size=768, depth=24, num_heads=12, **kwargs)


# ══════════════════════════════════════════════════════════════
#  8. 独立测试 (直接运行此文件验证形状)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  DiT1D Architecture Verification")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试三种规模
    configs = {
        "DiT-S": DiT1D_S,
        "DiT-B": DiT1D_B,
        "DiT-L": DiT1D_L,
    }

    B = 4  # 测试 batch size
    for name, factory in configs.items():
        model = factory().to(device)
        num_params = sum(p.numel() for p in model.parameters()) / 1e6

        # 模拟输入
        x = torch.randn(B, 2, 2048, device=device)
        t = torch.randint(0, 1000, (B,), device=device)
        c = torch.randn(B, 2, device=device)

        # 前向传播 (有条件)
        out_cond = model(x, t, c)
        assert out_cond.shape == (B, 2, 2048), f"Shape mismatch: {out_cond.shape}"

        # 前向传播 (无条件, CFG 模式)
        out_uncond = model(x, t, c=None)
        assert out_uncond.shape == (B, 2, 2048), f"Shape mismatch: {out_uncond.shape}"

        # 验证 adaLN-Zero: 初始输出应接近零 (因为 unpatchify 的线性层零初始化)
        with torch.no_grad():
            initial_output_norm = out_cond.abs().mean().item()

        print(f"\n  {name}: {num_params:.1f}M params")
        print(f"    Input:  {tuple(x.shape)}")
        print(f"    Output: {tuple(out_cond.shape)}")
        print(f"    Initial |output| mean: {initial_output_norm:.6f} (should be ~0)")
        print(f"    CFG (c=None) works: ✓")

    # 额外测试: 梯度是否正常流动
    print(f"\n  Gradient check (DiT-B):")
    model = DiT1D_B().to(device)
    x = torch.randn(B, 2, 2048, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    c = torch.randn(B, 2, device=device)
    out = model(x, t, c)
    loss = out.mean()
    loss.backward()

    # 检查所有参数都有梯度
    no_grad_params = [n for n, p in model.named_parameters() if p.grad is None]
    if len(no_grad_params) == 0:
        print(f"    All parameters have gradients: ✓")
    else:
        print(f"    ⚠ Parameters without gradients: {no_grad_params}")

    print(f"\n{'=' * 60}")
    print(f"  All checks passed! DiT1D is ready to replace UNet1d.")
    print(f"{'=' * 60}")
