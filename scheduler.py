"""
scheduler.py — DDPM 扩散调度器
阶段三：独立的 DDPMScheduler，与 U-Net 完全解耦。

数学核心:
    前向加噪:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    逆向去噪:  μ_θ = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε̂_θ)
              x_{t-1} = μ_θ + √β̃_t · z   (z ~ N(0,I) if t>0)

所有预计算的系数通过 register_buffer 缓存，不参与梯度计算。
"""

import torch
import torch.nn as nn

import config


class DDPMScheduler(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) Scheduler.
    
    参数:
        num_timesteps: 总扩散步数 T (默认 200)
        beta_start:    β₁ (默认 1e-4)
        beta_end:      β_T (默认 0.02)
    
    缓存的张量 (register_buffer):
        betas:                         (T,)
        alphas:                        (T,)
        alphas_cumprod:                (T,)  ᾱ_t
        alphas_cumprod_prev:           (T,)  ᾱ_{t-1}
        sqrt_alphas_cumprod:           (T,)
        sqrt_one_minus_alphas_cumprod: (T,)
        sqrt_recip_alphas:             (T,)
        posterior_variance:            (T,)  β̃_t
    """

    def __init__(
        self,
        num_timesteps: int = config.T,
        beta_start: float = config.BETA_START,
        beta_end: float = config.BETA_END,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        # ── 线性 Beta 调度 ──
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

        # ── 预计算所有系数 ──
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # ᾱ_{t-1}: 在前面补 1.0（t=0 时 ᾱ_{-1} = 1.0）
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # 前向加噪系数
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # 逆向去噪系数
        sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)

        # 后验方差: β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # 第一步 (t=0) 的后验方差为 0（直接返回均值）
        posterior_variance[0] = 0.0

        # ── 注册为 buffer（不参与梯度，但会跟随 .to(device)） ──
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("posterior_variance", posterior_variance)

    def _extract(self, coeff: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        按 batch 索引提取系数，并 reshape 为可广播的形状。
        
        Args:
            coeff:   (T,)     — 预计算的系数序列
            t:       (B,)     — 当前时间步索引
            x_shape: tuple    — 目标张量的形状 (B, C, L)
        Returns:
            (B, 1, 1) — 可与 (B, C, L) 广播的系数张量
        """
        batch_size = t.shape[0]
        out = coeff.gather(-1, t.long())  # (B,)
        return out.reshape(batch_size, 1, 1)  # (B, 1, 1)

    # ──────────────────────────────────────────────
    # 前向加噪: q(x_t | x_0)
    # ──────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向加噪: 从 x_0 直接一步计算 x_t。
        
        公式: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        
        Args:
            x0:    (B, 2, 128) — 干净的数据（已标准化）
            t:     (B,)        — 时间步索引
            noise: (B, 2, 128) — 可选的预生成噪声，默认随机生成
        Returns:
            x_t:   (B, 2, 128) — 第 t 步的带噪数据
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    # ──────────────────────────────────────────────
    # 逆向去噪: p(x_{t-1} | x_t)
    # ──────────────────────────────────────────────

    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步逆向去噪: x_t → x_{t-1}。
        
        公式:
            μ_θ = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε̂_θ(x_t, t))
            x_{t-1} = μ_θ + √β̃_t · z  (if t > 0)
        
        Args:
            model: U-Net 去噪网络
            xt:    (B, 2, 128) — 当前步的带噪数据
            t:     (B,)        — 当前时间步（batch 内所有样本的 t 相同）
        Returns:
            x_{t-1}: (B, 2, 128)
        """
        with torch.no_grad():
            # 预测噪声
            noise_pred = model(xt, t)

            # 提取当前步的系数
            sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, xt.shape)
            beta = self._extract(self.betas, t, xt.shape)
            sqrt_one_minus_alpha_bar = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, xt.shape
            )

            # 计算预测均值
            mu = sqrt_recip_alpha * (xt - beta / sqrt_one_minus_alpha_bar * noise_pred)

            # t > 0: 添加方差噪声；t = 0: 直接返回均值
            t_val = t[0].item()
            if t_val > 0:
                posterior_var = self._extract(self.posterior_variance, t, xt.shape)
                noise = torch.randn_like(xt)
                return mu + torch.sqrt(posterior_var) * noise
            else:
                return mu

    # ──────────────────────────────────────────────
    # 完整逆向循环: T → 0
    # ──────────────────────────────────────────────

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple = None,
        x_T: torch.Tensor = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        完整的逆向去噪循环: x_T → x_{T-1} → ... → x_0。
        
        Args:
            model:   U-Net 去噪网络（已设为 eval 模式）
            shape:   生成张量的形状 (B, 2, 128)，与 x_T 二选一
            x_T:     初始纯噪声张量，若提供则忽略 shape
            verbose: 是否打印进度
        Returns:
            x_0: (B, 2, 128) — 生成的干净数据（标准化空间）
        """
        device = next(model.parameters()).device

        if x_T is not None:
            x = x_T.to(device)
        else:
            assert shape is not None, "Must provide shape or x_T"
            x = torch.randn(shape, device=device)

        model.eval()

        for i in reversed(range(self.num_timesteps)):
            batch_size = x.shape[0]
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)

            if verbose and (i % 50 == 0 or i == self.num_timesteps - 1):
                print(f"  [Scheduler] Reverse step {i:>3d}/{self.num_timesteps}  "
                      f"x range: [{x.min():.3f}, {x.max():.3f}]")

        return x

    def ddim_sample_loop(
        self,
        model: nn.Module,
        c: torch.Tensor,
        shape: tuple = None,
        x_T: torch.Tensor = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        确定性 DDIM 快速采样循环 (eta = 0)，结合 Classifier-Free Guidance (CFG)。
        
        Args:
            model:               U-Net 噪声预测模型
            c:                   条件向量 (B, cond_dim)
            shape:               生成张量的形状 (B, 2, seq_len)，与 x_T 二选一
            x_T:                 初始纯噪声张量，若提供则忽略 shape
            num_inference_steps: 快速采样步数 (例如 50)
            guidance_scale:      引导权重 w (例如 3.0 ~ 5.0)，w=1.0 为纯有条件，w=0.0 为纯无条件
            verbose:             是否打印进度
        Returns:
            x_0:                 (B, 2, seq_len) — 生成的干净数据
        """
        device = next(model.parameters()).device

        if x_T is not None:
            x = x_T.to(device)
        else:
            assert shape is not None, "Must provide shape or x_T"
            x = torch.randn(shape, device=device)

        B, _, L = x.shape
        model.eval()

        # 生成均匀跳过的时间步序列，例如：[-1, 19, 39, ..., 999] (当 T=1000, steps=50 时)
        times = torch.linspace(-1, self.num_timesteps - 1, num_inference_steps + 1, device=device)
        times = times.round().long()

        for i in reversed(range(1, len(times))):
            t_curr_val = times[i].item()
            t_prev_val = times[i - 1].item()

            t_curr = torch.full((B,), t_curr_val, device=device, dtype=torch.long)

            # CFG 提速技巧：单次模型前向同时处理有条件与无条件
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_curr, t_curr], dim=0)
            
            c_null = torch.zeros_like(c)
            c_double = torch.cat([c, c_null], dim=0)

            with torch.no_grad():
                eps_double = model(x_double, t_double, c_double)

            eps_cond, eps_uncond = torch.chunk(eps_double, 2, dim=0)

            # CFG 外推公式
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # 获取 alpha_bar 系数
            alpha_bar_curr = self.alphas_cumprod[t_curr_val].view(1, 1, 1)
            if t_prev_val >= 0:
                alpha_bar_prev = self.alphas_cumprod[t_prev_val].view(1, 1, 1)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device).view(1, 1, 1)

            # 预测干净样本 x_0
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_curr) * eps_pred) / torch.sqrt(alpha_bar_curr)

            # 计算指向 x_t 的确定性方向
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * eps_pred

            # 确定性更新得到 x_{t-1} (eta = 0)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

            if verbose and (i % max(1, num_inference_steps // 10) == 0 or i == len(times) - 1 or i == 1):
                print(f"  [DDIM Scheduler] Step {num_inference_steps - i + 1:>2d}/{num_inference_steps} "
                      f"(t_curr={t_curr_val:>3d} -> t_prev={t_prev_val:>3d}) | "
                      f"x range: [{x.min():.3f}, {x.max():.3f}]")

        return x
