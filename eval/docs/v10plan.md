# DiT-B v10 改进方案

> 综合外部诊断报告（548 维 cluster-family 分析）与内部 DDPM eval 结果

## 诊断数据汇总

### 外部报告诊断（PDF cluster-family gap）

| 排名 | Feature Family | Median Gap | P90 Gap | 性质 |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **roughness_texture** | **2.195** | **4.552** | 🔴 最大差距 + v8→v9 回退 |
| 🥈 | **regime_summary** | **1.052** | **1.800** | 🔴 v8→v9 回退 |
| 🥉 | **block_level_heterogeneity** | **1.015** | **4.188** | 🟡 改善但仍大 |
| 4 | **volatility_clustering** | **0.554** | **0.411** | 🟡 持续弱项 |
| 5 | **volatility_burst_dynamics** | **0.552** | **0.475** | 🔴 v8→v9 回退 |
| 6 | path_texture | 0.417 | 0.000 | 🟢 尚可 |
| ✅ | cross-asset dependence | 负值 | — | 🟢 已达标，**勿破坏** |
| ✅ | autocorrelation | 负值 | — | 🟢 已达标，**勿破坏** |

### 内部 DDPM eval（score.py v2，v9 20k）

| 指标 | v9 分 | Real | 差距 | 加权损失 |
|:---|:---:|:---:|:---:|:---:|
| DDPM MSE (20%) | 29.01 | ~50 | -21 | 4.20 |
| SP500 ACF (10%) | 36.98 | ~50 | -13 | 1.30 |
| DGS10 ACF (10%) | 39.58 | ~50 | -10 | 1.04 |
| 无条件相关 (15%) | 40.95 | ~50 | -9 | 1.35 |
| 尾部相关 (10%) | 47.27 | ~50 | -3 | 0.27 |

### 两套诊断的交叉验证

| 问题 | 外部报告对应 | 内部 eval 对应 | 一致性 |
|:---|:---|:---|:---:|
| 路径粗糙度过高 | roughness_texture #1 | DDPM MSE 偏高 | ✅ |
| 波动率聚集不足 | volatility_clustering #4 | SP500/DGS10 ACF 偏低 | ✅ |
| 波动率爆发不真实 | volatility_burst_dynamics #5 | ACF + 尾部相关 | ✅ |
| 跨资产相关已达标 | cross-asset gap 为负 | uncond_corr 40.95 (中等) | ⚠️ 需保护 |
| Regime 不真实 | regime_summary #2 | 无直接对应 | — |

---

## 改进方案（按优先级排序）

### 🏆 优先级 1：降低路径粗糙度（Roughness）

**问题根因**：DiT 的 DDIM 50 步采样在高频区域引入了过多锯齿/短频噪声。生成路径的 return increment 局部变化过大。

**方案 A：增加采样步数（最简单）**

```python
# generate.py: 将 --num_inference_steps 从 50 提高到 100-200
python generate.py --model dit-b --num_inference_steps 100 ...
```

- 预期效果：更多去噪步骤 → 更平滑的路径 → roughness 下降
- 成本：生成时间翻倍（~14min → ~28min），零训练成本
- 风险：极低

**方案 B：Noise Schedule 调整（中等复杂度）**

修改 [scheduler.py](file:///home/u00134/src/scheduler.py)，将 beta schedule 从线性改为 **cosine schedule**：

```python
# scheduler.py 新增 cosine schedule
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (Improved DDPM, Nichol & Dhariwal 2021)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

- 预期效果：cosine schedule 在高 t 区间更平缓 → 减少高频噪声残留
- 成本：**需要重新训练**（~13h）
- 风险：中等（需验证不破坏其他指标）

**方案 C：EMA 后平滑**

在 `generate.py` 中添加轻度低通滤波后处理：

```python
# 对生成路径施加指数移动平均平滑
def smooth_path(x, alpha=0.05):
    """轻度 EMA 平滑，alpha 越大平滑越强"""
    smoothed = x.clone()
    for t in range(1, x.shape[-1]):
        smoothed[..., t] = (1-alpha) * x[..., t] + alpha * smoothed[..., t-1]
    return smoothed
```

- 预期效果：直接降低高频锯齿
- 成本：零训练成本
- 风险：可能过度平滑 → 损害 distribution_tail

> [!IMPORTANT]
> **建议先做 A（增加采样步数到 100），再试 C（轻度 EMA），最后考虑 B（重训）。**

---

### 🥈 优先级 2：增强波动率聚集（Volatility Clustering + Burst）

**问题根因**：DDPM 的高斯噪声假设天然不利于 GARCH 效应。生成路径的 |r| 自相关不足，波动率爆发（burst onset/duration/recovery）不像真实市场。

**方案：辅助 ACF Loss + GARCH 后处理**

#### 2a. 训练侧：Auxiliary ACF Loss

在 [train.py](file:///home/u00134/src/train.py) 中增加辅助损失项：

```python
def acf_loss(x_pred, x_true, max_lag=5):
    """计算 |r| 自相关的 L1 距离"""
    abs_r_pred = x_pred[:, 0, :].abs()  # SP500 channel
    abs_r_true = x_true[:, 0, :].abs()
    loss = 0
    for lag in range(1, max_lag + 1):
        acf_pred = (abs_r_pred[:, lag:] * abs_r_pred[:, :-lag]).mean(dim=-1)
        acf_true = (abs_r_true[:, lag:] * abs_r_true[:, :-lag]).mean(dim=-1)
        loss += (acf_pred - acf_true).abs().mean()
    return loss / max_lag

# 在 train loop 中：
# total_loss = mse_loss + 0.1 * acf_loss(x0_pred, x0)
```

- 成本：需要重新训练
- 风险：权重调不好可能干扰主 loss

#### 2b. 生成侧：GARCH 后处理

```python
def garch_modulate(returns, omega=1e-6, alpha=0.08, beta=0.90):
    """对生成的收益率施加 GARCH(1,1) 波动率调制"""
    T = returns.shape[-1]
    sigma2 = torch.zeros_like(returns)
    sigma2[..., 0] = returns[..., 0] ** 2
    for t in range(1, T):
        sigma2[..., t] = omega + alpha * returns[..., t-1]**2 + beta * sigma2[..., t-1]
    # 将原始 returns 按 GARCH 方差重缩放
    raw_vol = returns.std(dim=-1, keepdim=True)
    garch_vol = sigma2.sqrt()
    modulated = returns * (garch_vol / (raw_vol + 1e-8))
    return modulated
```

- 成本：零训练成本，只改 generate.py
- 风险：低（参数可从真实数据拟合）

> [!TIP]
> **建议先做 2b（GARCH 后处理），如果效果不够再做 2a（辅助 loss 重训）。**

---

### 🥉 优先级 3：改善 Regime 结构

**问题根因**：生成路径的 calm/stress 持续时间和状态转移频率与真实数据不符。

**方案：Regime-Aware Conditioning**

在条件向量 `c` 中增加 regime 信息：

```python
# dataset.py: 计算窗口级 regime 统计
def compute_regime_features(window):
    """提取窗口级 regime 特征"""
    sp = window[0]  # SP500 returns
    rolling_vol = pd.Series(sp).rolling(21).std().values  # 21日滚动波动率
    high_vol = (rolling_vol > np.median(rolling_vol)).mean()  # 高波动占比
    regime_switches = np.diff((rolling_vol > np.median(rolling_vol)).astype(int))
    transition_count = np.abs(regime_switches).sum()  # 状态转移次数
    return high_vol, transition_count / len(sp)
```

- 将 `c` 从 `(B, 2)` 扩展为 `(B, 4)` — 增加 high_vol_ratio 和 transition_freq
- 成本：修改 dataset + 模型 cond_dim + 重训
- 风险：中高（改变条件接口）

> [!WARNING]
> 这是中高复杂度改动，建议在优先级 1、2 验证有效后再考虑。

---

### 优先级 4：DDPM 推理参数调优（零成本）

不需要重训，直接在 generate.py 上做 grid search：

```bash
# Guidance Scale grid search
for w in 1.5 2.0 2.5 3.0 4.0 5.0; do
  python generate.py --model dit-b \
    --checkpoint logs/deep_v9_dit_b_20k/checkpoint_final.pt \
    --scaler logs/deep_v9_dit_b_20k/scaler.pt \
    --num_samples 512 --guidance_scale $w \
    --num_inference_steps 100 \
    --output output/v9_grid_w${w}.csv
done

# 对每个 w 值评分
for w in 1.5 2.0 2.5 3.0 4.0 5.0; do
  python eval/score.py --model dit-b \
    --checkpoint logs/deep_v9_dit_b_20k/checkpoint_final.pt \
    --scaler logs/deep_v9_dit_b_20k/scaler.pt \
    --real /home/u00134/data/train_sp500_us10y.csv \
    --fake output/v9_grid_w${w}.csv \
    --json eval/v9_grid_w${w}.json
done
```

- 预期：找到 roughness vs quality 的最优 w 和 steps
- 成本：纯计算成本（每组 ~7min）
- 风险：零

---

### 优先级 5：Guardrails（保护 v9 改进）

**绝对不能破坏的指标**（v8→v9 已改善）：
- distribution_level / distribution_tail
- cross-asset dependence / dynamic cross-asset dependence
- block_21 短窗口特征
- Wasserstein score (80.99)

**验证规则**：每次实验后必须同时检查：
1. 总分是否提升
2. roughness/regime/burst gap 是否下降
3. 上述 guardrail 指标是否未回退

---

## 推荐实验时间线

| 阶段 | 实验 | 耗时 | 需重训? |
|:---|:---|:---:|:---:|
| **Phase 1** | 增加采样步数 100/200 + Grid Search w | ~2h | ❌ |
| **Phase 2** | GARCH 后处理 | ~1h | ❌ |
| **Phase 3** | 轻度 EMA 平滑 | ~30min | ❌ |
| **Phase 4** | Cosine Schedule 重训 | ~15h | ✅ |
| **Phase 5** | Auxiliary ACF Loss 重训 | ~15h | ✅ |
| **Phase 6** | Regime Conditioning 重训 | ~15h | ✅ |

> [!IMPORTANT]
> **核心策略：先做 Phase 1-3（零训练成本），验证效果后再决定是否进入 Phase 4-6（需重训）。这样最坏情况也只浪费几小时计算时间，不浪费 GPU 训练时间。**

## Open Questions

1. **采样步数 100 vs 200**：是否愿意接受生成时间翻倍以换取 roughness 改善？
2. **GARCH 后处理的参数来源**：是否从真实数据拟合 GARCH(1,1) 参数，还是手动设定？
3. **是否需要生成原生 1260 行路径**：PDF 报告指出当前 1024+236 bridge 存在 boundary artifact 风险，v10 是否考虑直接生成 1260 行？
