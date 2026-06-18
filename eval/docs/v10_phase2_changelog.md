# v10 Phase 2 重训优化 — 代码改动日志

> 分支 `deep_v10`。目标：修复 v9 诊断出的**过平滑 / 欠离散**（生成路径比真实更平滑、
> 波动聚集弱、波动爆发不足、收益率过度持续）——这部分缺口固化在模型里，采样端
> （Phase 1 的 eta）无法根治，必须重训。环境 `conda activate ts_diffusion`。

## 0. 三项优化（数据驱动，对应报告 P1–P3）

| 标签 | 优化 | 攻击目标 | 涉及文件 |
|:---:|:---|:---|:---|
| **P1** | stylized-fact 辅助损失（\|r\| ACF + roughness） | 波动聚集 + roughness | `losses.py`, `train.py`, `config.py` |
| **P2** | 放宽 ±5σ→±8σ 硬截断 | 波动爆发 / 厚尾 | `config.py`, `dataset.py`(自动生效) |
| **P3** | min-SNR-γ 损失加权（ε-pred, γ=5） | 整体保真 / 收敛 | `losses.py`, `train.py`, `config.py` |

全部默认开启；逐项可在 `config.py` 关闭以做消融（基线复现：三项 `False` + `CLIP_RANGE=5.0`）。

---

## 1. 新增文件 `losses.py`

封装三项优化的纯函数（仅依赖 torch）：

- `min_snr_weight(scheduler, t, gamma)` → 逐样本 `min(SNR,γ)/SNR` 权重（ε-pred）。
- `recover_x0(scheduler, xt, t, eps)` → 由噪声预测反演 `x̂₀ = (x_t−√(1−ᾱ_t)·ε̂)/√ᾱ_t`。
- `_abs_acf(x, max_lag)` → \|x\| 的 lag 1..L 自相关（可微）。
- `_d2_energy(x)` → 二阶差分能量（roughness 指标）。
- `stylized_aux(x0_hat, x0, max_lag)` → 逐样本 (ACF L1 差距, roughness 相对差距)。

**关键设计**：辅助损失作用在反演的 `x̂₀` 上，且**仅在低噪声步（ᾱ_t > `AUX_ABAR_MIN`）施加、按 ᾱ_t 加权**——因为高噪声步 `x̂₀` 不可靠。roughness 用相对差距（尺度无关）。

## 2. `config.py`

```diff
- CLIP_RANGE      = 5.0       # 硬截断阈值 ±5σ
+ CLIP_RANGE      = 8.0       # ±8σ (v10 P2: 5→8 放宽，保留驱动波动爆发的尾部)
```
新增两个配置块：
- **Generation（v10 采样默认）**：`GEN_ETA=1.0`, `GEN_NUM_STEPS=200`, `GEN_GUIDANCE_SCALE=1.0`
  （Phase 1 验证的最优工作点，被 `generate.py` 用作默认值）。
- **Phase 2 训练开关**：`USE_MIN_SNR=True`, `MIN_SNR_GAMMA=5.0`, `USE_AUX_LOSS=True`,
  `AUX_ACF_WEIGHT=0.05`, `AUX_ROUGH_WEIGHT=0.05`, `AUX_ACF_MAX_LAG=5`, `AUX_ABAR_MIN=0.1`。

## 3. `train.py`

- `import losses`。
- 主损失由 `F.mse_loss(noise_pred, noise)` 改为 **逐样本 MSE × min-SNR 权重**。
- 增加 **stylized-fact 辅助损失**（`USE_AUX_LOSS` 时）：反演 `x̂₀` → 取 ᾱ_t>0.1 的可靠样本
  → `clamp(±3·CLIP_RANGE)` → 算 ACF/roughness 差距 → 按 ᾱ_t 加权 → 乘权重。
- `loss = loss_mse + loss_acf + loss_rough`。
- 训练头打印 Phase-2 配置；epoch 日志增加分项 `(mse … acf … rgh …)`。

## 4. `generate.py` / `scheduler.py`（Phase 1 已改，v10 沿用）

- `scheduler.ddim_sample_loop` 新增 `eta`（广义 DDIM：0=确定性，1≈DDPM）。
- `generate.py` 新增 `--eta`，且 `--num_inference_steps/--guidance_scale/--eta` 默认值改读 `config.GEN_*`。

---

## 5. 冒烟验证（已通过）

`--epochs 2` 微训：损失有限、正常下降、无 NaN。分项 `mse≈0.61→0.22`、`acf≈0.001`、
`rough≈0.006–0.015`（辅助项约占总损失 2–3%，温和）。每 epoch ~7.5s（与 v9 的 ~7.0s 相当，
辅助损失开销可忽略）。

## 6. 运行（手动）

> ⚠️ 预计耗时 **~23–24 小时**（v9 同配置 20000 epoch 实测 83802s≈23.3h；本机 RTX A6000）。
> 单卡训练，用 `CUDA_VISIBLE_DEVICES` 指定空闲卡，避免与生成/评分争用。

```bash
cd /home/u00134/src
# 先确认空闲 GPU: nvidia-smi  (下面用 0 号卡，可改)
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n ts_diffusion \
  python -u train.py \
    --model dit-b --epochs 20000 --batch_size 64 --lr 2e-4 --warmup 200 \
    --run_name deep_v10_dit_b \
  > logs/run_dit_v10.log 2>&1 &

# 跟踪进度（应每 10 epoch 一行，含 mse/acf/rgh 分项）：
tail -f logs/run_dit_v10.log
```

产物：`logs/deep_v10_dit_b/checkpoint_final.pt` 与 `scaler.pt`（clip=8）。

## 7. 训练后评估

```bash
# 用 v10 采样默认 (eta=1/steps=200/w=1) 生成
conda run -n ts_diffusion python generate.py --model dit-b \
  --checkpoint logs/deep_v10_dit_b/checkpoint_final.pt \
  --scaler    logs/deep_v10_dit_b/scaler.pt \
  --num_samples 5120 --cond_mode dataset --output output/deep_v10_retrained.csv

# 评分 + 诊断（注意 --scaler 用 v10 的，clip=8）
conda run -n ts_diffusion python eval/score.py --model dit-b \
  --checkpoint logs/deep_v10_dit_b/checkpoint_final.pt --scaler logs/deep_v10_dit_b/scaler.pt \
  --real /home/u00134/data/train_sp500_us10y.csv --fake output/deep_v10_retrained.csv \
  --json eval/v10_retrained_score.json

conda run -n ts_diffusion python eval/diagnostics.py --fake output/deep_v10_retrained.csv \
  --json eval/diag_v10_retrained.json --fig outputs/figures/diag_v10_retrained.png --label v10_retrained
```

**验收标准**：相对"仅 eta 优化的 v10 基线"，`d2_gap`(roughness) 与 `vol_gap`(波动爆发) 继续缩小、
`sp_acf/dg_acf` 上升；同时 guardrail 不破（`wasserstein`≥85、`sp_kurt`、`tail_corr`、`uncond_corr` 不显著回退）。

## 8. 消融建议（若想分辨各项贡献，可单独训练）

- 最便宜先验证 **P2**（仅 `CLIP_RANGE=8`，其余 False）——纯数据侧，无额外训练开销。
- 再 **P1+P3** 同训。
- 若辅助损失影响主 loss 收敛，调低 `AUX_*_WEIGHT`（如 0.02）或调高 `AUX_ABAR_MIN`（如 0.2，更保守）。

## 9. 改动文件清单

| 文件 | 状态 | 说明 |
|:---|:---:|:---|
| `losses.py` | 新增 | min-SNR + stylized-fact 辅助损失 |
| `config.py` | 改 | CLIP_RANGE 5→8；新增 GEN_* 与 Phase-2 开关 |
| `train.py` | 改 | 集成 min-SNR 加权 MSE + 辅助损失 + 分项日志 |
| `generate.py` | 改 | 采样默认值改读 config.GEN_*（eta 在 Phase 1 已加） |
| `scheduler.py` | 改 | ddim_sample_loop 加 eta（Phase 1） |
