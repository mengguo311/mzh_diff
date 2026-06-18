# DiT-B v10 改进方案（数据驱动复核版）

> 在 `v10plan.md`（WdXq 原始方案）基础上，结合**新增诊断脚本 + 采样扫描实测**对诊断与优先级做了重大修正。
> 分支 `deep_v10`；运行环境 `conda activate ts_diffusion`。

---

## 0. TL;DR

1. **诊断纠偏**：v9 的真正病根**不是"路径太粗糙"，而是"过平滑 / 欠离散"**（生成路径比真实更平滑、收益率过度持续、regime 太黏滞、波动爆发不足）。这与 `v10plan.md` 的头号前提方向相反。
2. **零成本大赢**：给 DDIM 采样加 `eta`（随机性）参数后，最优配置 **`eta=1, steps=200, w=1`，全量 N=5120 复核 score.py 总分 47.16 → 50.23（+3.07，零训练）**，补回约 58% 与真实自检（52.47）的差距，且 guardrail 未破（512 子集扫描峰值 51.01）。
3. **`v10plan.md` 的多数提案被数据否决或降级**：EMA 平滑（反方向）、cosine/加步数（反方向）、regime conditioning（regime 已被 eta 修复）、patch 修复（无伪影）、GARCH（eta 已部分恢复波动结构）。
4. **剩余缺口（roughness / 波动爆发 / 收益率过持续）固化在模型里**，采样无法根治 → 决策门指向 **Phase 2 重训**，且重排了优先级（统计量辅助损失 + 放宽 ±5σ 截断 + min-SNR）。

---

## 1. 方法

为闭合"平衡"目标（既看 score.py，又看外部 PDF 的 roughness/regime/burst），新增了 score.py **未覆盖**的诊断脚本：

- **`eval/diagnostics.py`**（纯 numpy/scipy，口径与 score.py 对齐）：
  - roughness：二阶差分能量 `d2_energy`、总变差 `tv`、**收益率 lag-1 自相关 `ret_acf1`**（score.py 只算 |r| 的 ACF，不算原始收益率的符号自相关）
  - regime：滚动 21 日波动率的高波动占比 / 切换频率 / **平均持续长度 `mean_run_len`** / 波动爆发 `max_rolling_vol` / `vol_of_vol`
  - patch 伪影：功率谱在周期-16 谐波处的尖峰比（检验非重叠 patchify 是否产生接缝）
- **`eval/run_sweep.sh` + `eval/aggregate.py`**：对每个 (eta, steps, w) 跑 生成→score→diagnostics 并汇总成对比表。
- **`scheduler.py` / `generate.py`**：给 `ddim_sample_loop` 增加 `eta` 参数（`eta=0` 确定性=现状；`eta→1` 注入随机性≈DDPM ancestral；默认 0.0，旧结果可复现）。

---

## 2. 诊断纠偏（实测，5120 路径 vs 2538 真实窗口）

| 指标 | 真实 | v9 生成 | 方向判断 |
|:---|:---:|:---:|:---|
| `d2_energy`（二阶差分能量） | 6.96e-4 | 4.77e-4 | **fake 更平滑** |
| `tv`（总变差） | 1.03e-2 | 8.77e-3 | **fake 更平滑** |
| `ret_acf1`（收益率 lag-1 自相关） | +0.015 | +0.069 | **fake 过度持续/趋势化** |
| `mean_run_len`（regime 持续长度） | 32.9 | 52.8 | **fake regime 太黏滞** |
| `max_rolling_vol`（波动爆发） | 3.71e-2 | 2.58e-2 | **fake 爆发不足 ~30%** |
| `vol_of_vol` | 4.84e-3 | 3.67e-3 | **fake 波动聚集弱** |
| patch-16 尖峰 excess | 1.30 | 0.98 | **无 patch 伪影** |

**结论**：PDF 报告里的 `roughness_texture gap` 是特征空间的**无方向距离**；实测方向是 fake **缺少**真实数据的高频/波动纹理（太平滑），而非太粗糙。这是经典扩散模型 mode-averaging：ε-MSE 训练 + DDIM 确定性采样（**最后一步输出模型 x0 均值预测，天然平滑**）。

---

## 3. Phase 1 结果：零成本采样优化（已完成）

N=512 扫描（最优行已用 N=5120 复核，见 §3.1）：

| config (eta·steps·w) | TOTAL | ddpm_mse | sp_acf | dg_acf | wasser | sp_kurt | run_gap | vol_gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **e1 · 200 · 1** ⭐ | **51.01** | 43.05 | 38.85 | 42.44 | 86.96 | 59.02 | +3.59 | -8.8e-3 |
| e1 · 200 · 3 | 50.57 | 40.28 | 38.23 | 42.31 | 87.96 | 58.99 | +3.58 | -8.8e-3 |
| e1 · 100 · 1.5 | 50.16 | 39.82 | 38.36 | 41.18 | 85.72 | 59.18 | +3.27 | -9.4e-3 |
| e1 · 100 · 3 | 50.15 | 38.62 | 38.07 | 41.80 | 87.50 | 59.19 | +2.96 | -9.0e-3 |
| e1 · 50 · 3 | 48.50 | 33.91 | 40.02 | 40.71 | 82.92 | 59.14 | +6.26 | -1.1e-2 |
| **v9_20k 基线** | **47.16** | 29.01 | 36.98 | 39.58 | 80.99 | 60.00 | — | — |
| e0 · 50 · 3（确定性） | 46.80 | 27.73 | 37.22 | 39.09 | 81.62 | 60.12 | +19.91 | -1.1e-2 |

**要点**：
- 收益主要来自 **`ddpm_mse` 27.7→43.1**（权重 0.20，单项贡献 ≈ +3.0）：随机采样把过平滑路径拉回模型流形。
- `wasserstein` 81→87、`dg_acf` 39→42 改善；regime 黏滞 `run_gap` +19.9→+3.6（**基本修复**）。
- guardrail：`sp_kurt` 60→59（-1.1）、`tail_corr` 稳定、`uncond_corr` 持平/升 —— **均未破**。
- 步数拐点 ~200（50→100→200 单调升，400 收益反降且 8× 慢）；`w=1` 略优于 `w=3` 且无需 CFG 放大。

### 3.1 推荐生成配置（新默认）

```bash
conda run -n ts_diffusion python generate.py --model dit-b \
  --checkpoint logs/deep_v9_dit_b_20k/checkpoint_final.pt \
  --scaler    logs/deep_v9_dit_b_20k/scaler.pt \
  --num_samples 5120 --cond_mode dataset \
  --num_inference_steps 200 --guidance_scale 1.0 --eta 1.0 \
  --output output/deep_v10.csv
```

> **全量 N=5120 复核：TOTAL = 50.23**（基线 47.16，+3.07）。产物：`output/deep_v10.csv`、`eval/v10_score.json`、`eval/diag_v10.json`、`outputs/figures/diag_v10.png`。
> 诊断 vs v9 基线：`d2_energy` gap -2.25e-4→-1.69e-4（粗糙度 ~25%↓）、`mean_run_len` gap +19.9→+3.39（regime 基本修复）、`max_rolling_vol` gap -1.13e-2→-9.6e-3（爆发 ~15%↑）；roughness/爆发仍有残余 → Phase 2 目标。

---

## 4. 决策门：剩余缺口与 Phase 2 重训方案

即便用近似完整 DDPM 采样，以下缺口也只补回 15~40% —— **固化在模型里，须重训**：

- roughness（`d2_gap` 仍 ~-1.4e-4，约 -40%）
- 波动爆发（`vol_gap` 仍 ~-8.8e-3，约 -24%）
- 收益率过持续（`ret_acf1` 0.065 vs 0.015）

根因：ε-MSE 均值寻优 + 最后一步输出 x0 均值；以及训练端 **±5σ 硬截断**抹掉了驱动波动爆发的尾部。

### 重排后的 Phase 2 优先级（每次重训 ~13–15h，需逐项确认）

| 优先级 | 方案 | 攻击目标 | 改动 | 备注 |
|:---:|:---|:---|:---|:---|
| **P1** | **统计量辅助损失**：由 ε̂ 反演 x̂₀，对 \|r\| ACF + 滚动波动分布(vol-of-vol/burst) + 二阶差分能量 的 gap 加 L1 惩罚（低 t 加权，权重 ~0.05–0.1） | roughness + 波动聚集 + 爆发 | `train.py` | 把 v10plan 的"ACF loss"扩成更全的 stylized-fact 匹配；最直接 |
| **P2** | **放宽 ±5σ 截断**（`config.CLIP_RANGE` 5→8 或软截断） | 波动爆发 / 厚尾 | `config.py`+`dataset.py` | v10plan 未提；数据侧根因，便宜 |
| **P3** | **min-SNR-γ 损失加权**（γ=5） | 整体保真/纹理 | `train.py` | 便宜，可与 P1/P2 同时上 |
| ~~×~~ | ~~cosine schedule / 加采样步数 / EMA 平滑~~ | — | — | **反方向**，已否决 |
| ~~×~~ | ~~regime conditioning~~ | — | — | regime 已被 eta 修复，**无需** |
| ~~×~~ | ~~patch 边界修复~~ | — | — | 实测无 patch 伪影，**无需** |
| ~~×~~ | ~~GARCH 后处理~~ | — | — | 风险高(伤 wasserstein/峰度)，eta 已部分恢复 |

> 建议先单独验证 P2（最便宜），再做 P1（+P3 同训）。每次重训后用 `eval/score.py` + `eval/diagnostics.py` 双重验证，并守住 guardrail（`wasserstein`≥85、`sp_kurt`、`tail_corr`、`uncond_corr` 不显著回退）。

---

## 5. 复现与文件清单

**复现 Phase 1**：
```bash
N=512 bash eval/run_sweep.sh        # 扫描 + 汇总
python3 eval/aggregate.py           # 仅重打汇总表
```

**本次新增/改动文件**（分支 `deep_v10`）：
- 新增 `eval/diagnostics.py`、`eval/run_sweep.sh`、`eval/aggregate.py`、本报告
- 改 `scheduler.py`（`ddim_sample_loop` 加 `eta`）、`generate.py`（加 `--eta`，默认 0.0 不改变旧行为）
- 产物：`output/deep_v10.csv`、`eval/v10_score.json`、`eval/diag_v10.json`、`outputs/figures/diag_v10.png`、各 `eval/{score,diag}_sw_*.json`

**未改**：训练/模型架构（Phase 2 待批）、生成 2048 / eval 1260 的现状（实测无 bridge 问题）。
