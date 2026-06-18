# 金融时间序列 DDPM 项目改进路线图 (Lead Architect 综合提案)

> 范围：`/home/u00134/src` — DiT-B 1D 扩散模型 (2 通道日频 SP500 log-return + DGS10 yield-diff, SEQ_LEN=2048, T=1000, eps-prediction, linear schedule, DDIM+eta, CFG with c=window[0])。
> 既有结论：v9 **过度平滑/欠离散**；`eta=1` 零成本把 score.py 从 47.16 抬到 50.23；Phase-2 (min-SNR γ=5 + stylized-fact aux loss + clip 5σ→8σ) 路径最真实但 TOTAL 跌到 48.58，**根因是 `score.py` 的 `ddpm_mse` (weight 0.20, line 153) 自指**——它衡量样本到模型自身过平滑流形的距离，惩罚更真实的模型。
>
> 本文档由多智能体研究 workflow 综合而成（5 维度 × 39 提案 × 对抗验证，21 条「同时增益生成+鉴别」）。

---

## 1. 核心论点 (Thesis)

用户的两个诉求在数学上是**同一个对象的两面**，路线图的脊梁就是这一点：

- **目标 (1) 生成器**：让 `G` 生成的合成路径分布 `Q` 逼近真实分布 `P`。
- **目标 (2) 鉴别器**：构造一个非自指的统计量 `D(·)`，对**任何**第三方伪造的 `Q'` 给出"离 `P` 多远"的取证打分。

任何一个**真实数据 vs 候选数据**的两样本散度 `D(P, Q)`，天然同时服务两者：
- 把它作为 **loss** 反向传播 `∂D/∂θ` 去训练 `G` → 改进生成；
- 把它作为 **score** 在冻结状态下评估外部样本 → 取证鉴别。

这正是 `score.py` 现行 `ddpm_mse` 失败的反面：`ddpm_mse` 是 `D(G_自身, candidate)`（自指、model-tied），而我们要的是 `D(P_real, candidate)`（data-vs-data、model-free）。**只要把打分对象从"模型流形"换成"真实数据"，自指悖论自动消失，且同一对象可回流训练。**

在所有候选 `D` 中，**path-signature 方法是唯一一族同时满足"严格正确刻画随机过程的律"+"可微可训练"+"可作两样本检验"的对象**：

- **Sig-MMD / signature-kernel score**（Chevyrev & Oberhauser, JMLR 2022; Salvi-Cass-Foster-Lyons-Yang, SIAM JMDS 2021; Issa-Horvath-Lemercier-Salvi, NeurIPS 2023）是**严格 proper scoring rule**，无法被自身过平滑流形 game；
- **Sig-Wasserstein**（Ni-Szpruch-Wiese-Liao-Xiao, ICAIF 2021; Liao et al., Mathematical Finance 2024）把 GAN min-max 退化为对期望签名的**解析回归**，训练稳定且其 `Sig-W1` 距离直接是鉴别统计量；
- 同一个 `eval/signature.py` 既给 `train.py` 提供 aux loss，又给 `eval/score.py` 提供 permutation-calibrated 检验。**一份代码，两个目标。**

围绕这条脊梁，其余金融方法 (GARCH/rough-vol/EVT/copula/HMM) 提供**互补的、抗规避的取证轴**（签名捕捉路径动态，但单独一个标量统计量易被针对性伪造），而 diffusion-SOTA 方法 (v-pred/EDM/cosine-ztSNR) 仅服务生成侧。

---

## 2. 生成模型改进 (Generator)

按"先零成本采样侧、后训练侧、再架构侧"排列。**所有验收一律用 realism 指标 (wasserstein / kurtosis / |r|-ACF / `diagnostics.py` roughness·burst·regime)，绝不用自指的 `ddpm_mse` TOTAL 把关。**

### 2.1 采样侧（无需重训，最高性价比）
- **EDM-style churn 采样** — `scheduler.py` 新增 `churn_sample_loop`，复用现有 CFG-batched forward 与 `x0_pred`：每步以 `gamma=min(S_churn/steps, sqrt(2)-1)`（限制在 `t∈[S_tmin,S_tmax]`）抬升噪声、按 `S_noise≈1.003` 补偿，再走原 DDIM update。直接在 `abar` 上做 churn（不转连续 σ，避免脆弱换算）。`config.py` 加 `SAMPLER∈{ddim,churn}, S_CHURN, S_TMIN, S_TMAX, S_NOISE`。这是 `eta=1` 成功的细粒度推广，给"在中等噪声层定点注入波动率爆发"的旋钮 (Karras et al., NeurIPS 2022)。**保留 `eta` 路径作 baseline；丢弃 DPM-Solver++/Heun ODE 半（确定性方向与欠离散症相反）。**
- **Discriminator Guidance（采样时）** — 训练一个 noise-conditioned 判别器 `d_φ(x_t,t)`（real 加噪 vs v9 样本加噪），在 `ddim_sample_loop` 里把 `grad_x log[d/(1-d)]` 加到 `eps_pred` 旁。比对抗微调稳定得多，常能拿到大部分生成收益 (Kim et al., ICML 2023, arXiv:2211.17091)。

### 2.2 训练损失侧（需重训）—— 这是 realism 的真正驱动
- **【P1·both】Signature aux loss** — 新建 `eval/signature.py`（PDE signature kernel, `sigkernel` 库 / `iisignature` 截断 depth 3–4）。在 `losses.py` 对 `recover_x0` 得到的 `x0_hat` 做：**time-augmentation + lead-lag → 随机裁剪 ~128–256 子窗（不可对整条 L=2048 直接取签名，否则被净增量主导、对粗糙度盲视）→ sig-MMD / kernel-score**。复用现有 `abar>AUX_ABAR_MIN` 低噪声门控与 abar 加权，`config.py` 加 `SIG_MMD_WEIGHT`，从小权重 ramp。它惩罚 eps-MSE 平均掉的时序/跨通道动态，直击过平滑 (Chevyrev & Oberhauser 2022; Issa et al. 2023)。**与 stylized_aux 重叠，应做消融择一或互补，不盲目叠加。**
- **【P2·both】Spectral (PSD-of-|r|) loss** — `losses.py` 加 `spectral_l1(x0_hat, x0)`：对 `|x0_hat[:,ch,:]` 去均值后 `torch.fft.rfft`，取 `log(1+|FFT|)` 的 L1，逐通道，丢弃/降权 DC bin。**匹配 |r| 的功率谱（波动率聚集所在），非原始 return 谱**。像 `loss_acf`/`loss_rough` 一样接线，`AUX_SPEC_WEIGHT≈0.02–0.05`。全局补足 lag-1..5 ACF 覆盖不到的 roughness/burst (Diffusion-TS, Yuan & Qiao, ICLR 2024; Cont 2001)。wavelet 项设默认 0、低优先。
- **【P2·both】rough-vol Hurst aux** — `losses.py` 加 `rough_vol`：`v_t=EWMA(x0_hat^2)` → `log(v_t+eps)` → q=2 structure function `S(Δ)=mean((logv_{t+Δ}-logv_t)^2)` over `Δ∈{1,2,5,10,21}` → 拟合斜率得 `2H_hat`，惩罚 `|H_hat - H_real|`。**`H_real` 必须从真实数据用同一管线经验估计，不可硬编码 0.1**（日频 EWMA proxy 偏差大，Cont-Das 2024 fact-or-artefact）。低噪声门控、权重 ~0.05 (Gatheral-Jaisson-Rosenbaum 2018)。
- **【P3·gen】leverage / tail / copula aux（仅 SP500 通道）** — 杠杆项 = 真实 vs `x0_hat` 的 `corr(r_t,|r_{t+k}|)` 符号不对称（**不要用 `corr(r,σ)` 混淆聚集与杠杆**，Glosten-Jagannathan-Runkle 1993）；tail 项 = `|x0_hat|` 高分位 (99/99.5%) 的可微 soft-quantile 匹配 (McNeil-Frey 2000)；copula 项 = 平滑联合下尾超越占比 (Sklar 1959)。**全部小权重、低噪声门控、利率通道不施加杠杆/EVT**；先用 `diagnostics.py` 确认 gap 符号再加（过平滑常使 vol-memory 已经太持久，盲目"加慢衰减"会适得其反）。

### 2.3 参数化/调度/架构侧（需重训，生成 only）
- **【P2·gen】v-prediction** — `scheduler.py` 加 `predict_v(x0,t,noise)=sqrt(abar)*noise - sqrt(1-abar)*x0`，`x0_from_v(xt,t,v)=sqrt(abar)*xt - sqrt(1-abar)*v`，`eps_from_v(xt,t,v)=sqrt(1-abar)*xt + sqrt(abar)*v`（**注意系数已纠正**）。`train.py` 目标改 `v`，min-SNR 权重切到 v-pred 形式 `min(SNR,γ)/(SNR+1)`。`losses.recover_x0` 改吃 `x0_from_v`（更干净，绕开 buggy 公式）。`ddim_sample_loop` 里先 `x0_pred=x0_from_v(...)`，需要 `dir_xt` 时再 `eps_from_v`。**加 round-trip 单测 <1e-5** (Salimans & Ho, ICLR 2022; Hang et al., ICCV 2023)。
- **【P2·gen】cosine + zero-terminal-SNR schedule** — 仅与 v-pred 配对（eps+ztSNR 终端除零）。按 Lin et al. (WACV 2024, arXiv:2305.08891) `rescale_zero_terminal_snr` 改 `sqrt_alphas_cumprod` 并**重算 `scheduler.py` 所有依赖 buffer**。当前 linear schedule 终端 SNR=4.0e-5（残留信号 0.6%，比 SD 温和 ~10×），故为温和补充而非银弹。
- **【P3·gen】EDM 完整预条件** — 仅作生成 only 消融分支 `edm_scheduler.py`，per-channel `SIGMA_DATA`（实测，**勿硬编码 1.0**），用**随机** Algorithm-2 采样器（保留 eta=1 的随机性收益）。`dit1d.py` 的 `TimestepEmbedding`(base 10000) 对 `c_noise=0.25*ln(σ)∈[-4,4]` 失配，须重拟合频率基或 rescale。**EDM 替代 min-SNR（互斥），不替代 stylized aux。** 高成本、不解决根悖论。
- **【P2·both】regime/realized-vol 条件**（替换近乎无用的 `c=window[0]`）— CoFinDiff 式**窗级属性条件**（arXiv:2503.04164, IJCAI 2025）。在真实 return 上 fit 2–3 态 Gaussian HMM（`hmmlearn`，固定 seed，按方差排序），降采样为每 ~64 天一标签 → `(32,)` regime 序列 + 窗均 realized-vol 标量。**序列须 per-token 注入**（additive embedding / cross-attention 进 `DiTBlock1D`），因 adaLN-Zero 把单向量广播到全部 128 token，单一静态标签会**抹平窗内 regime 动态、反而恶化聚集**。廉价首试：仅标量 realized-vol 经现有 `ConditionEmbedding` cond_dim bump。

> **流派选择建议**：Flow Matching / Consistency 蒸馏列为 P3 生成 only 实验，**不围绕它们建鉴别器**（FM exact-likelihood 在 4096 维噪声/昂贵，且对竞品 on-manifold 生成器 OOD 不可靠，Nalisnick et al. 2019；蒸馏倾向再平滑）。

---

## 3. 鉴别/打分模型 (Discriminator for others' fakes)

**第一原则：所有鉴别统计量必须是 `D(P_real, candidate)`，绝不引用本模型流形。** 在 `eval/` 下新建独立模块，输出 per-window `p(fake)` 与校准 p-value/AUC。

### 3.1 修复 `score.py` 自指缺陷（P1，低成本，立即做）
1. **降权而非删除** `ddpm_mse` (line 153) 从 0.20 → ~0.05，明确标注"self-referential — diagnostic only"，保留 `compute_ddpm_mse_per_path` 以维持 v9-vs-Phase2 可审计性。
2. 释放出的 ~0.15 权重重分配给下列 data-vs-data 指标，**重算真实自评基线**（不可与旧 ~52.47 直接比较）。
3. **Real-vs-Real 零假设带**：`calibrate_real_vs_real()` 把真实窗按**时间连续**两半（非随机，stride=5 重叠会泄漏）+ block-permutation 跑全套指标得 null (mean,std)；报告 candidate 的 `z=(score-null_mean)/null_std`，落在中心 90% 带外即 flag。同时修掉 line 714 `wasserstein=100` 的作弊自评。

### 3.2 核心取证检验（P1–P2）
- **【P1】Signature-MMD 两样本检验** — `eval/signature.py` detector mode：time-aug+lead-lag → 子窗签名 → permutation Sig-MMD（real vs candidate），返回统计量 + 经验 p-value（**先在 real-vs-real split 上验证标定**，arXiv:2506.01718 警示标定难）。非自指、依赖感知，是脊梁鉴别器。
- **【P1】feature-space C2ST**（Lopez-Paz & Oquab, ICLR 2017）— `eval/c2st.py`：**在 ~28 维 stylized-fact 特征向量上分类**（skew/kurt/|r|-ACF lags/2nd-diff energy/vol run-length/tail），**不要碰 raw 4096 维**（独立样本仅 2–3 个，会过拟合到 100% 饱和、无法区分 v9 与 v10）。**时间 block split + 非重叠窗 (stride≥2048) 或 block-bootstrap**；小型正则化 logistic/2-layer MLP（固定 seed）；报告 held-out accuracy + **permutation p-value (≥1000 shuffle)** + AUC + per-window 排序。**多生成器负样本池**（DiT + GARCH 模拟 + IID-shuffle + block-bootstrap）以学"非真实"而非"非 v9"。
- **【P2】learned deep-kernel MMD**（Liu et al., ICML 2020）— `eval/mmd_deepkernel.py`：小 1D-CNN `φ` + kernel `((1-ε)k_φ+ε)k_RBF`，按 power criterion `t=MMD^2/sqrt(σ_H1)` 训练；permutation detector + KID readout。修复 memory 中被移除的裸 RBF-MMD（裸 MMD 在 4096 维不稳，故 score.py v2 已删；正解是 learned-kernel + feature-space）。
- **【P2】Sinkhorn joint-Wasserstein** — 现 `calculate_1d_wasserstein` (score.py:672) **名为 joint 实为 flatten 后排序，是 bug**。`eval/metrics.py` 加 `calculate_sinkhorn_joint`（~30 行 log-domain debiased Sinkhorn，避免 geomloss/KeOps 新依赖），把每条路径 reshape 成 `(T,2)` per-day 点云做 OT。**但其时序置换不变，对自相关/聚集盲视**——故真正取证力来自对 delay-embedded 或窗级统计向量做 Sinkhorn (Feydy et al., AISTATS 2019)。
- **【P2】Energy distance / Energy Score**（Gneiting & Raftery, JASA 2007）— `eval/metrics.py` 加 `energy_distance`（用现 `compute_pairwise_sq_dist`，**记得取 sqrt**，能量核需 ‖·‖ 非 ‖·‖²）。**作用在标准化 stylized 特征向量上**（raw-path 被 level/drift 主导）。严格 proper、bandwidth-free，与 RBF-MMD 互补。

### 3.3 金融取证轴（P2–P3，抗规避补充）
- **GARCH/EVT filtered-residual 复合检验** — `eval/score.py` 每条 candidate **重新** fit GJR-GARCH(1,1)（DGS 用对称 GARCH），标准化残差，`forensic_residual` = `(alpha, persistence α+β, GJR γ, GPD 尾 ξ, LB-Q of z²)` 到真实窗参数**分布**的 Mahalanobis 距离 (McNeil-Frey 2000)。过平滑 v9 因 α/persistence/尾低而失败。**LB 用 Q 统计量不用 p-value**（2048 点饱和到 ~0）。加 `arch` 包到 `ts_diffusion`，try/except 映射不收敛为 fail-low。
- **regime transition 检验** — 真实数据 fit 2 态 Gaussian HMM（**在 returns 上、非 rolling-vol**，否则违反条件独立），冻结 emission，decode real 与 candidate，batch 聚合 Laplace-smoothed 2×2 转移矩阵，symmetric-KL + mean-dwell-time。直接抓 sticky-regime (Hamilton 1989; Ang-Bekaert 2002)。
- **EVT 尾形 / Hill / 下尾相依** — `eval/tail_dependence.py`：pooled 真实超越拟合 GPD 一次得 `(ξ_real, VaR99, ES975)`，candidate 用**经验 VaR/ES 距离**评分（无需逐路径 GPD）；Hill `α=1/ξ`（目标 ~3–4）；rank-based Kendall-τ + 非参下尾 `λ_L`（q=10%，pooled，**非参数 t-copula fit** 避免不收敛，Schmidt-Stadtmüller 2006）。替换脆弱的 `tail_corr`。
- **spectral fingerprint** — `eval/fingerprint.py`：`[log-PSD bins, patch_spike_excess@8/16/32, hf_power_ratio, d2_energy, |r|-ACF slope]` 训 GBM 分类器。**必须报 leave-one-generator-out 准确率**（in-distribution 无意义）。patch-16 spike 是 patchify k=s=16 架构指纹——若我方 fingerprint 大，把 `unpatchify` 从非重叠 Linear 改 overlapping transposed-conv 以降低我方可检测性 (Frank et al. 2020; Corvi et al. 2023)。
- **OOD 重建集成** — `eval/ood.py`：(a) 多噪声层 diffusion 重建误差**向量**（t∈{50..950}），对真实云做 Mahalanobis 两样本 OOD（非单 t 自指，Graham et al. CVPRW 2023）；(b) Deep-SVDD 1D-conv-AE 仅训真实 (Ruff et al. 2018)。

### 3.4 集成与标定（P3）
`eval/detector_ensemble.py` 汇聚 c2st/signature-MMD/fingerprint/energy/GARCH-residual，temperature scaling 校准 `p(fake)`，平均/投票出 verdict。**conformal 覆盖保证在日频独立样本极少时很弱**——优先 block-permutation null，老实标注。**鉴别器并入 `score.py` 前必须证明能 flag 训练中从未见过的 held-out 生成器家族。**

---

## 4. 同时增益两者的金融方法 (TABLE)

| 方法 | 服务 | 如何帮生成 (loss) | 如何帮打分 (forensic) | 工作量 | 参考 |
|---|---|---|---|---|---|
| **Path-signature MMD / sig-kernel score** | **both ★脊梁** | `losses.py` 对 x0_hat 子窗 sig-MMD aux，严格 proper、抓时序+跨通道动态 | permutation sig-MMD 两样本检验，非自指依赖感知 | 中 | Chevyrev-Oberhauser JMLR 2022; Salvi et al. SIAM JMDS 2021; Issa et al. NeurIPS 2023 |
| **Sig-Wasserstein critic** | **both** | Conditional Sig-W1（**autoregressive 过去窗→未来期望签名**，非 first-timepoint）作解析 critic，免对抗网络 | `Sig-W1` 期望签名距离 = 条件律失配检验 | 中 | Ni et al. ICAIF 2021; Liao et al. Math. Finance 2024 |
| **Sinkhorn / entropic-OT** | both | x0_hat batch 上 Sinkhorn 项加跨资产相关压力 | 修复 score.py:672 假 joint-W bug；联合分布散度（**时序不变，须 delay-embed**） | 低 | Genevay et al. AISTATS 2018; Feydy et al. AISTATS 2019 |
| **Energy Score / distance** | both | 小权重严格 proper 分布损失（非主目标，验证/选模为主） | bandwidth-free 严格 proper 两样本 + permutation p | 低 | Gneiting-Raftery JASA 2007; Székely-Rizzo 2013 |
| **Deep-kernel MMD (MMD-GAN)** | both | 冻结 φ 的非对抗 MMD moment-matching aux | power-optimized learned-kernel detector + KID | 中 | Li et al. NeurIPS 2017; Liu et al. ICML 2020 |
| **GARCH/GJR-EGARCH** | both | SP500 通道杠杆不对称 aux（小权重） | filtered-residual Mahalanobis（α/persistence/ξ/LB-Q） | 中 | Bollerslev 1986; GJR 1993; Nelson 1991 |
| **Rough vol (Hurst-of-logvol)** | both | structure-function 匹配 `H_hat→H_real` aux | log-vol Hurst 取证签名（过平滑必失败） | 中 | Gatheral-Jaisson-Rosenbaum 2018 |
| **EVT (POT/GPD)** | both | soft-quantile 尾匹配 aux（SP only） | GPD ξ + VaR99/ES975 距离 + QQ | 中 | McNeil-Frey 2000; EKM 1997 |
| **Copula / tail-dependence** | both (scorer 主) | soft 联合下尾超越占比 aux | rank Kendall-τ + 非参 λ_L | 中 | Sklar 1959; Embrechts-McNeil-Straumann 2002 |
| **HMM regime-switching** | both | per-patch regime 序列条件 (CoFinDiff) | 真实 HMM 转移矩阵 KL + dwell-time | 中 | Hamilton 1989; Ang-Bekaert 2002; CoFinDiff 2025 |
| **FIGARCH 长记忆 (|r|-ACF log-log slope)** | both (scorer 主) | 匹配 |r|-ACF lag 1–64 衰减斜率（**先验 gap 符号**） | DFA/GPH Hurst-of-|r| 距离 | 中 | Baillie-Bollerslev-Mikkelsen 1996; Ding-Granger-Engle 1993 |
| **Spectral/PSD + wavelet** | both | PSD-of-|r| L1 aux 补全 roughness/burst | PSD slope + patch-spike fingerprint | 中 | Diffusion-TS, Yuan-Qiao ICLR 2024; Cont 2001 |
| **WGAN-GP / Diffusion-GAN critic** | both | x0_hat 上小权重对抗微调（EMA-frozen critic, R1） | 独立 critic 训于多源 fake 池作 detector | 高 | Gulrajani et al. 2017; Wiese et al. QuantGAN 2020; Sauer ADD 2024 |
| **EBM / Discriminator Guidance** | both | 采样时 energy-gradient guidance（**无重训**） | log-ratio logit 连续 fakeness 分 | 高(训)/低(采样) | Che et al. 2020; Kim et al. ICML 2023 |
| **C2ST + TSTR/TRTR** | scorer(+弱gen信号) | TSTR/TRTR gap 作监控标量 | feature-space C2ST held-out acc + p-value | 中 | Lopez-Paz-Oquab ICLR 2017; Yoon et al. TimeGAN NeurIPS 2019 |

---

## 5. 评估体系重设计 (Eval Redesign)

把 `eval/score.py` 从"单一加权 fidelity 总分"重构为**双层报告**：

**A. 修复与重标定（P1，无重训）**
- `ddpm_mse` 0.20 → 0.05，标注 self-referential-diagnostic；释放权重重分配（见下）。
- 修 `calculate_1d_wasserstein` 假 joint bug → Sinkhorn joint-W；修 line 714 自评作弊。
- 加 `calibrate_real_vs_real()` 零假设带，所有 candidate 报 z-score 与 90% 带 flag。

**B. 新增 finance-grade 指标（P1–P2）**
- **严格 proper 分布分**：`eval/proper_scores.py` — energy_score（**per-timestep 2-vector / per-channel 边际**，非 flatten 4096 维）；variogram score `p=0.5`，短滞后权 `w_ij=1/|i-j|` for `|i-j|≤50`（针对 lag-1 ACF 偏正）+ 小跨通道块（**勿算全 ~2M 对**，Scheuerer-Hamill 2015）；per-channel CRPS。**variogram 偏置盲，须与边际矩并存。**
- **取证两样本**：sig-MMD / deep-kernel-MMD / energy-distance + permutation p-value（§3.2）。
- **判别/预测效用 (TSTR)**：feature-space C2ST acc+AUC+p；**predictive score 重定目标为 `|r|`/`r²`（波动率，可预测）与 DGS10 yield level（持久），不预测带符号日 return（近随机游走，TSTR/TRTR floor 无区分力）**，`100*exp(-mae/mae_TRTR_oracle)`，多 seed 平均 (Yoon et al. 2019)。
- **风险回测 (downstream)**：`eval/downstream_risk.py` — Kupiec POF（主）+ Acerbi-Székely ES Z2（直击欠离散），把生成尾分位作 VaR/ES、在 held-out **真实** return 上数突破（TSTR-for-risk）。**Christoffersen 独立性/CC 降级为 diagnostic**（近无条件生成器无逐日条件 VaR）。
- **stylized-fact 距离**：`score.py` import `diagnostics.py` 的 roughness/regime/psd block，经 `_score_exponential` 对**新鲜重算的真实基线**评分；新维度 `stylized_facts`（leverage / agg-gaussianity kurtosis-decay / vol-burst / switch-rate / ret-acf1 / Hill）。**声明：此为生成质量 + 共享词汇，非鲁棒取证器（可被针对性匹配规避）。**

**建议重标定权重（示意，需重跑真实自评）**：`ddpm_mse 0.05, sig_mmd 0.12, energy 0.08, variogram 0.08, c2st/disc 0.10, sp/dg skew+kurt 0.15, sp/dg acf 0.12, sinkhorn_joint 0.10, tail_dependence 0.10, risk_backtest 0.05, stylized_facts 0.15`，归一化到 1。

**C. 双 headline**：fidelity TOTAL（生成选模用）与 forensic detector AUC + 校准阈值（goal 2 用），**二者分离，互不污染**。

---

## 6. 落地优先级与下一步实验 (Roadmap)

| Phase | 项 | 类别 | 重训? | 成本 | 风险 | 文件 |
|---|---|---|---|---|---|---|
| **P1** | 修 `ddpm_mse` 自指 (降权+标注) + real-vs-real null band + 修 Sinkhorn joint-W bug | scorer | 否 | 低 | 低 | `eval/score.py`, `eval/metrics.py` |
| **P1** | feature-space C2ST detector (block-split, 多源 fake 池, perm p) | **both** | 否 | 中 | 中(数据稀缺) | `eval/c2st.py` |
| **P1** | **Signature-MMD detector mode** (脊梁, real-vs-real 标定) | **both** | 否 | 中 | 中(标定) | `eval/signature.py`, `score.py` |
| **P1** | EDM churn 采样 + Discriminator Guidance（采样侧抗平滑） | gen | 否 | 低 | 低 | `scheduler.py`, `generate.py`, `config.py` |
| **P2** | **Signature aux loss** 进 `losses.py`（x0_hat 子窗, abar 门控） | **both** | **是** | 中 | 中(与stylized重叠→消融) | `losses.py`, `train.py`, `config.py` |
| **P2** | spectral PSD-of-|r| + rough-vol Hurst aux | **both** | **是** | 中 | 低 | `losses.py`, `config.py` |
| **P2** | v-pred + cosine-ztSNR + min-SNR(v 形式) + round-trip 单测 | gen | **是** | 中 | 中 | `scheduler.py`, `train.py`, `losses.py` |
| **P2** | GARCH filtered-residual + HMM regime + EVT/Hill/copula 取证指标 | scorer | 否 | 中 | 低-中(加`arch`包) | `eval/score.py`, `eval/tail_dependence.py`, `eval/ood.py` |
| **P2** | proper-scores (energy/variogram/CRPS) + TSTR/predictive + VaR/ES 回测 | scorer | 否 | 中 | 低 | `eval/proper_scores.py`, `eval/predictive.py`, `eval/downstream_risk.py` |
| **P2** | regime/realized-vol per-token 条件 (CoFinDiff) | **both** | **是** | 中-高(架构) | 中(adaLN→cross-attn) | `dataset.py`, `dit1d.py`, `generate.py` |
| **P3** | deep-kernel MMD + spectral fingerprint(LOGO) + detector_ensemble + 标定 | scorer | 否 | 中 | 中 | `eval/mmd_deepkernel.py`, `eval/fingerprint.py`, `eval/detector_ensemble.py` |
| **P3** | EDM 完整预条件 / Flow-Matching / Consistency 蒸馏（生成 only 消融） | gen | **是** | 高 | 中-高 | `edm_scheduler.py`, `flow_matching.py`, `consistency.py` |
| **P3** | WGAN-GP / Diffusion-GAN 对抗微调 + EBM detector（gate 在 scorer 验证后） | **both** | **是** | 高 | 高(不稳) | `dit1d_critic.py`, `train.py`, `eval/energy_detector.py` |

### 第一个该跑的实验（本 repo，零重训）
**在 `eval/c2st.py` 与 `eval/signature.py` detector mode 上，对已存在的 `v9_score.json`/`v10_score.json`/`v10_retrained_score.json` 对应的生成样本 + 真实窗，跑 feature-space C2ST 与 sig-MMD permutation test。** 验证两点：(1) real-vs-real 落在 acc≈0.5 / p≈0.5（标定正确）；(2) **更真实的 Phase-2/v10 取证得分应优于 v9**——这直接证伪 `ddpm_mse` 悖论、并交付 goal (2) 的最小可用鉴别器。成功后再启动 P2 的 signature aux loss 重训。

> 优先级原则：**both 类（尤其 signature 脊梁）排在纯生成/纯打分之前**；所有 realism 验收用 `diagnostics.py` + 非自指指标，绝不让 `ddpm_mse` TOTAL 把关重训接受。
