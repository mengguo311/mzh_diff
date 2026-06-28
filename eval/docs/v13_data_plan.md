# v13 数据"治本"方案 —— 在 ~7 个独立宏观窗的硬约束下生成【新颖且真实】的金融序列

> 由 11-agent 数据策略设计工作流产出(5 视角提案 × 对抗批判 + 高强度综合,2026-06-21)。
> 代码行号/事实经子智能体回查仓库核实(`num_patches=seq_len//16` 自适应、`c=window[0]`、L=2048 复制地板 p95=0.0%、`ConditionEmbedding` cond_dim 可改)。
> 判定一切用诚实闸门:`eval/memorization.py`(复制率)+ `eval/novelty_rerank.py`(C2ST_新颖/SigP_新颖)+ `eval/diagnostics.py`,**绝不用 ddpm_mse**。

## 1. 定调

记忆化↔过平滑的 Pareto 张力已被三组实验钉死:根因不是损失/架构/采样/正则任何一项,而是 **`14734/2048 ≈ 7.2` 个非重叠独立宏观窗 << 32–58M 参数**。任何只动模型侧的改动都只在同一堵墙上挪 Pareto,不打穿它。**唯一能把"独立训练信号"的分母从 7 抬高的杠杆在数据侧**——而数据侧只有两条不失真目标分布的真路:**(a) 缩短建模窗(同一份真实数据切出更多非重叠段,零新数据/零合成/零增广失真)**,和 **(b) 引入真·正交的新独立序列(多资产同构对 / surrogate 热身先验)**。其余(signflip / phase-rand / mixup / GARCH 流形 / 评分空间 reweight)要么不增独立信号,要么失真目标尾部,要么是评估作弊,一律不纳入主力。

## 2. 方案分层

> 纳入门槛:对抗批判 `verdict ∈ {strong, plausible}` **且** `adds_independent_signal=true`。命中 4 个:缩窗 L=512、block bootstrap、多资产同构对预训练、context-conditioning。

### A 立即可做(只改 dataset.py / config.py,不拿新数据)

#### A1 — 缩短建模窗 L=2048→512(**首选主力**,×4 独立段)
- **机制**:`14734/512 = 28.8` 个非重叠独立段 vs `7.2`。可背诵的真实目标从 7 涨到 ~29 → 记忆化机制性下降(非 artifact)。**全集唯一一行 config 就改根因的杠杆。**
- **代码改动**:`config.py` `SEQ_LEN=512`、`STRIDE=2`(维持训练样本量万级);`dit1d.py` `DiT1D(seq_len=512)`(`num_patches=512//16=32` 自适应,sincos PE 自适应,**patchify 逻辑无需改**);`dataset.py` **结构无需改**(`__getitem__` 用 `self.seq_len`,`c=window[0]` 不变)。
- **保**:厚尾/峰度、`|r|`-ACF 到 lag~5、逐窗 SP500-DGS10 相关、杠杆(短滞后)——512d(~2 年)全装得下。
- **唯一硬伤**:>512d 的长程 `|r|`-ACF 慢衰、regime `mean_run_len`(真实牛熊跨多年)装不进单窗。
- **验证(杜绝隐性作弊)**:三闸门**必须在 L=512 原生重算 real-vs-real 标定**(短窗更自相似,copy 地板/`c2st_cal` 都上移,**绝不能拿 L=2048 的 0.0% 地板/0.495 标定去判 L=512**);**`diagnostics` 的 regime `mean_run_len`/长滞后 ACF 强制在【拼接回 2048 全长】上跑、与 L=2048 基线同尺度比**,防"512 内读数漂亮、2048 长程已塌"的盲区。

#### A2 — Block Bootstrap 增广(对抗批判里唯一 `strong`;叠在 A1 上)
- **机制**:对真实标准化序列做**前向时序、块长 192、拼接长 ≤512** 的 moving-block bootstrap(整块搬运)。块内保住杠杆(-0.11)、偏度(-0.6)、峰度(~15)、`|r|`-ACF(~0.23),只打乱块间宏观次序 → 制造**新宏观次序排列**而不失真块内 stylized fact,不靠任何凸组合/平均(后者必削尾)。
- **代码改动**(`dataset.py` 新增 + config 开关 `USE_BLOCK_BOOTSTRAP=False` 默认关):`__init__` 末基于已标准化 `self.data` 以 `BLOCK_LEN=192` 切块、有放回拼接成 `self.boot_windows`,`__getitem__` 按比例(30% boot / 70% 真实)采样。**块整搬,不插值/不 mixup/不加噪**;`c=boot_window[0]`。
- **验证**:bootstrap 窗喂 `memorization.py` 当 fake,复制率应 ≈ L=512 真实地板(证它是"新序列");`diagnostics` 峰度/杠杆/`|r|`-ACF(lag1-5)相对纯 A1 退化 <5%。

### B 需要拿数据(多资产 / surrogate)

#### B1 — 多资产【同构对】预训练 + SP500/DGS10 微调(真·正交独立信号)
- **数据**:严格只取"股指 ↔ 本国 10Y 国债"同构对(DAX↔Bund10Y、FTSE↔Gilt10Y、Nikkei↔JGB10Y、TSX↔CAN10Y);**商品/汇率排除**。
- **防失真四道硬门**:① **每对独立 per-pair z-score**(不可全局 scaler);② 入选门槛:候选对 `c2st.featurize` 特征须落 SP500-DGS10 的 p5–p95,且与 SP500 全样本相关+尾部相关 `tc` 双低于阈值(防相关近重复);③ **危机去重**:全球同步暴跌日(多指数同日 <-3σ)降权(防 2008/2020 灌水厚尾);④ **两阶段**:多资产预训练学"律"→ SP500/DGS10 微调,`corrcoef`/`tail-corr` 发布级一票否决,`SigP_novel` 塌设红线。
- **泄露专测**:训完用 `memorization.py` 对**原 SP500-DGS10 bank** 单测复制率,须未因引资产上升(防借道 DAX 复制 SP500);新数据**绝不写入** `train_sp500_us10y.csv`,eval bank 仍只用原 CSV。
- **定位**:除缩窗外唯一不靠合成的真独立信号,但增益 log 级(危机同步折损),**成本/收益不如 A1**,列为 A1 之后的乘性叠加。

#### B2 — GJR-GARCH-t surrogate 预训练做热身先验(仅热身,不当独立真值)
- GJR-GARCH(1,1)-skew-t 参数族池生成 surrogate 预训练;**只当热身,早停由 `C2ST_novel` 决定**;上池前每条过 `featurize` 体检(落 real p5–p95);加无-surrogate 对照臂。进阶:FIGARCH(幂律长记忆)、DCC-GARCH(动态相关)。**替补 B1。**

### C 训练侧配合(只在 A/B 之上做正则)

#### C1 — Context-conditioning 自回归(A1 的长程修补)
- **机制**:条件从 `c=window[0]`(2 维)升级为 `c_ctx`=前置上下文窗富统计向量(波动率/regime 占比 ~16 维);生成时自回归拼接回 2048,用 `c_ctx` 链式重建长程。
- **代码**:`dataset.py __getitem__` 额外算前 L 天统计做 `c`;`dit1d.py` `ConditionEmbedding(cond_dim=2)→16`(加性融合 `cond_emb=t_emb+c_emb`,改 cond_dim 即可)。
- **必做消融(防 Sig-MMD 式 no-op)**:`novelty_rerank` 比 `zero-ctx` vs `real-ctx` 的 regime 分布——**无差异 → cond 被忽略、退化为短窗 i.i.d. 拼接、长程必塌,立即停。**
- **接缝**:训练喂跨接缝 512 窗;平滑只用模型自洽重叠取均值,**严禁真实数据填缝**(=变相记忆);`diagnostics` PSD/d2-energy 把关。

## 3. 有序实验序列(每步带诚实达标判据)

> 锚点:复制率基线 **36.5%**(DiT-S 最低记忆点)、C2ST_novel 当前最优 ~0.755、SigP_novel(越高越真)、diagnostics 不退化。**所有判据在该步自身 L 上、对该 L 重标定的 real-vs-real 地板比较。**

| 步 | 实验 | 时长 | 诚实达标判据(全满足才推进) |
|---|---|---|---|
| **0** | **重标定 L=512 闸门** | <1d | L=512 重算 `memorization` 非重叠 real 地板、`novelty_rerank` `c2st_cal`/`sig_p_cal`、`diagnostics` 基线。**不做这步后面全是隐性作弊。** |
| **1** | **A1 缩窗 L=512** 验证训练 | 12h | ① 复制率(L=512 标尺)显著 < 36.5%;② `C2ST_novel < 0.755`;③ `SigP_novel` 不塌(> L=512 标定下沿 且 ≥ v11 的 0.090);④ `diagnostics` 在**拼接后 2048 全长** regime `mean_run_len` gap 未扩大、roughness/峰度退化 <5%。**④ 失败=长程塌,即使 ①②③ 漂亮也判负。** |
| **2** | **A1 + A2 block bootstrap**(30%) | 12h | 步1基础上:复制率进一步↓或持平;`C2ST_novel`↓;块内 stylized(杠杆/峰度/`|r|`-ACF lag1-5)退化 <5%;boot 窗当 fake 复制率 ≈ 地板。 |
| **3a** | **C1 context-cond** 消融门 | 12h | `zero-ctx` vs `real-ctx` regime 分布**须显著不同**;否则停 C1。 |
| **3b** | **B1 多资产预训练+微调**(若步1–2 复制率仍顽固 > 36.5%) | ~24h | 原 bank 复制率未升;`corrcoef`/`tail-corr` 不破(发布级否决);`SigP_novel` 不塌;`C2ST_novel`↓。 |
| **4** | 全量重训最优配置(nohup) | ~24h | 同步1判据,N=5120;**取证 headline(real-vs-novel 的 C2ST_novel/SigP_novel)是最终裁判。** |

## 4. 反共识警示(点名)

**(a) 造"相关近重复"、降不了记忆化:**
- **多资产近因子扩容(DJIA/Wilshire/Russell/DGS7/DGS20)** — >0.9 相关、共享同组 2008/2020 实现,`7→~25` 撼不动 `58M/7`。降级为 1 天 sanity-check。
- **multi-resolution 的 256 臂** — 256d 是 512d 同一 58 年的严格子段,"joint 57 窗"是双重计数会计幻觉;且 256d 装不下 regime 周期、`tc` 跌破 `m.sum()>3` 守卫致杠杆/尾相关被低估。至多低权重 random-crop 正则。
- **time-reversal / sign-flip** — 不增独立信号;杀杠杆不对称(leverage -0.116→+0.0145);`corr=-1`/反转能绕过 signed Pearson 0.95 阈值 = 与逐点复制同款作弊。**禁用。**

**(b) 失真目标分布(尤其尾部):**
- **Latent Mixup / 值域插值** — 重尾窗凸组合数学上必向正态收敛、系统削峰减尾,而厚尾/峰度是核心目标;会使 `C2ST_novel` 不降反升。图像域有效不可迁移。
- **Phase randomization** — 摧毁 `|r|`-ACF+峰度+偏度,只能当鉴别器负样本,无训练信号。
- **GARCH 当独立真值** — 几何 ACF 偏离真实幂律长记忆、static Cholesky 装常数相关、双重峰度过冲。只配热身先验。

**(c) 评估作弊(最致命,红线):**
- **在 `c2st.featurize` 评分空间 reweight / 从 held-out 真值抽 condition** — 把训练分布往评分器/考题对齐 = Goodhart 泄露,`C2ST_novel` 下降无法区分"学会生成"还是"被对齐到考题"。**绝不复用 gate 的 featurize 做训练侧 reweight;终裁须用盲测(holdout 后 20%)特征。**

## 5. 一句话:若只做一个数据实验

**做 A1**:`config.SEQ_LEN=512`、`STRIDE=2`、`DiT1D(seq_len=512)`(`num_patches`/patchify/`dataset.py` 逻辑均无需改),跑 12h 验证训练。它是全集唯一一行配置就动根因(独立段 7.2→28.8,×4)、零新数据/零合成/零失真增广的杠杆;**验收三连**:先在 L=512 重标定三闸门地板,再要求复制率(L=512 标尺)< 36.5% 且 `C2ST_novel < 0.755`、`SigP_novel` 不塌,**最关键把 `diagnostics` 的 regime `mean_run_len`/长滞后 ACF 强制在【拼接回 2048 全长】上跑、与 L=2048 基线同尺度比**——长程退化即判负,即便短尺度漂亮也不放行。

---

## 📦 备件清单(已提前备好,翻开关即用 — 2026-06-22)

> 全部默认关 / 互不干扰 / 不影响在跑的 A1。Step 2/3a 执行时直接用。

**Step 1 A1(进行中)**:`config SEQ_LEN=512/STRIDE=2`(已生效);评估 `eval/auto_eval_v13_a1.sh`(轮询中)。
- L=512 标定标准:`eval/calib_{mem,rerank}_L512.json`(复制地板 0.0% / C2ST_cal 0.496 / **SigP_cal 0.116**)。

**Step 2 A2 block-bootstrap** — 代码 ready,默认关:
- 启用:`config.USE_BLOCK_BOOTSTRAP=True`(`BLOCK_LEN=192`/`BOOT_FRAC=0.3`),沿用 A1 配方重训 `deep_v13_a2_boot`。
- 评估:`nohup bash eval/auto_eval_v13_poll.sh deep_v13_a2_boot <gpu> > logs/ae_v13_a2.log 2>&1 &`(通用轮询器 → `eval/eval_v13_run.sh`:L=512 诚实闸门 + concat2048 长程基线)。
- 已验证:boot 窗与真实窗 max|pearson| 0.13–0.17 ≪ 0.95(非复制)。

**Step 3a C1 富条件** — 代码 ready,默认关:
- 启用:`config.USE_CONTEXT_COND=True`(自动 `COND_DIM=16`),重训 `deep_v13_c1_ctx`(DiT-S@512,cond_dim=16)。
- 自回归采样器:`generate_autoregressive.py`(链式 512→2048,`ctx_features` 与训练同口径,已结构冒烟)。
- 评估+消融:`nohup bash eval/auto_eval_v13_c1.sh deep_v13_c1_ctx <gpu> ...`:
  自回归生成 real-ctx 与 zero-ctx(`--force_null`)→ `eval/ctx_ablation.py` 判 context 是否被用上(regime KS;**无差异即停 C1**)→ 长程闸门@2048(对标 L=2048 标定 + v10_retrained)。
- 注:消融未过则不上自回归长程,直接退回 A2 配置。

**通用评估工具**(已就绪):`eval/{memorization,novelty_rerank,diagnostics,concat_windows,ctx_ablation}.py` + `eval/eval_v13_run.sh`(per-512 通用)。`memorization`/`novelty_rerank` 支持 `--L` 标定。

**Step 3b B1 多资产 / B2 GARCH** — 数据/包依赖,未编码(需先备料):
- B1:需获取股指↔本国10Y国债同构对(DAX↔Bund10Y 等)日频数据,**放独立文件不入主 CSV**;落地时按 §2-B1 四道硬门(per-pair z-score / featurize 筛选 / 危机去重 / 两阶段预训练-微调)写 `dataset` 多源加载 + 预训练脚本。
- B2:`arch` 包**未安装**(`pip install arch`),需先装;再写 GJR-GARCH-t surrogate 生成器做热身先验(早停由 C2ST_novel 决定)。**仅 B1 缺数据时的替补。**

## 相关文件(绝对路径)
- `config.py`(改 `SEQ_LEN`/`STRIDE`,新增 `USE_BLOCK_BOOTSTRAP`/`BLOCK_LEN`)
- `dataset.py`(A2 block bootstrap;C1 `c_ctx`;`__getitem__:209-213` `c=window[0]` 是改 condition 锚点)
- `dit1d.py`(`seq_len`/`cond_dim` 入参;`num_patches=seq_len//patch_size` 自适应)
- `eval/memorization.py`(L=512 重标定 real 地板;当前 L=2048 地板 p95=0.0%)
- `eval/novelty_rerank.py`(L=512 重算 `c2st_cal`=0.495/`sig_p_cal`=0.326)
- `eval/diagnostics.py`(regime/roughness 必须在拼接后 2048 全长跑)
