# THIRDPARTY_CLAUDE.md —— 队友/外部代码同步说明

> 给 Claude Code CLI 的 `~/src/thirdparty/` 工作流同步文档。这里放**他人开发、本项目复用**的代码
> (与 `~/src` 自研代码区分)。每次涉及 thirdparty 时先读本文件。最后更新:**2026-06-24**。

## 目录清单

| 子目录 | 来源 | 作用 |
|---|---|---|
| `hw01_discriminator_share_20260624/` | 队友(HW01 团队) share 包 | **v5 取证判别器**:判一条 2 通道金融序列是真实还是生成 |

---

## hw01_discriminator_share_20260624 —— 队友的 v5 判别器

### 它是什么
"风格事实引导的对比判别器"(Stylized-Fact Guided Contrastive Discriminator),用于 HW01
"mixed-mask" 作业:输入文件含 `mask{1..5}_sp500` / `mask{1..5}_DGS10` 列,判每条 mask 是真/假。

打分链路(`discriminator/phase2_tcn_infonce_discriminator_v5.py`):
```
风格事实特征 + DGS10 指纹 + TCN(InfoNCE 对比)多尺度嵌入
  → LogReg/RandomForest/ExtraTrees 校准集成 → p_real_classifier
combined_real_score = 0.75 * p_real_classifier + 0.25 * sf_probability
```
原版自检极强:test_AUC ≈ 0.998(`artifacts/training_metrics.json`)。

### ⚠️ 关键事实(复用前务必记住)
1. **运行环境**:README 写的 `conda activate mizuho` **本机不存在**。本机用 **`ts_diffusion`**
   env(已含 torch/sklearn/scipy/pandas/matplotlib,torch 2.5.1+cu124 CUDA 可用)。
2. **数据口径 = 收益/差分,与本项目一致**:判别器内部把 `sp500` 当**日收益率**
   (`level=(1+sp).cumprod()`)、`DGS10` 当**日收益差分**(`zero_diff_ratio` / 0.01 取整指纹)。
   → 本项目 `data/train_sp500_us10y.csv`(sp std≈0.01,dgs std≈0.06)**可直接喂**,
   无需做价格水平↔收益的口径转换。**价格水平(level)口径的数据不兼容**(std>1 会被判异常)。
3. **窗长 `N=1260`**:对长序列内部自动 `.tail(1260)`;短于 1260 的(如 L=512)按其原生长度算特征。
4. **DGS10 指纹**:真实 DGS10 差分量化到 0.01(`rounding_ratio_0.01` 高);生成数据通常连续
   (该比例≈0)→ 这是判别器识别生成数据的一个合法信号。
5. **share 包不含 sklearn 分类器 pickle**(只存了 `tcn_infonce_encoder.pt`)→ **复用必须重训**
   分类器。好在 seed 固定(`20260525` + 各模型 `random_state`),重训可复现(实测 test_AUC=0.9998)。
6. **依赖原版默认路径**(`/home/u00111/...`、`/share/mizuho/...`)在本机不存在 → 用本项目路径覆盖。

### 原版怎么跑(仅作参考,本机缺 `mizuho` env + 缺 mixed_dir)
```bash
bash hw01_discriminator_share_20260624/scoring/run_v5_scoring.sh   # 本机不可直接用
python tests/test_v5_share_package.py                              # 冒烟:校验文件/映射一致
```

---

## 集成:用队友判别器给本项目 mguo 生成数据打分

### 驱动脚本
`/home/u00134/data/generation/mguo/discriminate_mguo.py`(本项目自写,import 队友模块)。

**做了什么**:
1. 复刻队友 `main()` 的**训练段**(真实窗 vs 8 类合成负样本 → TCN 编码器 + 风格事实分 +
   校准分类器集成),真实数据用本项目 `data/train_sp500_us10y.csv`、超参同 `run_v5_scoring.sh`
   (220 epoch / temp 0.12 / lr 7e-4 等)。
2. 把 `mguo/*_top1000.csv`(宽表:每行一条序列,列 `sp500_0..` + `dgs10_0..`)每行转成
   `(sp500,DGS10)` frame,走与队友 `analyze_mixed_file` **完全一致**的链路
   (`aggregate_multiscale_features → transform_new_features → ensemble_predict →
   scaled_sf_probability → combined_real_score`)。
3. 按训练阈值判 `real(被骗)/generated(识破)`,写日志到 `mguo/log/`。

### 怎么跑(长任务,≥40min,务必 nohup 后台)
```bash
cd /home/u00134/data/generation/mguo
nohup conda run --no-capture-output -n ts_diffusion python -u discriminate_mguo.py \
  --device cuda > log/run_discriminate.log 2>&1 &
# 冒烟:--limit 4 只评前 4 行;脚本内可临时压 contrastive_epochs 加速
```
耗时:训练~7min(epoch 159 早停)+ 每个 L=2048 文件~190s,12 文件全量约 **46min**。

### 输出日志(均在 `/home/u00134/data/generation/mguo/log/`)
| 文件 | 内容 |
|---|---|
| `discrimination_summary.txt` / `.csv` | 12 模型排名(mean_p_real / mean_combined / fool_rate / catch_rate) |
| `discriminate_<model>.csv` | 逐样本 `p_real_classifier / sf_probability / combined_real_score / judged_real` |
| `discrimination_meta.json` | 判别器自检(test_AUC)、真实数据参考基线、阈值 |
| `run_discriminate.log` | 完整运行日志 |
| `_train_artifacts/` | 重训产物(encoder.pt / 特征表等),可复用免重训 |

### 指标定义
- `fool_rate` = 被判成"真"的比例(p_real ≥ 阈值)→ **越高 = 该生成器越能骗过队友判别器**。
- `catch_rate` = 被识破比例 = 1 − fool_rate。
- 真实参考基线:真实 test 窗 mean_p_real=**0.643**,judged_real=**100%**(判别器对真数据校准良好)。

### 判别结果(2026-06-24,test_AUC=0.9998,阈值 0.507)
| 模型 | mean_p_real | mean_combined | fool_rate | catch |
|---|---|---|---|---|
| deep_v9_20k | 0.659 | 0.619 | 67.9% | 32.1% |
| deep_v9_20k_200step | 0.659 | 0.619 | 67.9% | 32.1% |
| deep_v9 | 0.647 | 0.609 | 66.6% | 33.4% |
| deep_v10 | 0.629 | 0.593 | 63.2% | 36.8% |
| deep_v13_c1_ctx_realctx2048 | 0.527 | 0.497 | 52.4% | 47.6% |
| **l2_clip11(line2 最优)★** | **0.476** | **0.452** | **45.0%** | **55.0%** |
| deep_v12_dits_antimem | 0.413 | 0.412 | 38.7% | 61.3% |
| deep_v13_c1_ctx_zeroctx2048 | 0.334 | 0.338 | 32.0% | 68.0% |
| deep_v11_noSig | 0.363 | 0.364 | 31.6% | 68.4% |
| deep_v13_a1_concat2048 | 0.297 | 0.297 | 29.0% | 71.0% |
| deep_v11_val | 0.302 | 0.316 | 27.2% | 72.8% |
| deep_v10_retrained | 0.297 | 0.299 | 24.3% | 75.7% |
| deep_v13_a1_L512 | 0.124 | 0.175 | 4.3% | 95.7% |

> **★ l2_clip11(line2 最优)补评(2026-06-26)**:对最新模型输出 `output/line2/l2_clip11_final.csv` 前 1000 条评分,**复刻同口径**(test_AUC **0.9998** / 阈值 **0.5062** / 真实基线 mean_p_real **0.6413**、judged_real **100%**,与 6-24 的 0.643/100% 一致 → 与上表 12 模型可比)。结果 **fool 45.0% / mean_combined 0.452**,居 c1_realctx2048(52.4%)与 v12(38.7%)之间。
> **判读(与下述归因一致,非异常)**:clip11 是 line2 **内部最优**(复制率 0.02% / C2ST_新颖 0.62 / 厚尾充足峰度~19,见 `PROJECT_SUMMARY_line2.md`);正因其**真·厚尾**,被 hw01(负类含 GARCH-t/SABR 厚尾合成)推向"合成"侧 → fool 低于过平滑的 v9/v10(67%)、但高于抗记忆缩窗的 a1_L512(4.3%)/v10_retrained(24%)。**再次印证"越真实厚尾越易被该鉴别器识破"**。**hw01 是 line2 的外部参照、非优化目标**(追 fool=过平滑迎合=远离真实)。

### ⚠️ 重要判读 ——「为什么越先进的模型越易被鉴别」(2026-06-24 实证归因 + 对抗验证)

> 用 11-agent workflow 复刻了这个判别器(固定 seed,test_AUC **0.99992**,阈值 0.5108),
> 复现 fool_rate 与上报值**同序**(Spearman **0.809**)、量级吻合 → 下列归因可信。
> 实证脚本与逐模型特征表见 `/home/u00134/data/generation/mguo/log/external_disc_{feature_attribution.py,analyze.py,feature_attribution_result.json}`。

**现象**:外部判别器排名与项目内部"诚实排名"**几乎相反**。内部综合第 1 的 `deep_v13_a1_L512`
在队友判别器前**最易被识破**(fool 4.3%);内部认为"靠记忆撑/欠新颖"的 `v9` 系列/`v10`
反而**最能骗过**(fool 63–68%)。

**根因(经实证修正,⚠️ 与本节旧版结论不同)** —— 头号信号**不是**显式统计矩,**也不是**逐点记忆,
而是 **TCN-embedding 空间里"贴近真实窗质心/近邻"**:
- 跨 12 模型 Spearman(特征均值, fool_rate):`nearest_distance_mean_k5` **−0.839** /
  `mean_sf_prob` **+0.914** / `frac(p_real>0.9)` **+0.904** / `real_minus_synth_distance` **−0.741** /
  `nearest_real_ratio_k5` **+0.734**。显式峰度仅 −0.34(且偏相关下归零);
  **项目复制率 copy_rate vs fool ≈ −0.15(几乎零/弱负 → 逐点记忆与"骗过"解耦)**。
- 队友 RF/ExtraTrees 重要性榜首即 `embedding_nearest_real_ratio_k5/k10`(0.07–0.095,复刻模型上前4 embedding 特征占 ~99.7%);
  其次才是 DGS10 量化指纹。
- **链 1**:v9/v10 过平滑 + 统计像真实窗"平均脸" → embedding 落进真实窗近邻簇(k5≈0.59–0.60,紧贴)→ 判真
  (v9 逐样本 52–54% 的 p_real>0.9,被判真样本里 ~80% 来自此簇)。抗记忆/缩窗/上下文条件的"先进"模型
  主动**远离真实窗** → k5 跌到 0.13–0.21、近邻距升到 ≈0.34(随机量级)→ 判假。偏相关:控制 k5 后峰度偏相关归零(−0.09),控制峰度后 k5 偏相关存活(+0.69)。
- **链 2(OOD 反噬)**:负类含**厚尾合成**(GARCH-t/SABR) → 判别器把"真实厚尾"推向"合成"侧。
  **它并不奖励真实厚尾**:过平滑欠厚尾的 v9/v10(峰度~2.1)最能骗过,更真实厚尾的 v11/v12/v13(峰度更高)更易被识破。
  即"修好 v9 过平滑"这个真实改进,恰让样本更像它的合成负类。(峰度 −0.71 较脆,主由 L512 离群点撑;稳健结论 = 它不奖励厚尾。)
- **链 3(Goodhart/目标错位)**:项目优化自家诚实闸门(新颖+不抄),与判别器头号信号(贴近真实窗)反相关;
  仅 ~7 个独立 SEQ_LEN 段下"既新颖又落在真实簇内"不可兼得。`sig_p_novel` vs fool −0.55。
- **副轴 DGS10 量化指纹**:真实利率差分量化到 0.01(rounding 0.081、8% 零差分);**全部 12 模型一致塌成连续浮点**
  (rounding≈0、unique≈0.9999)→ 确定性"所有扩散都假"硬抓手,但 12 模型一致,**只解释"扩散整体被识破"、不解释模型间排序**。

**两个反常(已定量解决)**:
- `deep_v10_retrained` 复制率最高(53.9%)却 fool 垫底(24.3%):它**抄的是厚尾真实窗**(链 2)+ stylized 损失把 embedding 重心改向合成簇
  (`real_minus_synth`=0.052/到真实原型 0.656 全体最差)→ 被自家厚尾出卖。这正是 copy_rate 与 fool 解耦之源。
- `deep_v13_a1_L512` fool 4.3% 是**窗长伪影**:同底模型自回归拼到 2048(`concat2048`)后 fool **2.4%→34.4%**(中游),
  仅窗长变、生成器没变摆动 14×。512<判别器设计 1260,峰度/近邻/原型距四项全崩。

**结论**:这是"**对抗不同判别器结论不同**"的硬实证(对抗验证判 supported)——两轴正交甚至冲突,
且**无一是地面真值裁判**(连队友判别器都惩罚真实厚尾、奖励过平滑)。队友判别器**不能**替代本项目
`eval/{memorization,novelty_rerank,c2st,signature}` 诚实闸门,而是**互补的外部视角**:内部闸门防记忆化/测新颖,
外部判别器测"能否骗过经典-合成训练的鉴别器"。**若给项目闸门补 DGS10 量化检测,两者相反性可部分缓和。**

---

## 维护约定
- thirdparty 子目录**保持队友原样**,不要改它的源码;集成/驱动脚本写在本项目侧(如 mguo 目录)。
- 新增 thirdparty 包时,在本文件"目录清单"加一行 + 一节说明(来源/环境/口径/怎么跑/坑)。
- 重跑判别只需 `discriminate_mguo.py`;若只换数据不换判别器,可复用 `log/_train_artifacts/` 免重训
  (当前脚本默认每次重训,如需复用需小改 `force_rebuild_cache` / 载入已存 encoder)。
