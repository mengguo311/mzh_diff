# 数据稀缺金融时序生成 —— 模型选型总览与执行摘要

> 总分析师综合报告。汇总 5 族(econometric / fin-neural-gen / diffusion-eff / foundation / sample-eff)的调查 + 对抗评判,落到本项目"~7 独立宏观窗"硬约束下的可执行选型。
> 配套:`candidates_by_family.md`(5 族逐候选详表)、`recommended_experiments.md`(Top 推荐落成可执行实验序列)。

---

## 1. 调研问题

在 **2 通道日频金融时序(SP500 log-return / DGS10 日差分,14734 行)** 的生成任务上,现有 1D DiT-B(~58M 参数)被一条**锁死的 Pareto**卡住:
- **记忆化端**:v10_retrained 复制率 53.9%(近半输出是训练窗逐点复制);
- **过平滑端**:抗记忆配置(缩窗 L512→复制率 14.5%)又把签名/长程/厚尾打塌;
- **诚实闸门 floor**:line2 实证 **C2ST_新颖 ≈ 0.60–0.62**(无单特征 AUC>0.58),被文档化为"模型侧三杠杆(clip/eta/block-bootstrap)耗尽后的**数据稀缺指纹**"。

问题:**有没有比现有 DiT 路线更适合极端数据稀缺的生成模型/范式?** 本报告横扫 5 族 ~50 个候选作答。

## 2. 本项目的硬约束(贯穿全部评判的主线)

> **根因已钉死 = 数据稀缺,而非损失/架构/采样细节。**

- **~7 个独立 SEQ_LEN=2048 宏观窗**(14734 行 / 2048 ≈ 7;重叠滑窗 ~2538 个但**独立信息只有 ~7**)对 58M 参数 → 深陷记忆化区。
- 理论佐证:扩散的**记忆化→泛化过渡所需样本数 ∝ 数据内禀维度线性增长**(Kadkhodaie ICLR2024;On-Memorization 2023);CIFAR 类数据 ≥10k 样本才近零记忆 → **~7 窗无论换什么模型类都在记忆化区**。
- **诚实铁律**:没有任何方法能"凭空"在 7 窗上造出独立信息。能做的只有三件事 ——
  1. **抬分母**:注入真·外部独立信息(更多真实数据 / 跨市场迁移);
  2. **强先验替代数据**:用低维参数律(GARCH/SDE/签名矩)把信号约束到低内禀维子空间,绕开"估 2048 维联合"的瓶颈;
  3. **防记忆 + 抗过平滑**:正则 / 早停 / 矩匹配目标,把锁死的 Pareto 沿"低记忆+保真"方向推得更好(**推动而非消除** floor)。

## 3. 两类候选必须分清

| | **本项目真正需要(数据高效 / 强先验 / 迁移)** | **不适合(数据饥渴大模型 / 范式错配)** |
|---|---|---|
| 机制 | 学低维递归律 / 匹配总体矩 / 注入外部分布 | 估完整高维联合密度 / 对抗判别器 / 大语料预训直迁 |
| 在 7 窗上的命运 | 免疫独立窗约束(GARCH 学 ~5-8 参)或抬分母 | 必塌成记忆化或过平滑(项目已实证) |
| 代表 | FHS / GARCH-t、签名矩匹配、B1 多资产、B2 合成预训、强正则早停、Time-Causal VAE | **QuantGAN、TimeGAN/SDE-GAN、Latent Diffusion、CoFinDiff、CSDI/TimeGrad、Consistency 蒸馏、TimesFM、TimeGPT、Chronos/Moirai/Lag-Llama 整栈直迁** |

> **点名说明**:对抗派 GAN(QuantGAN/TimeGAN)在"重叠窗→少独立段"的同一困境下,相对现有 DiT **无数据效率净增益**;蒸馏会**继承并放大教师记忆**;通用域 TSFM 直迁金融被 DELPHYNE(2025)实证**负迁移、劣于 GARCH 基线**;Latent Diffusion 再叠一个需在 7 窗上训的 AE = 两头不讨好且抹平厚尾。这些列为"研究过的反例"。

---

## 4. 分族候选总表(按 verdict × 数据效率排序)

> 列含义:**数据效率fit**(strong/plausible/weak,对 ~7 窗的契合)、**避记忆化**、**2ch 金融保真**(厚尾/波动聚集/杠杆/跨通道/长程)、**可实现性**(复用现有 DiT/eval/signature 栈、依赖是否已装)。依赖现状:`arch / torchsde / sigkernel / nflows / zuko / pyro / numpyro / statsmodels / signatory / iisignature` **全缺**;`scipy/sklearn/numpy/torch` 在;仓库 `eval/signature.py`(numpy depth-3 + Sig-MMD 置换检验)、`losses.py`(可微 depth-2 sig_mmd + recover_x0)、`memorization.py / c2st.py / novelty_rerank.py / forensic_suite.py` 诚实闸门**已就位**。

### ★ Top-pick(数据效率 strong,优先落地)

| 候选 | 族 | 数据效率fit | 避记忆化 | 2ch 金融保真 | 可实现性 | verdict |
|---|---|---|---|---|---|---|
| **FHS(GJR/EGARCH-t 滤波 + 残差自助)** | econometric | strong | 极高(新 innovation→复制≈0) | 强(残差经验分布原生峰度~19,递归=波动聚集,非对称=杠杆) | 高(纯 numpy ~60 行,arch 缺无碍,复用 eval) | **top-pick** |
| **GARCH×扩散杂交(白化预处理 / 增广)** | econometric | strong | 低且可控(残差近 iid→记忆动机降) | 强且互补(厚尾/聚集外包 GARCH,扩散学残差+跨通道非线性) | 中(dataset 加滤波层 + generate 乘回波动,复用 DiT/eval) | **top-pick** |
| **GARCH-t + 小神经 copula(GARCH-GMMN)** | sample-eff | strong | 内禀零记忆(5 参仿真器) | 强(GJR-t 给厚尾/聚集/杠杆,2D copula 建跨通道创新) | 中(arch 缺,~50-100 行手撸 MLE + 小 copula 网) | **top-pick** |
| **B2 合成 surrogate 先训→真实微调** | foundation | strong | 最强(合成淹没真实 7 窗 + 早停/LoRA) | 强(DCC/GJR-GARCH-t 内建跨通道相关+多 stylized) | 最高(全离线,复用现 DiT + 全套闸门,两阶段脚手架现成) | **top-pick** |
| **B1 多资产同构对跨市场预训** | foundation | strong | 直接降(更多真实独立窗,复制率随独立段升而降) | 强(股↔本国债结构同构,跨市场共享 stylized) | 中(联网取数 + per-market 对齐;FRED 基建半就位) | **top-pick** |
| **期望签名 / Sig-MMD 矩匹配(对 held-out 库)** | fin-neural-gen / sample-eff | strong | 内禀强(分布级目标不奖励逐点抄) | 强(level-2 交叉项原生编码 SP500×DGS10);厚尾须叠边缘层 | 最高(signature.py 已 80% 就位,纯 PyTorch 零新依赖) | **top-pick** |
| **强正则 + 早停(dropout/谱归一/wd/EMA)** | diffusion-eff / sample-eff | strong(防御向) | 直击头号杀手(dit1d dropout=0 未动用余量) | 中性(通道无关,不增 stylized) | 最高(train.py 改 epoch/容量/wd,零新管线) | **top-pick** |

### ◑ Worth-trying(数据效率 strong/plausible,作增强臂或对照)

| 候选 | 族 | 数据效率fit | 避记忆化 | 2ch 金融保真 | 可实现性 | verdict |
|---|---|---|---|---|---|---|
| 非对抗 Neural-SDE + 签名核(Issa-Salvi'23) | fin-neural-gen | strong | 低(连续动力学无法存离散路径) | 强(扩散矩阵 off-diag 给跨通道);薄尾须 jump | 中(torchsde+sigkernel 缺,新生成器) | worth-trying |
| GJR-GARCH-t / EGARCH-t 核心引擎 | econometric | strong | 极低 | 厚尾+聚集+杠杆一次拿下;单通道需配 DCC | 高(被 FHS 包含,作引擎件) | worth-trying |
| Time-Causal VAE(2024 SIAM) | fin-neural-gen / sample-eff | plausible | 低-中(causal-OT + KL 瓶颈) | 多元原生,因果利长程;高斯解码欠厚尾须对冲 | 中(RealNVP 可手搓,闸门复用) | worth-trying |
| DCC-GARCH + t-copula | econometric | plausible | 低 | 直击时变+尾相关;危机段欠识别 | 中(copula 手写) | worth-trying |
| stationary/tapered block bootstrap(残差域) | econometric | plausible | 中(须残差域+短/随机块) | 保聚集/厚尾;原收益域接缝失败(已证) | 极易(config 钩子已有) | worth-trying |
| FIGARCH / component-GARCH / GARCH-MIDAS | econometric | plausible | 低 | 直补长程 |r|-ACF;MIDAS 用 DGS10 耦合 | 中(无新包) | worth-trying |
| DELPHYNE(B2 论据 + 配方捐献) | foundation | strong | 低 | any-variate + Student-t 混合头 | 折叠进 B2(非独立栈) | worth-trying |
| Patch Diffusion(随机子序列) | diffusion-eff | strong | 机制性降记忆 | 通道无关;牺牲长程须 C1 补 | 高(DiT 已 patchify) | worth-trying |
| Diffusion-TS(Fourier 损失 + x0 目标) | diffusion-eff | plausible | 无内禀(须配早停) | Fourier 利 PSD/长程;趋势/季节先验对收益错配 | 中(增量嫁接 losses.py) | worth-trying |
| Self-conditioning(回灌 x̂0) | diffusion-eff | weak | 中性(略增拟合) | 缓解过平滑/欠离散 | 高(~30 行) | worth-trying |
| v-prediction / EDM 预条件 | diffusion-eff | weak | 中性 | 治"过优化 eps-MSE→过平滑",减 clip 依赖 | 中(改 scheduler/losses) | worth-trying |
| Kronos(金融域可生成 FM) | foundation | plausible | 预训正则降记忆 | 金融 stylized 内建;OHLCV≠股债跨通道 | 中-高(HF 独立栈,需 re-tokenize) | worth-trying |
| 抗记忆训练期正则(并入早停包裹) | sample-eff | plausible | 直击记忆 | 中性 | 最高 | worth-trying |

### ▽ Deprioritize / ✗ Reject(数据饥渴 / 冗余 / 错配 / 大改)

| 候选 | 族 | verdict | 一句话原因 |
|---|---|---|---|
| MS-GARCH / HMM-GARCH | econometric | deprioritize | 瞄准真伤口(regime)但用全族最吃数据机制,危机态欠识别撞核心约束 |
| Logsig-RNN | fin-neural-gen | deprioritize | 骨干优化非治本;iisignature/signatory 缺 |
| 条件 NF / GARCH-NF | fin-neural-gen | deprioritize | 攻已被 clip20 解决的尾(冗余);纯 flow 在 7 窗对训练点过拟合 |
| Fourier Flow | fin-neural-gen | deprioritize | 波动聚集=幅相非线性耦合,线性谱流欠;exact-likelihood 小数据过拟合 |
| QuantGAN(TCN-GAN) | fin-neural-gen | deprioritize | stylized 强但对抗 GAN 数据饥渴,落锁死 Pareto;仅作真实度标杆 |
| 贝叶斯 SV / BSTS / GP-SSM | sample-eff | deprioritize | 与 GARCH 高度重叠却需 pyro/numpyro(缺),GARCH 更轻 |
| 数据增广(IAAFT/相位随机化) | sample-eff | deprioritize | 朴素相位随机化**摧毁波动聚集**;安全形态退化为 B2 附属 |
| EVT / 厚尾流 | sample-eff | deprioritize | 只攻已被 clip20 赢下的厚尾轴,边际值低 |
| 跨域预训练 + few-shot | sample-eff | deprioritize | 预训阶段仍数据饥渴 = 等价 B1,非"7 窗内"高效 |
| GBM-SDE 乘性扩散 | diffusion-eff | deprioritize | 纯几何 BM 不产波动聚集/杠杆;厚尾已被 clip20 解;scheduler 手术风险最高 |
| Ambient Diffusion | diffusion-eff | deprioritize | 抗记忆已被缩窗解到 0.02%(冗余);腐蚀训练加重过平滑 |
| SSSD(S4) | diffusion-eff | deprioritize | 长程已被 C1 修好;需引入 S4 层(HiPPO finicky) |
| TSDiff 自引导 | diffusion-eff | deprioritize | 条件化已被 CFG/context-cond 覆盖 |
| PFN + 金融先验 | foundation | deprioritize | 注入金融先验即坍缩成 B2 却更贵(新架构 R&D 风险) |
| Chronos / Moirai / Lag-Llama | foundation | deprioritize | 通用域负迁移 + 预测式 AR rollout 过平滑;单变量丢跨通道 |
| Rough vol(rBergomi) | econometric | **reject** | 核心参 H 需期权隐含面识别,本项目**无期权数据** |
| TimeGAN / SDE-GAN | fin-neural-gen | **reject** | 对抗 + mode collapse,~7 窗最差;SDE-GAN 被非对抗版严格支配 |
| Latent Diffusion(TimeLDM/TimeAutoDiff) | diffusion-eff | **reject** | 再叠数据饥渴 AE,往返抹平厚尾,正打中头号病 |
| CoFinDiff(小波→图像) | diffusion-eff | **reject** | 大改 + pywt 缺 + 无根因机制 |
| CSDI / TimeGrad | diffusion-eff / foundation | **reject** | 数据饥渴 + 条件补全范式错配(非无条件生成) |
| Consistency / 蒸馏 | diffusion-eff | **reject** | 只压采样步;蒸馏放大教师记忆 = 有害 |
| TimesFM | foundation | **reject** | 确定性点预测→生成必过平滑(v9 病同源) |
| MOMENT | foundation | **reject(生成)** | 无采样头不产路径;但 embedding 可强化**取证鉴别器**这半目标 |
| TimeGPT | foundation | **reject** | 闭源 API,无法微调/离线/内省 |

---

## 5. Top 推荐(5 个,跨族收敛)

> 选取逻辑:**多族独立收敛 + 数据效率 strong + 复用现有基建 + 互补可叠加**。这 5 个不是 5 个孤立模型,而是一条"**先验底座 → 杂交破 floor → 抬分母**"的有序纲领,每一步都用现成诚实闸门(复制率 + C2ST_新颖)把关。

1. **FHS / GARCH-t 计量先验(零训练强基线 + 引擎)** —— 三族(econometric top-pick、sample-eff GARCH-GMMN top-pick、foundation B2 top-pick)独立收敛到 GARCH 系。它学 ~5-8 参低维递归而非 2048 维联合,**免疫 7 窗约束**;经验残差**原生给真峰度~19**(直击 C1/line2 头号 FAIL 欠厚尾);新 innovation→**复制率≈0**;纯 numpy 60 行先出"史上最强零训练基线"。

2. **签名矩匹配主损失(对固定 held-out 真实签名库的 Sig-MMD/Sig-W1)** —— fin-neural-gen 与 sample-eff 双 top-pick。这正是项目关键发现 4 开出、v11 没做对(in-batch+权重 0.05=no-op)的**正确修复版**;签名 level-2 交叉项原生编码 SP500↔DGS10 相关(DiT 最弱处),目标分布级→**内禀抗逐点抄**;**signature.py 已 80% 就位,零新依赖**。

3. **GARCH×扩散杂交(白化预处理 / 合成增广)** —— econometric top-pick,是**唯一跳出"模型侧三杠杆已耗尽"面的结构侧正交杠杆**:把厚尾/聚集/杠杆外包给数据高效 GARCH,扩散只在近 iid 残差上学(记忆动机降),最有希望**同时压 C2ST_新颖 floor 与复制率**;落地轻(dataset 加滤波层 + generate 乘回波动)。

4. **B1 多资产同构对跨市场预训练** —— foundation top-pick,**唯一加真·独立窗**的方向(~20 市场股指↔本国 10Y → 真实独立窗 7→~140),格式与现 DiT 100% 同构,直接攻根因;诚实地说:全球危机同步使有效独立 regime 远不到 20×,但仍是 CLAUDE.md 钉死的治本方向。

5. **强正则 + 早停(near-zero-cost 抗记忆 hygiene)** —— diffusion-eff top-pick,实测 dit1d 的 attention dropout=0 / 无谱归一 = **尚未动用的抗记忆余量**,早停已被证明(ep999 质量饱和);**作为每个新实验的默认包裹**,巩固低复制 + 省半算力。它治记忆不治真实度 floor,故是地基而非主攻。

## 6. 与现有 DiT 路线的关系

- **不推翻 DiT,而是给它换"目标 / 数据 / 先验"三处油**:推荐 2/5 直接改 DiT 的损失与训练外壳(复用 signature.py/losses.py/train.py);推荐 3/4 改 DiT 的输入数据管线(dataset.py + generate);推荐 1 是 DiT 的**白盒对照基线 + 杂交底座**,且其 GARCH 残差正是推荐 3 的白化对象。
- **诚实定位**:五个推荐**没有一个能"破"** C2ST_新颖 ≈0.60 信息论 floor —— 它是数据稀缺下界。能动它的只有推荐 1/3(若真过程近 GARCH-t,用正确参数律换掉数据)与推荐 4(注入真新信息)。其余(2/5 + 多数 worth-trying)只把锁死的 Pareto 沿"低记忆+保真"推得更好。
- **取证(另一半目标)**:MOMENT embedding、forensic_cross.py 跨通道特征可强化 C2ST/Sig 假数据检出 —— 与生成主线正交,顺手收益。

## 7. 一句话结论

> 在 ~7 独立窗的硬墙前,**放弃"换更深的网",转向"强先验小模型 + 注入外部分布律 + 抗记忆 hygiene"**:落地序 = **FHS/GARCH-t 强基线 →(并行)签名矩匹配修 SigP + 杂交白化破 C2ST floor → B1 多资产抬分母**,全程用现成"复制率 + C2ST_新颖"诚实闸门把关,诚实承认 floor 是数据下界而非可凭空突破。
