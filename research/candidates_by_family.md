# 5 族逐候选详表

> 每候选:是什么 / 数据效率(对 ~7 独立窗)/ 2ch 金融适配 / 记忆化风险 / 可实现性(复用现有 DiT/eval/signature 栈)/ refs / **评判 verdict + 关键风险**。
> 评判口径锚定仓库实证:C2ST_新颖 ≈0.60–0.62 弥散 floor(模型侧三杠杆耗尽的数据稀缺指纹)、复制率(real-vs-real 底噪 0.0%)、厚尾欠(kurt 8 vs 真实 17–19,已被 clip20 攻下)、依赖现状(arch/torchsde/sigkernel/nflows/zuko/pyro/numpyro/statsmodels/signatory/iisignature 全缺;scipy/sklearn 在;signature.py/losses.py/forensic_suite.py 已就位)。

---

## 族 1 · econometric(计量 / 参数化生成)

**族判读**:数据效率优势真实(学低维递归而非 2048 维联合,免疫 ~7 窗约束),内禀避记忆(新 innovation→复制≈0)。两个 top-pick:FHS(确定性引擎/强基线)+ GARCH×扩散杂交(唯一跳出"模型侧已无杠杆"面)。落地序:FHS 强基线 → 杂交白化破 floor → component-GARCH/DCC 增强旋钮。

### 1.1 Filtered Historical Simulation(FHS:GJR/EGARCH-t 滤波 + 标准化残差自助) — **top-pick**
- **是什么**:GARCH 拟合后收益除以条件波动得近 iid 标准化残差 → 自助重采残差 → GARCH 递归把波动注回生成新路径。VaR 业界金标准。
- **数据效率**:极高。GARCH 4-6 参/通道,MLE 在 14k 日观测稳定收敛,不依赖独立窗数;残差用全样本经验分布。
- **2ch 适配**:强。递归=波动聚集,GJR/EGARCH 非对称=杠杆,经验残差=**原生峰度~19**(无 clip 天花板);双通道联合自助标准化残差向量保同期跨相关。
- **记忆化风险**:低且机制性。递归连续生成绝无接缝(修掉 A2 block-bootstrap 的 patch_spike 40x 失败);残差重抽 + 全新 innovation→逐窗 Pearson>0.95 复制率本征≈0。残余=残差块过长会抄子序列,用短/iid 残差消。
- **可实现性**:高。纯 numpy/scipy ~60 行(arch 缺无碍),零额外训练批量生成 L=2048 喂 forensic_suite。
- **refs**:Barone-Adesi/Giannopoulos/Vosper 1999;BoE WP 525 (2015);arXiv 2505.05646 (2025);MDPI Risks 13(9):166 (2025)。
- **关键风险**:GARCH(1,1) 几何衰减波动 ACF 难捕长记忆 |r|-ACF 双曲慢衰减 → 残留 long-range/run_len 项可能仍被 24 维 C2ST 微弱检出;可叠 component-GARCH 修。**相对峰度大胜属次要。**

### 1.2 GJR-GARCH-t / EGARCH-t 核心波动引擎 — worth-trying
- **是什么**:单变量条件异方差递归 + Student-t/偏-t innovation。可裸仿真,亦作 FHS/DCC/杂交的滤波核。
- **数据效率**:极高(5-7 参,整族最省数据原子,t 自由度直接控尾)。
- **2ch 适配**:厚尾+聚集+杠杆一次拿下;但裸 iid-t 仿真欠多尺度聚集与精确尾形(故 FHS 用经验残差替代);单通道需配 DCC。
- **记忆化风险**:极低(新 innovation,数学上不可能逐点复制)。
- **可实现性**:高(纯 numpy,可接 losses.py 作可微 GARCH 正则)。
- **refs**:Bollerslev 1986;GJR 1993;Nelson 1991 (EGARCH);Bollerslev 1987 (t-GARCH)。
- **关键风险**:作独立生成器与 FHS 冗余(被其包含),边际增量低;**定位=FHS 内部件 + clip20_acf baseline,不单独立项。**

### 1.3 DCC-GARCH + t-copula(双通道时变 + 尾相关) — worth-trying
- **是什么**:各通道单变量 GARCH → DCC(Engle 2002)建时变条件相关阵 → t-copula 接残差联合捕共尾。
- **数据效率**:高(DCC +2 标量,copula 1-2 参);但相关动态有效样本受危机段稀少限制。
- **2ch 适配**:本族最直击"双通道相关+跨通道厚尾":DCC 给时变相关(flight-to-quality),t-copula 给非线性尾相关。
- **记忆化风险**:低(残差重抽/参数仿真)。
- **可实现性**:中(DCC 正定化可手写但较繁;copula 须手写 scipy 多元 t;vine 在 2-3 通道属过度)。
- **refs**:Engle 2002;Aas et al. 2009;arXiv 2406.15582 (2024);arXiv 2505.06950 (2025)。
- **关键风险**:a/b 在极端相关切换上欠识别/不稳;相对 FHS 静态联合自助的边际收益可能很小。**定位=FHS 之上的相关增强臂。**

### 1.4 GARCH×扩散杂交(白化预处理 / 先验 / 数据增广) — **top-pick**
- **是什么**:(a) 白化——GARCH 标准化残差上训 DiT,生成端乘回条件波动;(b) 增广——FHS 仿真路径预训/扩充 DiT 训练集。
- **数据效率**:高→治本。把数据饥渴部分外包给数据高效 GARCH;扩散在更简单近平稳残差上学,等效大幅降独立窗需求;增广把有效训练量从 7 窗放大若干量级。
- **2ch 适配**:强且互补。GARCH 保证厚尾/聚集/杠杆/跨相关"廉价律",DiT 补残差高阶/跨通道非线性。**最有希望同时破 C2ST_新颖 ~0.60 floor 与压复制率。**
- **记忆化风险**:低且可控(残差白化后目标更平滑→记忆动机降;增广语料新仿真路径稀释逐窗记忆)。
- **可实现性**:中。dataset.py 加 GARCH 滤波层 + generate 端乘回 / FHS 路径混入 dataloader,无需新包,复用现 DiT/eval。
- **refs**:arXiv 2510.01169 (2025);arXiv 2512.21791 (2025);Quant. Finance 2025.2528697。
- **关键风险**:白化路须同时生成一致波动路径(GARCH 滤波不完美则残差留结构→DiT 仍记忆);增广路可能把 GARCH 模型偏误教给 DiT(模型误设泄露)。须 force_null 类对照确认增益非来自 context 坍缩。

### 1.5 MS-GARCH / HMM-GARCH(2-3 状态) — deprioritize
- **是什么**:隐马尔可夫在 calm/turbulent/crisis 多个 GARCH 状态间切换,直击 regime/run_len/long-range。
- **数据效率**:中(本族最吃数据一环)。crisis regime 在 14k 天罕访 → 转移阵+危机参数有效样本极小、最不稳、label-switching。
- **2ch 适配**:直击 A1/C1 残余 regime 太黏;但需联合 regime。
- **记忆化风险**:低(仿真生成);但状态过多+小数据易过拟转移结构(参数级)。
- **可实现性**:中-高(Hamilton 滤波+EM+collapsed 须手写,MSGARCH/arch 缺)。
- **refs**:Hamilton 1989;Haas/Mittnik/Paolella 2004;Ardia et al. 2019。
- **关键风险**:危机态参数+转移概率被 ~7 窗欠识别 → label-switching / 参数级过拟历史危机(失真记忆),与数据稀缺约束正面冲突。**同目标由 FIGARCH/component-GARCH(1-2 参)严格占优。**

### 1.6 stationary / tapered block bootstrap(作用于标准化残差) — worth-trying
- **是什么**:几何随机块长(Politis-Romano 1994)+ 边缘加窗(Paparoditis-Politis),**作用于 GARCH 标准化残差而非原收益**。
- **数据效率**:最高(完全非参、零参数);但上限硬——无外推、生成不出训练分布外新极端。
- **2ch 适配**:保聚集/厚尾/长程;双通道须同时刻整块重采。
- **记忆化风险**:中(核心顾虑)。长块=逐字抄子序列→复制率高(v13 病根);缓解=残差域+短/随机块+加窗。
- **可实现性**:极易(config 已有 USE_BLOCK_BOOTSTRAP/BLOCK_LEN/BOOT_FRAC,改几何块长+残差域+加窗)。
- **refs**:Politis-Romano 1994;Paparoditis-Politis 2001/2002;arXiv 2311.07738 (2023)。
- **关键风险**:作用于原收益则接缝伪影(patch_spike 40x,已观测 C2ST 反劣 0.636);全非参无外推→尾部封顶。**=FHS 的残差重抽步,勿独立用原收益。**

### 1.7 FIGARCH / GARCH-MIDAS / component-GARCH(长记忆/分量) — worth-trying
- **是什么**:FIGARCH(Baillie 1996)分数差分 d 建波动 ACF 双曲慢衰减;GARCH-MIDAS 分长短波动分量,慢分量可由宏观驱动。
- **数据效率**:中-高(仅 +1-2 参);d 估计在短/非平稳样本上 finicky。
- **2ch 适配**:直补 line1/A1 长程塌(high_vol/run_len);**GARCH-MIDAS 用 DGS10 作 MIDAS 驱动→有金融含义的双通道耦合**(利率宏观→股票慢波动)。
- **记忆化风险**:低(仿真生成)。
- **可实现性**:中(ARCH(∞) 截断递归 finicky;component-GARCH 更易;无新包)。
- **refs**:Baillie/Bollerslev/Mikkelsen 1996;Engle/Ghysels/Sohn 2013;Ding/Granger/Engle 1993。
- **关键风险**:d 对截断阶敏感;属次级长程精修非破 floor 主力。**component-GARCH 更稳,优先于 FIGARCH;定位=FHS 边缘到位后补长程的下一旋钮。**

### 1.8 Rough volatility(rBergomi / rough Heston) — **reject**
- **是什么**:分数布朗驱动随机波动率(H≈0.1),波动"粗糙"。
- **数据效率**:参数极少但**识别性差**(H/η 主要靠期权隐含面校准)。
- **2ch 适配**:弱-中(为期权定价设计,对日收益无条件厚尾不直接保证;双通道与 DGS10 不自然;需前向方差曲线输入)。
- **可实现性**:中(hybrid scheme 可写但校准 awkward)。
- **refs**:Gatheral/Jaisson/Rosenbaum 2018;Bayer/Friz/Gatheral 2016。
- **关键风险**:**核心参 H 在无期权数据下不可识别;目标错配(为隐含面而非 2 通道无条件 stylized fact)→无法从可得数据校准,ROI 最低。**

---

## 族 2 · fin-neural-gen(金融专用神经生成)

**族判读**:最优=签名/评分规则派(把目标从"拟合个别路径/对抗判别器"换成"匹配总体律+强结构先验",正是 ~7 窗该走的偏-方差权衡,无 min-max 无 mode collapse)。与本仓契合极高(signature.py + 发现3"签名是脊梁" + 发现4"固定 held-out 库")。务实下一步="签名(律/相关/长程)+ 重尾边缘(clip20/EVT)"混合体。

### 2.1 SigCWGAN / Sig-Wasserstein(期望签名匹配,Sig-W1) — **top-pick**
- **是什么**:用路径签名的"期望签名"作过程律的紧凑指纹,GAN 的 min-max 换成对固定真实签名特征的监督回归(Sig-W1);Conditional 版以历史签名为条件自回归。
- **数据效率**:★极适合 ~7 窗。匹配有限维矩向量而非个别路径;无判别器→无 min-max→稳定、样本高效;容量由签名截断阶显式控制。限制因子=期望签名估计方差(小数据"该有的"失效模式=偏向律)。
- **2ch 适配**:多维签名 level-2 S^{i,j} 原生编码 SP500↔DGS10 lead-lag/相关;**厚尾是签名弱项,须叠加边缘尾匹配(clip/EVT 层)。**
- **记忆化风险**:低。矩匹配不奖励逐点复制;对比**固定 held-out 真实签名库**(发现4药方)+ 低容量签名空间内禀抗记忆。
- **可实现性**:★高。repo signature.py 已 depth-3 截断签名 + Sig-MMD;losses.py 已有可微 depth-2 sig_mmd_loss + recover_x0(但 v11 是 in-batch=no-op)。修法=固定 held-out 库 + 主权重 + depth2→3 + 全量 n_perm,零新依赖,生成器复用 DiT。
- **refs**:Ni/Szpruch/Wiese/Liao/Xiao, Conditional Sig-W GANs, Math. Finance 2024 (arXiv 2006.05421);Sig-WGANs ICAIF 2021 (arXiv 2111.01207)。
- **关键风险**:签名对边缘厚尾盲视(峰度19须 clip20/EVT 叠加);held-out 库由重叠子窗采样→有效自由度仍≈7,期望签名方差大,大概率仍破不动 0.62 floor(只推好 Pareto);完整 conditional-AR 版工作量中等。

### 2.2 非对抗 Neural-SDE + 签名核评分规则(Issa-Salvi NeurIPS'23) — worth-trying
- **是什么**:生成器=Neural SDE(连续时间,即扩散过程),签名核评分规则(path-space proper scoring rule)非对抗训练。
- **数据效率**:高(评分规则+SDE 强结构先验样本高效,无对抗无 collapse)。
- **2ch 适配**:Brownian 维=通道数,跨相关由扩散矩阵 off-diagonal 原生给出;**纯 Brownian 薄尾达峰度19须 jump-diffusion。**
- **记忆化风险**:低(签名核 MMD 匹配分布 + SDE 紧凑参数类)。
- **可实现性**:中。需 torchsde(缺)+ sigkernel(缺,解 PDE 重)+ 全新生成器(不复用 DiT)+ adjoint 反传。
- **refs**:Issa/Horvath/Lemercier/Salvi, NeurIPS 2023 (arXiv 2305.16274)。
- **关键风险**:落地成本远高于候选1;若用仓内截断核近似则退化为候选1+SDE 生成器,得不偿失。

### 2.3 Time-Causal VAE(TC-VAE,2024 SIAM) — worth-trying
- **是什么**:因果约束 VAE(t 只依赖 ≤t),损失=causal(adapted)Wasserstein 上界,集成 RealNVP 先验。标题即"小数据稳健"。
- **数据效率**:高(causal-OT 正则 + VAE 瓶颈/KL 限容)。
- **2ch 适配**:多元原生,因果 Wasserstein 利长程;**高斯解码器倾向过平滑欠厚尾(撞 Pareto 另一端,与 C1 余病峰度8 同向)须重尾输出对冲。**
- **记忆化风险**:低-中(latent 瓶颈+KL 天然抗逐点复制)。
- **可实现性**:中(VAE+RealNVP 纯 torch 可手搓,c2st/memorization/diagnostics 闸门同口径复用;生成器非 DiT)。
- **refs**:Acciaio/Eckstein/Hou, arXiv 2411.02947 (2024);SIAM J. Fin. Math doi 10.1137/24M1711650。
- **关键风险**:是锁死 Pareto 上的另一个点而非逃逸;论文好结果用的数据远多于 7 独立窗;须配重尾输出且需调。**独特价值=因果"学习式长程"或优于 GARCH 指数衰减,值得作长程对照。**

### 2.4 Logsig-RNN(签名生成器骨干) — deprioritize
- **是什么**:log-signature 层把高频/长流压成粗分区 log-sig 序列喂 RNN,作 SigWGAN/Neural-SDE 生成器骨干。
- **数据效率**:高(log-sig 压缩→有效步少→样本高效)。
- **2ch 适配**:多维 log-sig 含跨通道项,长序列优势;厚尾同签名族弱项。
- **可实现性**:中(需 logsig 层,iisignature/signatory 缺,或经 BCH 从仓内截断签名手搓)。
- **refs**:Liao/Lyons/Ni et al., BMVC 2021 (arXiv 2110.13008)。
- **关键风险**:骨干优化非治本;复用 DiT+候选1 签名损失已 captured 大部分收益,骨干替换边际收益小。

### 2.5 条件 NF / GARCH-NF 混合 — deprioritize
- **是什么**:条件 RealNVP/coupling flow 建收益条件密度(学重尾/偏度);GARCH-NF=GARCH 抓聚集+flow 建重尾 innovation。
- **数据效率**:GARCH-NF 高;纯深 flow 中(exact-likelihood 在 7 窗对训练点给高密度=过拟合)。
- **2ch 适配**:厚尾=flow 强项(可学重尾基分布,与签名族互补);纯 flow 长程/regime 弱。
- **记忆化风险**:中-高(exact-likelihood flow 易过拟合);GARCH-NF 低。
- **可实现性**:中(normflows/nflows 缺;GARCH-NF 需 arch 缺)。
- **refs**:arXiv 2311.00580 (2024);arXiv 2311.14735 (2023);Risks 14(5):100 (2025)。
- **关键风险**:**攻的是已被 clip20 廉价解决的尾(kurt 18.8≈真实)=边际冗余,不触 0.62 联合分布 floor;依赖缺失。**

### 2.6 Fourier Flow / 谱方法 — deprioritize
- **是什么**:频域归一化流,DFT 转定长谱表示后做 flow,exact likelihood 无对抗。
- **数据效率**:中(谱压缩降维;flow 参数仍偏多,7 窗可记忆)。
- **2ch 适配**:谱天然抓 PSD/长程;**波动聚集=幅-相非线性耦合,线性谱流易欠;厚尾须重尾基分布。**
- **记忆化风险**:中-高(exact-likelihood 小数据过拟合)。
- **可实现性**:中(RealNVP+DFT 纯 torch 无特殊依赖)。
- **refs**:Alaa/Chan/van der Schaar, ICLR 2021。
- **关键风险**:两个最硬 stylized 目标恰是其弱点;投入产出比低。

### 2.7 QuantGAN(TCN-GAN) — deprioritize(stylized 强基线)
- **是什么**:TCN 生成器(随机波动 vol-TCN×drift-TCN)+ TCN 判别器,Wasserstein;高保真复现聚集/厚尾/杠杆/ACF。
- **数据效率**:低(对抗 GAN,原论文单指数 7000+ 日重叠窗=同困境,不增独立数据)。
- **2ch 适配**:原生偏单变量,2 通道需扩联合 TCN;stylized 是强项(可作真实度标杆)。
- **记忆化风险**:高(对抗+灵活 TCN 判别器,~7 窗落锁死 Pareto)。
- **refs**:Wiese et al., Quant. Finance 2020 (arXiv 1907.06673);Kwon & Lee, ICAIF 2024。
- **关键风险**:相对现有 DiT 无数据效率净增益;外部 hw01 参照判别器本身即 TCN-InfoNCE,部分冗余;**仅作 stylized-fact 真实度对照标杆,非主攻。**

### 2.8 TimeGAN / SDE-GAN — **reject**
- **是什么**:TimeGAN=RNN 自编码+监督+latent 对抗;SDE-GAN=生成器/判别器皆 Neural SDE 的无穷维 WGAN。
- **数据效率**:低(皆对抗;TimeGAN 已知"数据越多样越塌";SDE-GAN 需 gradient penalty)。
- **refs**:Yoon/Jarrett/van der Schaar, NeurIPS 2019;Kidger et al., ICML 2021。
- **关键风险**:**mode collapse/记忆化=头号杀手,~7 窗最差适配;SDE-GAN 被非对抗 Neural-SDE(候选2)严格支配,仅作消融参照。**

---

## 族 3 · diffusion-eff(扩散/score-based + 让现有 DiT 更数据高效)

**族判读**:family_summary 诚实承认"没有任何模型能凭空在 7 窗上学到泛化分布"。13 候选无一是多资产扩充,故评判退化为"治根的便宜防御 vs 冗余/错配/大改"。唯一 top-pick=强正则+早停;worth-trying=四个便宜的正交质量打磨(攻记录在案的"过平滑/欠离散"但不破 floor)。

### 3.1 强正则 + 早停(dropout/weight-decay/谱归一/EMA) — **top-pick**
- **是什么**:过参数化扩散施加显式正则 + 在记忆化窗口前早停("先泛化后记忆"隐式动力学正则窗口)。
- **数据效率**:不增数据,但直击小数据头号病=记忆化。实测"质量 ep999 即饱和、后 4000ep 只增复制率 0%→2.2%"→早停 ~1500-2000ep 免费抗记忆+省半算力。
- **记忆化风险**:正是为降记忆而设。**dit1d attention dropout=0.0 / MLP 无 dropout / 全仓无谱归一 = 尚未动用的抗记忆余量;config 已有 WEIGHT_DECAY=1e-2/EMA。**
- **可实现性**:最高(train.py 改早停/容量/wd + 给 Linear 套谱归一,零新管线,可与任意方法叠加)。
- **refs**:Why Diffusion Models Don't Memorize, arXiv 2505.17638 (2025);Early Stopping Overparameterized Diffusion, arXiv 2505.16959;Kadkhodaie et al. ICLR2024 (arXiv 2310.02557)。
- **关键风险**:记忆化已是 line2 的【已解决】项(clip11 复制率 0.02%),正则只巩固低复制+省算力,**不能把 C2ST_新颖 推过 0.62 floor——治记忆不治真实度天花板。**

### 3.2 Patch Diffusion(随机子序列训练,NeurIPS'23) — worth-trying
- **是什么**:随机裁剪/随机尺度子序列(patch)上训练,patch 坐标作条件,一条长样本派生大量子样本。
- **数据效率**:强。把有效样本数成倍放大(7 长窗→成百上千子窗)+ 小感受野降内禀维=双击根因。与项目最强杠杆(缩窗 L512→复制率 14.5%)同源但更系统。
- **2ch 适配**:通道无关;代价=牺牲长程(L512 已见长程塌),须配 context-cond/自回归(C1 备件)。
- **记忆化风险**:低(机制性降记忆)。
- **可实现性**:高(DiT 已 patchify,dataset 加随机子窗 + dit1d 加 patch 尺度/坐标嵌入)。
- **refs**:Wang et al., Patch Diffusion, NeurIPS 2023。
- **关键风险**:核心(缩窗)项目已做且判定杠杆耗尽;随机尺度相对固定 L512 只是增量,**预期落在同一 floor 非突破。**

### 3.3 Diffusion-TS(分解+x0重构+Fourier 损失,ICLR'24) — worth-trying
- **是什么**:Transformer 显式分解趋势(多项式)+季节(Fourier)+残差,直接重构 x0,叠频域 Fourier 损失。
- **数据效率**:分解先验约束到低维结构子空间→等效降内禀维;但金融日收益主导成分无趋势无季节→对收益率通道增益有限(对 DGS10 水平更有用)=中等。
- **2ch 适配**:Fourier 频域损失利波动谱/长程,x0+频域缓解过平滑。
- **记忆化风险**:x0 重构在 7 窗仍可逐点记忆,无内禀防记忆,须配早停。
- **可实现性**:中。**增量嫁接不必换骨架**:先只移植 (a) losses.py 加 Fourier 损失 + (b) eps→x0/v 目标(recover_x0 已在)。
- **refs**:Yuan & Qiao, ICLR 2024 (arXiv 2403.01742)。
- **关键风险**:趋势/季节分解先验对无趋势收益错配→"分解降内禀维省数据"被高估;须配早停。

### 3.4 Self-conditioning(回灌上一步 x̂0,ICLR'23) — worth-trying
- **是什么**:训练/采样把上一去噪步 x̂0 估计作额外条件回灌,训练期 ~50% 概率开启。
- **数据效率**:不直接降数据需求,但固定数据下提升单位样本采样质量,零额外数据/算力。
- **2ch 适配**:通道无关;缓解"过平滑/欠离散"(更锐利 x0 轨迹)。
- **可实现性**:高(dit1d.forward 增一路 x0 输入,~30 行最低成本试点)。
- **refs**:Chen/Zhang/Hinton, Analog Bits, ICLR 2023 (arXiv 2208.04202)。
- **关键风险**:正交增益不破 floor;略增拟合度须与早停/dropout 同用。属打磨非治本。

### 3.5 v-prediction / EDM 预条件 — worth-trying
- **是什么**:重参数化目标 v=√ᾱε−√(1−ᾱ)x0(或 EDM 预条件),全 t 尺度更均衡,t→0 更稳免硬 clip。
- **数据效率**:不降数据需求(正交);直接缓解"过优化 eps-MSE→过平滑/欠离散"(关键发现4)。
- **2ch 适配**:可去掉对 ±8σ 硬 clip 部分依赖(clip 是峰度天花板)→厚尾更自然透出。
- **可实现性**:中(改 scheduler 目标+losses 加权+采样换算;可只取 v-pred 不上完整 EDM)。
- **refs**:Salimans & Ho, ICLR 2022;Karras et al., EDM, NeurIPS 2022。
- **关键风险**:正交于数据稀缺(不破 floor);收益是"去过平滑风险"的间接利好而非真实度跃升。

### 3.6 GBM-SDE 对齐扩散(乘性噪声前向) — deprioritize
- **是什么**:前向加噪 SDE 对齐几何布朗(噪声方差∝状态水平),使重尾/聚集/杠杆作 SDE 结构内生。
- **数据效率**:plausible(物理先验替代数据,哲学正确)。
- **关键风险**:**纯几何 BM 是常波动→不产波动聚集/杠杆/regime(恰是真正未解短板);厚尾已被 clip20 攻下;改 q_sample 状态依赖方差+重推 DDIM 反演系数是清单里风险最高的 scheduler 手术;真正需要的随机波动先验需 GARCH/Heston,而 arch 未装(B2 已封)。**
- **refs**:arXiv 2507.19003 (2025);arXiv 2410.18897。

### 3.7 Ambient Diffusion(腐蚀数据训练抗记忆,NeurIPS'23) — deprioritize
- **是什么**:重度腐蚀样本训练,结构上无法逐点复制训练样本。
- **关键风险**:**抗记忆已被缩窗+富条件解到复制率 0.02%(边际冗余);重腐蚀在仅 7 窗上易把厚尾/波动爆发抹平,加重头号病(过平滑),与真正卡住的厚尾真实度对着干。**
- **refs**:Daras et al., NeurIPS 2023。

### 3.8 SSSD(S4 结构化状态空间) — deprioritize
- **关键风险**:**长程塌硬伤项目已用 C1 富条件+自回归修好(high_vol 0.49/d2 0.98x)→在解已解问题;需引入 S4/S5 层(HiPPO finicky),原范式偏 imputation 须改无条件生成。** refs:Alcaraz & Strodthoff, TMLR 2023 (arXiv 2208.09399)。

### 3.9 TSDiff 自引导 — deprioritize
- **关键风险**:条件化已被 CFG/context-cond 覆盖,边际价值低;卖点是 forecasting 观测引导非无条件生成保真。refs:Kollovieh et al., NeurIPS 2023 (arXiv 2307.11494)。

### 3.10 Latent Diffusion(TimeLDM/TimeAutoDiff) — **reject**
- **关键风险**:**再叠一个需在 7 窗上训的 AE=两头不讨好:AE 极易潜空间塌缩/过拟合,重构-潜空间往返本质是过平滑机器,直接抹平厚尾打中头号病。** refs:TimeAutoDiff, OpenReview 2024;TimeLDM 2024。

### 3.11 CoFinDiff(Haar 小波→图像→2D 扩散) — **reject**
- **关键风险**:大改+pywt 未装+无根因机制;2 通道须重设小波/图像编码;唯一可借鉴的交叉注意力条件已被 context-cond 覆盖。refs:Tanaka et al., IJCAI 2025 (arXiv 2503.04164)。

### 3.12 CSDI / TimeGrad — **reject**
- **关键风险**:为大规模面板设计、数据饥渴,且条件补全/预测范式(非无条件生成)根本错配;7 窗上同样严重记忆。refs:Tashiro et al., NeurIPS 2021;Rasul et al., ICML 2021。

### 3.13 Consistency Models / 蒸馏 — **reject**
- **关键风险**:**只压采样步(本项目瓶颈是质量/记忆非速度);蒸馏继承并放大教师已记忆样本,在小数据上固化记忆=主动有害。** refs:Song et al., ICML 2023。

---

## 族 4 · foundation(时序基础模型 / 预训练 / 迁移)

**族判读**:抬"7 窗"分母的唯一真杠杆=向先验注入外部分布律。三条诚实约束:(1) 多数 TSFM 是【预测】模型,无条件 GENERATE 需 AR rollout→长程系统性欠离散;(2) **负迁移真实存在**(DELPHYNE 2025 实证通用 TSFM 迁金融劣于 GARCH);(3) 预训练抬的只是【共享结构】分母,SP500↔DGS10 市场特异残差仍只有 ~7 窗。

### 4.1 B2 合成 surrogate 先训→真实微调(GARCH/DCC + 金融先验) — **top-pick**
- **是什么**:DCC-GARCH+GJR/EGARCH+Student-t(必要时 FIGARCH)生成无限合成 2 通道先训现有 DiT,真实 14734 行轻量微调。
- **数据效率**:最高之一。合成把"可学结构"分母从 7 抬到无限,真实 7 窗只做校准。DELPHYNE 实证 GARCH-only 合成 NLL 0.0865 优于混合;Chronos-2 报"纯合成≈全量"。
- **2ch 适配**:原生贴合(DCC/BEKK 产 2 通道时变相关,Student-t 厚尾,GJR 杠杆)。
- **记忆化风险**:最强防御(真实 7 窗占比极小 + 微调早停+LoRA)。
- **可实现性**:本仓最易(全离线,复用现 DiT+全套闸门,~150 行 numpy 手撸 DCC/GJR-GARCH 灌 dataset.py 预训练流)。
- **refs**:DELPHYNE arXiv 2506.06288;Chronos-2 arXiv 2510.15821;综述 arXiv 2503.11411;DCC Engle 2002/GJR Glosten 1993。
- **关键风险**:易把 GARCH 伪影(规整指数衰减聚集、缺长记忆、香草无杠杆)当真相灌进权重,7 窗微调难纠;救不了市场特异残差(floor 移动非消除)。**须 GJR(杠杆)+FIGARCH(长程)+DCC(跨通道)精心构造,naive GARCH 在杠杆/长程必塌。**

### 4.2 B1 多资产同构对跨市场预训练 — **top-pick**
- **是什么**:~20 国同构对(股指 log 收益,10Y 日差)——DAX↔Bund/FTSE↔Gilt/Nikkei↔JGB 等——预训现有 DiT,再微调 SP500↔DGS10。
- **数据效率**:最"原理正确"的抬分母(真实独立窗 7→~140),直接攻根因,无合成失真。
- **2ch 适配**:格式与现 DiT 100% 一致;股↔本国债结构同构,跨通道相关迁移自然。
- **记忆化风险**:直接降(更多真实独立窗,复制率随独立段升而降)。
- **可实现性**:中(联网取数 FRED 多国 10Y/Stooq 股指 + per-market 归一对齐;FRED 基建已半就位 fetch_fred_treasury.py)。
- **refs**:v13_data_plan.md B1 路线;数据源 FRED/Stooq/Yahoo。
- **关键风险**:**"7→140"被高估:全球危机(2008/2020)同步命中所有市场→有效独立宏观 regime 全局共享,真实独立窗远不到 20×;各国股债相关符号随 regime 翻转异质,跨市场平均可能稀释 SP500↔DGS10 特异跨通道相关。** 需联网+交易日历对齐。

### 4.3 Kronos(金融域可生成开源 FM,AAAI'26) — worth-trying
- **是什么**:专用 tokenizer 把 OHLCV K线离散,decoder-only 自回归 12B K线/45 交易所预训,支持合成 K线生成(+22% 保真)。开源 mini/small≈25M/base≈100M,MIT。
- **数据效率**:高且避负迁移(海量金融语料=大分母+金融域匹配)。
- **2ch 适配**:金融 stylized 内建;**但"通道"是 OHLCV 五元组,先验里没有 SP500↔DGS10 跨资产相关。**
- **可实现性**:中-高(HF 独立栈,需 re-tokenize + 2 通道适配微调;可作金融域权重初始化/教师蒸馏进本 DiT)。
- **refs**:Kronos arXiv 2508.02739, AAAI 2026;HF NeoQuasar/Kronos。
- **关键风险**:建股债相关须重 tokenize 丢掉大半价值;decoder AR→长程欠离散(长程塌同源病);7 窗全参微调过拟合须 LoRA/早停。

### 4.4 PFN 先验拟合网络 + 金融先验(TimePFN/ForecastPFN/TempoPFN) — deprioritize
- **是什么**:在合成先验上摊销贝叶斯推断,一次前向 in-context 预测(40-500 例)。
- **数据效率**:对极少数据结构性最优(学推断算法非数据集,训练不见真实 7 窗→结构上无法记忆)。
- **关键风险**:**注入金融先验即坍缩成 B2 却更贵——原版 GP/季节先验产不出厚尾/聚集,要补就得换 GARCH 先验=B2,但跑在未验证的 PFN 新架构 + 产路径样本需改采样头(非现成 R&D 风险);被 B2 严格支配。** refs:TimePFN AAAI 2025 (arXiv 2502.16294);ForecastPFN NeurIPS 2023。

### 4.5 Chronos / Chronos-2 — deprioritize
- **是什么**:数值量化成 token,语言模型自回归采样路径;Chronos-2(120M)group-attention 多变量+协变量,可采样。
- **关键风险**:**双撞失败模式:① 通用语料→DELPHYNE 负迁移劣于 GARCH;② 预测式长 AR rollout 系统性欠离散=过平滑(v9 病同源)。厚尾非内建须微调+合成→收敛回 B2;量化 token 对近零均值收益率差有损。** refs:Chronos arXiv 2403.07815;Chronos-2 arXiv 2510.15821。

### 4.6 Lag-Llama — deprioritize
- **是什么**:decoder-only 单变量概率预测,预测头=Student-t 分布,自回归采样。
- **关键风险**:**单变量致命伤——两通道只能独立跑,彻底丢失 SP500↔DGS10 跨通道相关;通用域负迁移。整栈不可取,仅 Student-t 厚尾头思想可移植进本 DiT。** refs:arXiv 2310.08278。

### 4.7 Moirai / Moirai-2(any-variate) — deprioritize
- **关键风险**:any-variate 多变量是唯一亮点,但通用域负迁移 + 掩码-预测/插补式(非无条件生成)改成 from-seed rollout 别扭且长程欠离散。refs:Moirai arXiv 2402.02592;Moirai-2 2025。

### 4.8 TimesFM — **reject**
- **关键风险**:**主打确定性点预测→用于 GENERATE 输出过平滑均值路径=核心失败模式(v9)直接复现;单变量无随机性无跨通道无厚尾。** refs:arXiv 2310.10688。

### 4.9 MOMENT — **reject(生成);保留作取证**
- **关键风险**:无采样头根本不产路径,无法完成生成主任务;**但其预训练 embedding 可作 C2ST/forensic 特征强化假数据检出(取证半目标的旁路)。** refs:arXiv 2402.03885, ICML 2024。

### 4.10 TimeGPT — **reject**
- **关键风险**:闭源 API,不能自由微调/无条件生成/离线复现/诚实闸门内省,与"自有可控生成器+取证"双目标根本冲突。refs:arXiv 2310.03589。

### 4.11 DELPHYNE(金融预训练基线 + GARCH 合成) — worth-trying
- **是什么**:金融时序预训练(encoder+any-variate+RoPE),GARCH+wavelet 合成预训,Student-t 混合头,少步微调超通用 FM。
- **数据效率**:strong——对本项目最大价值=**B2 的实证背书 + 配方捐献**:(a) 通用 TSFM 负迁移劣于 GARCH;(b) GARCH-only 合成有效(NLL 0.0865);(c) 10-100 例微调即足。
- **可实现性**:折叠进 B2(any-variate/Student-t 混合厚尾头/GARCH+wavelet 合成注入本 DiT),非直接接其栈。
- **refs**:arXiv 2506.06288 (2025)。
- **关键风险**:是【预测】模型非无条件生成器,权重公开性待确认。**本质是 B2 的证据与配方,非独立可跑生成器。**

---

## 族 5 · sample-eff(跨切面样本高效范式)

**族判读**:强先验/矩匹配 >> 数据饥渴大网。C2ST_新颖弥散 floor 是信息论下界,仅"注入真新信息(B1)"或"施加正确参数律(GARCH-t,若真过程近之)"可能推动;凡在 7 窗内学联合密度者只能改 SigP/复制率而被同一 floor 卡住。落地纲领=GARCH 基底(杀记忆+锁边际)+ held-out Sig-MMD 主损失(补路径微结构+跨通道 Lévy area)+ 抗记忆正则(免费 hygiene 包裹)。

### 5.1 期望签名 / Sig-MMD 矩匹配(对 held-out 库) — **top-pick**
- **是什么**:把期望签名作分布距离匹配,Sig-MMD/Sig-W1 作主损失对比固定 held-out 真实签名库;生成器可极小(MA 噪声+浅网,Lu&Sester 2024)或 neural SDE。
- **数据效率**:高度高效(矩匹配只需估有限维期望签名,生成器参数极少);7 窗下须截断 depth3-4 + held-out 库稳定化。
- **2ch 适配**:签名天然编码跨通道交互(Lévy area)→刻画 DGS10×SP500;厚尾须 Lambert/Gaussianize 预处理。Lu&Sester 在 S&P500 验证波动聚集/尾部/ACF 优于 GAN。
- **记忆化风险**:内禀强(分布级目标 + 固定 held-out 库 + 小生成器)。过去 v11 失败正因 in-batch+权重可忽略。
- **可实现性**:最高(signature.py depth-3 + Sig-MMD 已在;losses.py 已会 x̂0 反演;把签名从 no-op 辅助升主损失,纯 PyTorch)。可参考 PyTorch repo luchungi/Generative-Model-Signature-MMD。
- **refs**:Lu & Sester 2024 (arXiv 2407.19848);Ni et al., Math. Finance 2024 (arXiv 2006.05421)。
- **关键风险**:depth-3 签名对短尺度粗糙/波动聚集与边际峰度~19 相对盲视(signature.py 自注"全 L 被净增量主导");held-out 库由重叠子窗采样→有效自由度仍≈7,能钉 SigP 却未必动 C2ST_新颖 floor;作 DiT 主损失时与驱动记忆化的 eps-MSE 竞争,干净版是小生成器纯 Sig 训练=换模型。

### 5.2 GARCH-t + 小神经 copula(GARCH-GMMN) — **top-pick**
- **是什么**:GARCH 做重活(AR-GJR/EGARCH-t ~5-8 参,给聚集+厚尾+杠杆),小神经网(GMMN/copula)只建 2 维标准化创新跨通道联合依赖。
- **数据效率**:极高(本族最高之一)。GARCH 是 5 参强先验,7 窗也能稳估;NN 只学残差 copula 微小问题。**14734 点对 8 参数绰绰有余,彻底搬离"7 窗 2048 维联合"瓶颈。**
- **2ch 适配**:完美(GJR-GARCH-t 分别拟合两通道,GMMN/copula 建二维创新联合,含尾部共动/收益率曲线相关)。
- **记忆化风险**:内禀零记忆(5 参仿真器生成无限新鲜路径,复制率定义性≈底噪)。
- **可实现性**:中(arch 缺但 scipy 在,~50-100 行手撸 MLE + 小 copula 网;可作扩散基底)。
- **refs**:Hofert/Prasad/Zhu 2021 (arXiv 2002.10645);GINN 2024 ICAIF。
- **关键风险**:GARCH(1,1)/GJR 波动 ACF 指数衰减→低估真实幂律长记忆,在 long-range/run_len/high_vol_frac(最难轴)被 24 维 C2ST 检出=**模型误设**(用误设换掉过平滑/记忆,非免费);静态 copula 只给同期相关漏动态领先滞后(需 DCC/BEKK);**是唯一可能真正推动 C2ST floor 的方向(前提:真过程近 GARCH-t)。**

### 5.3 Neural SDE 强结构先验(+ Sig-W1 损失) — worth-trying
- **是什么**:NN 参数化漂移/扩散(常内嵌乘性异方差 GBM 结构),Sig-W1 或对抗损失训练。
- **数据效率**:高(强归纳偏置,尾部/波动尺度"免费")。
- **记忆化风险**:内禀强(连续动力学无法存离散路径)。
- **可实现性**:中(torchsde 缺需手撸 Euler-Maruyama;损失复用 signature.py)。
- **refs**:Sig-WGAN arXiv 2006.05421;GBM-score arXiv 2507.19003。
- **关键风险**:通用 neural SDE 网容量不天然低于 DiT→7 窗仍过平滑;对抗版小数据不稳;一旦强加波动结构即收敛回候选2/4 却多一层工程;Markov SDE 长程偏弱。**值得作候选1 损失的载体试,优先级低于 1/2。**

### 5.4 贝叶斯 SV / BSTS / GP-SSM — deprioritize
- **是什么**:隐随机波动率 AR(1)+Student-t(~6 参),或 BSTS/GP-SSM,贝叶斯先验下 7 窗良定后验。
- **数据效率**:strong(少参+先验正则);从后验仿真→结构性零记忆。
- **关键风险**:**与候选2(GARCH=频率派 SV)高度重叠却需缺失的 pyro/numpyro 或手撸 VI/粒子 MCMC;AR(1) log-vol 同指数衰减长程不优于 GARCH;GP-SSM 富核 O(N^3) 成本高。直接选候选2 更轻。** refs:Frigola 2015;Kim-Shephard。

### 5.5 数据增广(IAAFT/联合相位随机化 + 平稳块 bootstrap) — deprioritize
- **是什么**:IAAFT/相位随机化保功率谱+边际,或平稳 bootstrap 在 GARCH 残差上重采。作正则非最终输出。
- **数据效率**:高(FFT/重采样近零成本扩有效样本)。
- **关键风险**:**朴素相位随机化/IAAFT 正是非线性检验零假设→摧毁波动聚集,且无法联合保留"厚尾+聚集";项目已踩坑朴素块拼接(接缝粗糙);金融安全变体须先有 GARCH→耦合回候选2;代理非真新信息被 7 窗信息封顶。** 仅宜作轻正则。refs:Schreiber & Schmitz IAAFT 1996/2000;Politis & Romano 1994。

### 5.6 Time-Causal VAE / 因果最优传输 — worth-trying
- **是什么**:因果约束 VAE,控制 adapted Wasserstein 上界,RealNVP 柔性先验。
- **数据效率**:中高(因果距离理论界+无对抗+β-KL/RealNVP 抗过拟合)。
- **2ch 适配**:S&P500+VIX 实测复现偏度/超峰度/厚尾/平方收益聚集/长程 |r| 相关。
- **可实现性**:中(标准 VAE+RealNVP+因果 masked 网纯 PyTorch,比签名/GAN 稳)。
- **refs**:arXiv 2411.02947 (2024)。
- **关键风险**:本质仍是对整条路径的摊销重构/密度模型→与 DiT 同一稀缺 regime,重构项仍拉向抄 7 窗、KL 过强 posterior collapse 成均值路径(锁死 Pareto);防记忆是相对 plain-VAE 改善非结构性免疫;**独特价值=学习式长程可能优于 GARCH 指数衰减,作长程对照试。**

### 5.7 EVT / 柔性厚尾归一化流(Tail-GAN/Pareto-GAN/Tail-Adaptive Flow) — deprioritize
- **是什么**:显式密度小模型,EVT/广义 Pareto 尾或 Tail-Adaptive Flow;Tail-GAN 用 VaR/ES 可激发性损失专攻尾。
- **关键风险**:**只攻已被 clip20 攻下的厚尾轴(line1 tail_kurt PASS)→边际值低;GAN 变体在 7 窗加对抗不稳;nflows/zuko 缺需手撸;只作候选1/3 之上窄杠杆。** refs:Tail-GAN (Cont 等);flexible-tails NF arXiv 2311.00580。

### 5.8 跨域预训练 + few-shot / meta 微调 — deprioritize
- **关键风险**:**预训练阶段数据饥渴(用别处数据替代)→根本不解决 7 窗稀缺,等价 B1 多资产(需联网取外部同构对+预训练算力,非离线);"sample-efficient"标签名不副实。诚实归类:并入 B1 评估。** refs:OpenReview p324ryBKTc (2024);Few-Shot Diffusion arXiv 2205.15463。

### 5.9 抗记忆训练期正则(给现有 DiT 的零成本加挂) — worth-trying
- **是什么**:不换模型,加"泛化窗"早停 + 降容量/强 wd + EMA + 隐式动力学正则。
- **数据效率**:不增数据但把固定 7 窗"安全期"用满,直接压复制率。
- **可实现性**:最高(train.py 改早停/容量/wd,零新依赖,可与任意方法叠加)。
- **refs**:Why Diffusion Models Don't Memorize, NeurIPS 2025;Early Stopping arXiv 2505.16959。
- **关键风险**:**项目已大量榨取(DiT-S+wd→53.9%→36.5%、A1 缩窗→14.5%)→边际新空间小;只压复制率,对 C2ST_新颖 floor(复制率已低后剩下的过平滑/多特征联合微差)零作用——真正余墙是过平滑而非记忆。** 必要非充分的卫生措施,作每个新实验默认包裹。
