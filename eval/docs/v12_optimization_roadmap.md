# v12 优化路线图 —— 结合 v11 失败的完整复盘

> 由 15-agent 设计工作流产出(7 视角提案 × 对抗批判 + 高强度综合,2026-06-20)。
> 代码行号引用均经子智能体回查仓库核实。判定一切质量**绝不用 `score.py` 的 `ddpm_mse`/总分**。

## 0. 🚨 重大修正(2026-06-20,`eval/memorization.py` 实测后追加)

执行 A1 记忆化检查发现一个**改写全局结论**的事实:**所有训练出的扩散模型都在大比例复制训练窗**。
复制率(与最近邻训练窗原序列 Pearson>0.95 的样本占比;标定 real-vs-real 非重叠底噪 = **0.0%**):

| v10_retrained | v11 | v10_sampling | v9 | real-vs-real 标定 |
|---|---|---|---|---|
| **53.9%** | 48.5% | 41.9% | 30.1% | **0.0%** |

- 近半数生成样本是训练窗的**近乎逐点复制**(pearson 0.9999,值吻合到 4 位小数)。已排除 by-construction:`c=window[0]` 仅 2 维初值、`x_T` 从纯噪声起 → 复制 = **记忆化**。
- **复制率反相关于既往"真实度"排名**:被判"最佳/最真实"的 v10_retrained 复制**最多**。**既往评估被记忆化污染**——模型显得真实部分因半数输出就是真实训练窗;C2ST/Sig-MMD 被复制的那一半愚弄。
- **结论重排**:头号病根从"过优化/过平滑"上修为 **数据稀缺导致的记忆化**(~2538 重叠窗、仅 ~7 个独立段 vs 58M 参数)。下面路线图的**优先级据此调整**:
  1. **B 层(抗记忆化:减容 B3 / 正则 B2 / 数据增广 B5 / 更多数据)从"中等"升为头等**;v11 的"过优化"只是同一数据稀缺病的一个症状。
  2. **评估必须新增"记忆化/新颖度闸门"**(用 `eval/memorization.py`):达标模型复制率须**显著低于** v10_retrained 且新颖半数仍通过鉴别器——现有 wasserstein/峰度/C2ST/Sig-MMD **都在奖励复制**,不能单独作为质量判据。
  3. 任何"让分数更好看"的改动须先问:**是真的更真,还是复制得更多?**

> 下文 §1–§5 是 memorization 发现**之前**产出的路线图,逻辑仍有效但需在上述修正下读:A 层 A1 已落地并升级为一等闸门;B 层抗记忆化条目优先级整体上调;C 层签名/CFG 在记忆化未解前收益存疑。

---

## 1. v11 教训定调

v11 的失败不是签名思路错,而是**两个机制性事实**:(1) 在 2538 个高度重叠窗(实测仅 ~7 个非重叠 SEQ_LEN 段)上,把 `eps-MSE` 经 `CosineAnnealingLR(eta_min=1e-6)` 一路退火压到 0.001 = **过拟合/记忆化训练目标**,样本分布反而更偏离真实——"更低训练损失 ≠ 更真样本";(2) `SIG_MMD_WEIGHT=0.05` + 同 batch 子窗对比(`losses.py` 无偏 MMD² 早早卡在 ~−0.001 噪声地板)= **无效信号**,既被 eps-MSE 淹没又易被平凡满足。**核心纠正:先用正则/早停堵住"过优化去噪场"这个土壤,再谈任何辅助损失;辅助损失永远救不了一个被训到背诵训练集的去噪器。**

---

## 2. 路线图(三层)

> 通用验证铁律(贯穿全部):**绝不用 `score.py` 的 `ddpm_mse`/总分做 go/no-go**。质量判据 = `c2st.py` acc(标定 real-vs-real≈0.495)+ `signature.py` Sig-MMD p(标定≈0.326)+ `diagnostics.py`(d2_energy / high_vol_frac / PSD gap)+ wasserstein/峰度单项。**统一参照基线一律用 `output/deep_v10_retrained.csv`,不再用 v9。**

### A 层 — 零/低成本立即可试(zero-retrain 或纯评估脚手架)

| # | 提案 | 假设 | 方法(改哪/怎么改) | 治哪个病根 | 成本 | 风险 | 验证指标 |
|---|---|---|---|---|---|---|---|
| **A1** | **记忆化检查 `eval/memorization.py`(新建)** | v11 是记忆化而非学到更真分布 | 把 5120 条 fake 与全部 2538 train 窗在 `c2st.py` 的 24 维 stylized 特征空间算最近邻距离 d_NN;以 held-out 真实窗的 d_NN 分布作非记忆参照;top-k 最近对再回原 z-score 空间做逐窗 Pearson 二次确认 | **直接证伪/坐实 v11 根因诊断** | 极低 | 几乎无(纯测量) | v11 的 d_NN/参照 ratio 是否显著 < v10_retrained;报 bootstrap 区间 |
| **A2** | **统一 v10_retrained 为对照基线 + 三联验收闸门** | 当前以 v9 为基线导致参照系混淆 | 在 v12 handoff 写死:任何新模型须同时通过 c2st acc<0.750 ∧ Sig-MMD p>0.395 ∧ diagnostics 三族不劣化,且**每次带 real-vs-real 标定臂用"到标定的相对距离"判定**,不看绝对 p | 评估方法学根因 | 极低 | 无 | 复现既有同序结论(v10重训最真 / v9+采样被检出 / v11回退) |
| **A3** | **鉴别器卫生:真异源标定 + 同口径断言**(`signature.py` / `c2st.py` 微改) | 现 real-vs-real 标定是"同源不同 seed",偏乐观 | `signature.py:145-146` 标定改为 train 子窗 vs **held-out** 子窗;`c2st.py` 加断言 vol_threshold 对 real/fake 同口径;加 **fake-vs-fake** 警报(异常低→记忆/塌缩) | 量具校准 | 低 | 改判据口径须先回测 | 新口径仍输出四候选同序,否则以新口径为准 |
| **A4** | **末步 churn(配合降 steps)** | DDIM 末步 `x=sqrt(1.0)*x0_pred` 是零方差均值输出 | `scheduler.py` 末步 `sigma` 改用 DDPM `posterior_std(t_curr)`,系数扫 {0,0.5,1.0};**关键配合 steps=50**(末步 t≈19 残留 std≈0.076 才有可注噪频带),steps=200 时 t=4 信号已 99.97% 干净,收益接近零 | 过平滑(高频带,已大部被 eta=1 修复→收益小) | 零重训 | 低(可能过冲 d2_energy) | N=512 选点 → **N=5120 复核** Sig-MMD≥0.395 / C2ST≤0.750,任一退即弃 |

### B 层 — 中等(单次重训可验证)

| # | 提案 | 假设 | 方法(改哪/怎么改) | 治哪个病根 | 成本 | 风险 | 验证指标 |
|---|---|---|---|---|---|---|---|
| **B1 [最高]** | **削训练预算 + 采样侧早停 + checkpoint 扫描** | v11 病根是过优化退火太深 | `config.NUM_EPOCHS` 5800→**3500**(回到 v10 量级);`eta_min` 1e-6→**1e-5**(别把 mse 拖到 0.001);`CHECKPOINT_EVERY` 2000→**500~1000** 密集存档;**早停判据挂在采样侧**——每 ~500ep 抽 256 条跑 diagnostics + c2st/signature,选"判别器最不可分"的 EMA ckpt,**不用** `min(loss_history)`(`train.py:288` 自指) | **过优化(头号根因)** | 中(1 次重训) | 极低(最坏复现 v10) | c2st/signature 在采样曲线达峰的 ckpt;eps-MSE 仅作过拟合监控 |
| **B2** | **容量正则网格(EMA/wd/dropout)** | 58M vs ~7 独立窗的容量错配 | `EMA_DECAY` 0.995→**0.9995**(现时间常数仅 ~200ep,对数千 ep 近乎无正则;须与 B1 早停 epoch 联调,打印 EMA 与在线权重参数距离确认追平);`dit1d.py:210` attention + **MLP 补 `nn.Dropout` 插槽** dropout=0.1;`WEIGHT_DECAY` 1e-3→**1e-2** 单独一档;**wd 用参数分组**只作用 attn/mlp,不碰 LayerNorm/bias/adaLN | 容量错配土壤 | 中 | 过强→欠拟合回到 v9 过平滑(可由 diagnostics 即时观测回退) | 过拟合拐点后移 + c2st/signature 检出下降;roughness/regime gap 未扩大 |
| **B3** | **DiT-S(32.6M)缩容裸跑** | 容量过剩是 v11 越训越崩的根因 | `train.py --model dit-s`(分支已存在);**完全沿用 v10 配方、关 `USE_SIG_MMD`、不加 dropout**,作干净基线把"缩容"与"加正则"解耦 | 容量错配(最直接证伪实验) | 中 | 可能欠拟合丢尾部(鉴别器可一票否决) | DiT-S vs DiT-B 在 v10 配方下的 val gap 与 c2st/sig;持平且 gap 更小即证容量过剩 |
| **B4** | **v-prediction 替代 eps**(与 B1/B3 捆绑) | eps-MSE mode-average + 末步均值化致过平滑 | 改 `scheduler.py`/`losses.py`/`generate.py` 三处 + `config` 加 `PARAM_TYPE` 开关保留 eps 可回退;`USE_MIN_SNR` 必须**关**(v-pred 自带等价加权);x̂0 走 `x0=√ᾱ·x_t−√(1−ᾱ)·v` | 过平滑(结构性,不增过拟合自由度) | 中 | 链路一处不同步即静默劣化(本仓库 ddim eta/CFG 改错不报错) | **上线前三件硬对拍**:eps↔v 互转数值一致到 1e-5;用现有 eps ckpt 仅采样端"v 反推 eps"确认 wasserstein/峰度不变;2-ep 冰烟看梯度健康。**单独消融**净效应 vs v10_retrained |
| **B5** | **保形数据增广:整窗符号翻转** | log-return 近似对称,\|·\|/平方类 stylized fact 对 r→−r 不变 | `dataset.py` 仅 train split 开:整窗 r→−r,**双通道联动同号翻转**(保跨通道相关,独立翻转留消融);**条件向量 c=window[0] 必须同步翻转** | 容量错配(边际缓解,~7→14 独立窗) | 低 | 边际收益小;须查峰度未被虚增 | 靠 B1 的 val gap 确认真降过拟合;终判峰度/wasserstein 未漂移。**砍掉幅度抖动**(虚增尾部) |

### C 层 — 研究性大赌注(仅在 B1+B2/B3 证明过拟合拐点后移后再碰)

| # | 提案 | 假设 | 方法 | 治哪个病根 | 成本 | 风险 | 验证指标 |
|---|---|---|---|---|---|---|---|
| **C1** | **可微 Sig-MMD 重做:固定 held-out 真实参照池 + 持续梯度** | 同 batch 对比无梯度是 v11 真 bug 之一 | `dataset.py` **先切真正时间 held-out 段**(两侧各丢一个 SEQ_LEN 缓冲带防泄露);bank n≥512、每若干 ep 重采子窗防记忆;带宽用 bank 中位数一次固定;**缩放改为与 eval 一致**;**1-batch 冰烟确认 sig grad norm 与 eps-MSE 可比才启动**;权重 0.02 起 | 签名脊梁(辅助损失) | 大 | **v11 同型陷阱近亲**:仍可能只匹配低阶矩、被过优化淹没 | 必叠 B1 锁住 eps-MSE;判据用**与训练不同 depth=3/不同子窗/不同 seed** 的 signature.py,出现"sig 降而 c2st 升"立即回滚 |
| **C2** | **强 CFG 的有效条件重设计** | 现 c=window[0] 与实现波动相关≈0.023,CFG 全程做无用功 | c 换多统计向量 [std, mean(\|r\|), 趋势, 前导段 d2_energy];**额外保留 lookback 段不计入待生成 x0**(切断训练/采样口径泄露);w∈{1,1.5,2,3} 各跑鉴别器 | 无效条件(次要病) | 大 | 强 CFG 削尾,危及 v10 最值钱的峰度/尾部资产 | regime 改善是否以 Sig-MMD p 下降为代价;**峰度/尾部不退设一票否决** |

---

## 3. 推荐的下一步实验序列(有序,每步含非自指达标判据)

1. **【根因消融,最先做】** 用**与 v11 完全相同的 5800ep + 完全相同配方,仅 `USE_SIG_MMD=False`** 重训一版 `deep_v11_noSig`。**达标判据:** 隔离出"Sig-MMD 项"与"过优化退火"各自的贡献——若 noSig 版仍回退(c2st>0.750 / Sig-MMD p<0.395),则证实**退火过深才是主因**(B1 优先级最高);若 noSig 版回到 v10 水平,则证实 Sig-MMD 项主动有害。**这是整张路线图的归因基石,必须先于一切重训。**
2. **【零重训,并行】** 建 `eval/memorization.py`(A1)+ 统一 v10_retrained 基线与三联闸门(A2)+ 鉴别器卫生(A3)。**达标判据:** memorization ratio 对四候选给出可解释排序且复现"v10重训最真";A2/A3 新口径回测仍输出四候选同序。
3. **B1 削预算 + 采样侧早停 + 密集 checkpoint** 重训(`NUM_EPOCHS=3500`,`eta_min=1e-5`,`CHECKPOINT_EVERY=500`)。**达标判据:** 存在某中段 EMA ckpt,其 c2st acc<0.750 且 Sig-MMD p>0.395,且 diagnostics 三族 gap 不劣于 v10_retrained。
4. **B3 DiT-S 裸跑**(v10 配方,关 Sig)。**达标判据:** DiT-S 的 c2st/sig 持平或优于 DiT-B 且 train/val eps-MSE gap 更小 → 坐实容量过剩,后续以 DiT-S 为主干。
5. **B2 容量正则网格**(在 B1 或 B3 胜者上做单变量消融:EMA→0.9995 / wd→1e-2 / dropout=0.1)。**达标判据:** 过拟合拐点后移、c2st/sig 检出下降,且 roughness/regime gap 未向 v9 漂移。
6. **B4 v-prediction**(捆绑 B1 削预算 + 三件硬对拍)。**达标判据:** 净效应消融——v-pred + 关 min-SNR + 关全部 aux,训到 v10 同等 mse 落点,d2_energy/vol-of-vol gap 缩小且 c2st/sig 不劣于 v10_retrained。
7. **(仅当 3–6 证明拐点后移)** C1 固定 bank Sig-MMD,作独立小权重消融,判据用错开配置的 signature.py。

---

## 4. 反共识警示(看似合理但很可能重蹈 v11 覆辙,点名否决/降级)

- **❌ GradNorm / 自适应辅助权重自动抬升** — `likely-regress`。它的设计目的就是"eps-MSE 降时自动抬高辅助项权重持续施压",恰恰把 v11 的踩雷动作**自动化、去掉人工权重上限这个安全阀**。辅助项是真实分布的低维投影,持续加压必诱发 metric-gaming(该投影变好而 c2st 整体变差)。**仅保留其数值卫生碎片**(recover_x0 的 ᾱ 门控加上限 0.1<ᾱ<0.9、aux clamp 从 8σ 收到 ~3σ),不抬权重。
- **❌ GAN 式可微判别器与生成器协同进化** — `likely-regress`,比 v11 更危险。2538 重叠窗 + 无 held-out → 判别器学到窗口捷径并泄露给生成器,**独立 c2st 评测必恶化**;把对抗目标=评估特征族直接变损失=最严重的同流形自欺。
- **❌ cosine β-schedule / zero-SNR 末端重采样** — `likely-regress`。train/inference schedule 失配是硬伤,几乎必然打崩 wasserstein/峰度,而峰度/尾部正是 v10_retrained **最不该拿去赌的资产**;且 abar_T 致均值泄漏仅 0.6%,理论收益微乎其微。
- **❌ 多种子 ensemble + 按 regime 重采样** — `likely-regress`,**评估诚信红线**。按 high_vol_frac 筛选 = 把测试集泄漏进后处理,制造"分数涨模型没变好"的假象。仅可作诊断,结论不得写进质量报告。
- **⚠️ 多尺度 patch embedding / 可学习相对 PE** — `weak`。在容量过剩土壤上**增加表达力 = v11 同向操作**;相对 PE 在仅 128 token 上增益通常测不出,却引入手写 attention 的静默 bug 风险。容量根因未解前不碰。
- **⚠️ 单段尾部 held-out + holdout eps-MSE 早停** — 自相矛盾:holdout eps-MSE 与失败量同型(去噪回归),**测不到"采样分布偏离真实"**;且尾部 15% 仅 33 个 99% 重叠窗,选 ckpt 极不稳。**早停判据必须挂采样侧**(diagnostics/c2st/signature),不用 holdout eps-MSE;若要 holdout 改**分块 walk-forward** 仅作旁证。
- **⚠️ STRIDE 5→2 扩样** — 收益递减,独立信息增量近零(同 7 段行情更密切片),反加重窗间相关、拖慢训练。**不优先做。**
- **⚠️ 显式杠杆效应损失** — `weak`。定义的 corr(r,|r_{t+k}|) 与 c2st 的 tc(截面相关)**不是同一统计量** → 盲优化判别器不看的量;先写诊断脚本测 gap,若本就小直接放弃。

---

## 5. 一句话:若只能做一件事

**做实验序列第 1 步——用与 v11 完全相同的 5800ep 配方仅关掉 `USE_SIG_MMD` 重训一版,隔离出"过优化退火"与"Sig-MMD 项"各自的责任**;它零设计成本却唯一能告诉你 v11 到底败在哪,从而决定后续是先削预算(B1)还是先弃签名损失,避免再盲飞出一个 v11。

---

## 相关文件(绝对路径)
- 核心改动:`config.py`(NUM_EPOCHS/eta_min/EMA_DECAY/WEIGHT_DECAY/CHECKPOINT_EVERY/USE_SIG_MMD/dropout 开关)、`train.py`(`CosineAnnealingLR(eta_min=1e-6)` @L125-128、`min(loss_history)` 自指 best @L288、sig 门控 `>=4` @L218)、`losses.py`(`sig_mmd_loss` 同 batch 对比)、`scheduler.py`(DDIM 末步零方差 `x0_pred` @L288)、`dit1d.py`(dropout=0.0 @L210、MLP 无 dropout 插槽)、`dataset.py`(全量 fit 无 split)
- 评估:`eval/{c2st,signature,diagnostics,score}.py`;新建 `eval/memorization.py`
- 基线数据:`output/deep_v10_retrained.csv`(对照)、`deep_v11_val.csv`(回退样本)
