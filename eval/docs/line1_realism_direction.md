# line1 真实度优先路线图 —— 真实度直追市场 + 同步增强 OUR 鉴别器

> 总撰稿:line1 深度优化（基于 gen / disc / method 三视角提案 + 对抗校验）
> 日期:2026-06-24 ｜ 分支 `deep_v10` ｜ 仓库 `~/src`
> 配套档案:`configs/line1.py`（须重写,见 §4）｜ 记分牌:`eval/scoreboard.py`（须改三栏,见 §5）
> 关联:`eval/docs/v13_data_plan.md`(line2 数据治本)｜ `thirdparty/THIRDPARTY_CLAUDE.md`(hw01 仅作机制借鉴)

---

## 0. 执行摘要(修正后的北极星)

**旧 line1 北极星已作废。** `configs/line1.py` 当前仍是【旧 fool-recipe】:
`PRIMARY_METRIC="external_fool"` + `CLIP_RANGE=8.0` + `USE_AUX_LOSS=False` —— 每一条都是
**为骗过队友 hw01 鉴别器而主动牺牲真实度**(贴近过平滑质心、关掉厚尾辅助因为厚尾会被 hw01 判假)。
对抗校验实证(`thirdparty/THIRDPARTY_CLAUDE.md`):hw01 判别器**奖励过平滑、惩罚真实厚尾**
(其 OOD 负类含 GARCH-t 厚尾 → 真厚尾被推向"合成"侧)。**追 hw01 fool = 主动远离真实市场。**

### 修正后的北极星(本路线图唯一目标)

> **以 v9/v10 全窗(L=2048)为基座,容许复制(不缩窗、不罚记忆),让生成样本在【诚实分布距离】
> 上逼近真实市场;同时把 DGS10 量化指纹等真实结构补全、并增强 OUR forensic 鉴别器(双重目标
> 另一半)。hw01 fool 率仅作【外部参照】并报,绝不作为优化目标。**

一句话方向:**放开 CLIP 峰度天花板(8→15)+ 开 stylized 辅助攻波动聚集 + DGS10 量化吸附补真实结构;
判定改用【否决式真实度向量】(任一维落出 real-vs-real 自检带即 FAIL,不互相补偿),并用
all-vs-novel 污染 gap 守住"真实度不是靠抄来的"。** 三个 P0 全是零训练 post-process,先打通再烧 GPU。

### 与 line2 的正交性(务必守住)
- line2 = **生成器侧**(缩窗 L=512 + context-cond + 罚复制,北极星=新颖度)。
- line1 = **生成器侧真实度(容许复制、全窗 L=2048)** + **鉴别器侧增强(eval-only)**。
- 二者不交。本路线图所有"攻厚尾"是 **容许复制版**(放宽 clip + AUX),与 line2(缩窗 + 罚复制)不重复。

---

## 1. v9/v10 真实度硬伤诊断

真实数据自检(零训练 numpy,已复核):真实 SP500 列**本身已是日 log-return**
(`mean≈0.0003 / std≈0.01 / 含负值`,`dataset.py` 直接 z-score,不再 diff/log),
标准化后 `excess_kurt = 18.80`,`max|z| ≈ 19.4`;真实 DGS10 diff `excess_kurt ≈ 6.76`。

| 硬伤 | 病灶证据 | 根因 | 靶点(诚实北极星指标) |
|---|---|---|---|
| **欠厚尾 / 峰度天花板** | v9 峰度 ~4.7、v10 重训 ~7.7、c1 ~8(锁死);真实 18.8。numpy 实证:`CLIP_RANGE` 是峰度天花板 —— clip8→`excess_kurt 7.92`(精确=v10 的 7.7)、clip12→10.98、clip15→**12.90**、clip20→18.80 | **数据管道硬截断**(`dataset.TimeSeriesScaler.transform` ±8σ)把训练数据本身的峰度钉死;模型在被 clip 的数据上**学不到超过 clip 的峰度** | SP500 峰度(7.7→12-15 逼近 18.8)+ wasserstein 尾部分位 + `\|r\|>4σ` 频率 |
| **波动聚集弱** | `\|r\|-ACF` 偏弱(`diagnostics.py` lag1-5);v9 波动爆发不足 | **eps-MSE mode-averaging** 只学条件均值、抹平高阶结构(关键发现 1) | `\|r\|-ACF` 曲线 L2 距离 + `high_vol_frac`(目标 0.5) + `vol_of_vol` |
| **regime 黏滞 / 长程过持续** | `mean_run_len` 过长(真实 ~39)、收益率过持续;DDIM 确定性末步均值化 | 末步输出 x0 均值 + eps-MSE 不惩罚长程波动结构 | `mean_run_len`(~39) / `switch_rate` / PSD 低频能量占比 |
| **DGS10 缺量化指纹** | 真实 DGS10 diff `frac_on_grid(0.01)=1.0`(100% 落网格,债券按 basis point 报价);**全部**扩散生成 `≈0`(连续浮点) | 生成端从未做量化吸附(`DGS10_QUANTIZE` 是死旋钮,见 §3.2) | `frac_on_grid`(目标 ≈1.0) + DGS10 通道 wasserstein |

> ⚠️ **per-channel 解耦**:CLIP 放宽对 **ch0(SP500)** 有大收益(7.92→12.90);但 **ch1(DGS10 diff)**
> 峰度 clip12 即 plateau(~6.76)、clip8(~6.26)已近饱和 → 放宽到 15 对 ch1 **无收益且可能引入数据错误离群**。
> 建议 per-channel clip(**ch0=15 / ch1=8**)或接受 ch1 微损。

---

## 2. 两条 Track 全表(合并去重)

> **TrackG = 生成真实度(生成器侧)**;**TrackD = 鉴别器增强(eval-only)**。
> 校验裁定后:gen 5 项(2 keep / 2 downgrade / 1 拆分),disc 6 项(3 keep / 2 downgrade),method 6 项(4 keep / 2 downgrade)。
> 合并去重后:**DGS10 量化**(gen-2 = disc-B1 = method-③)三方汇聚 → P0;**长程修复**(gen-5)前半拆为独立零训练扫描。

### TrackG — 生成真实度

| # | 方向 | 治哪 | 成本 | 最低成本验证 | 北极星指标 | 防 Goodhart / 防泄露 | 优先级 |
|---|---|---|---|---|---|---|---|
| G1 | **放宽 CLIP_RANGE 8→15**(ch0;ch1 保 8) | 欠厚尾/峰度天花板 | full-retrain | **已跑** numpy 上界:clip{8,12,15,20}→kurt{7.92,10.98,**12.90**,18.80}。重训前拉 ep500/1000/1500 中间 ckpt,`diagnostics.py` 看峰度随 clip 抬升;仍卡 ~8 则非 clip 病立即停 | SP500 峰度(→12-15)+ wasserstein 尾部分位 + `max_rolling_vol` | 放宽 clip 会让 hw01 fool **下降**而我们**接受** = 反 Goodhart 实证。容许复制不引盲:novel 子集(§G6)暴露是否靠抄真厚尾窗。⚠️ 修正校验命令:**直接对 sp500 列 z-score+clip**,勿 `np.diff(np.log(sp))`(对已是收益率含负值列再 log→NaN) | **P0** |
| G2 | **DGS10 量化吸附 0.01 网格**(生成端接线 + 后处理) | DGS10 缺量化(真属性) | post-process(零训练) | **已跑**:v9 on_grid `≈0`→量化后 `1.0`,扰动 mean 0.00058/max 0.005(半格内);对 wasserstein 仅 0.0003 冲击、峰度 6.81→6.817 零破坏 | `frac_on_grid`(→1.0)+ DGS10 wasserstein/峰度 | 真·真实属性非 gaming。⚠️ **作用边界**:hw01 实证它"只解释扩散整体被识破、不解释模型间排序"→**不计入真实度排序主因**,只补结构 + 喂 OUR 鉴别器。⚠️ 量化是确定性后处理,**任何造假者一行 `np.round` 即可绕过** → 在鉴别器里仅作低成本筛、非主信号(见 §D5) | **P0** |
| G3 | **开 stylized 辅助 USE_AUX_LOSS=True + 新增 kurt 项** | 波动聚集弱 + 厚尾 | full-retrain | 中间 ckpt 轨迹看 `\|r\|-ACF` gap / `high_vol_frac` 随训练改善。**先单跑 USE_AUX_LOSS=True 但 clip 仍 8 的消融**,隔离 AUX vs clip 贡献 | `\|r\|-ACF`(lag1-5)+ `high_vol_frac`(0.5)+ `mean_run_len`(~39)+ 峰度 | kurt 项 `((z**4).mean()-3)` 须对 **全量真实四阶矩常数**(非 in-batch,避关键发现 4 的 Sig-MMD 平凡满足教训)。门控复用 `AUX_ABAR_MIN`(仅低噪声步 x̂₀ 可靠)。监控复制率是否因 AUX 上升(stylized 曾推 v10 到 53.9%),并报 novel 子集 | **P1** |
| G4 | **长程 eta/steps 零训练扫描**(从 gen-5 拆出) | regime 黏滞 | inference(零训练) | v10 ckpt 扫 `--eta {0.5,1.0} × --num_inference_steps {200,500}`,每批 ~5min,`diagnostics.py` 比 `mean_run_len/switch_rate/high_vol_frac`。**若纯采样就把 `mean_run_len` 拉到 ~39 则长程辅助损失免做、省一次重训** | `mean_run_len`(~39) / `switch_rate` / `high_vol_frac` / PSD 低频 | 选 eta 工作点约束:**不破坏峰度**(eta 增纹理但勿打崩尾部);报 novel 子集确认不是靠复制改善 regime | **P1** |
| G5 | **v-prediction 替换 eps-prediction** | 过平滑(结构级) | full-retrain | **不可纯 numpy 验**。先写单测:`q_sample`→`get_v`→反演 x0/eps 与现公式数值一致(<1e-5)才接训练。短跑 1500ep + 中间 ckpt,与 eps 同 clip/AUX 做 A/B | 同诚实套件(峰度/wasserstein/`\|r\|-ACF`/regime),与 eps 在**相同 clip/AUX**下 A/B | 与 G1/G3 高度共线(都攻过平滑/厚尾),clip=8 天花板下也填不出 >12.9;增量恐被 P0/P1 吸收。**降级排在 clip15+AUX 出结果后**,仅当峰度仍 <12 且诊断指向末步均值化才上。保留 `PREDICTION_TYPE` 开关可回退 | **P2** |
| G6 | **长程滚动波动 ACF 辅助损失**(gen-5 后半) | 长程波动聚集 | full-retrain | 同 G3 中间 ckpt。**全窗 L2048 本就保长程上下文**(line2 缩窗才塌长程),长程病在 line1 未必严重 | `mean_run_len` / PSD 低频 / `vol_of_vol` | 与 G3 的 `\|r\|-ACF` 项**强重复**,须合并避免重复正则、共用门控;对全量真实 ACF 常数(非 in-batch)。**若 G4 零训练扫描已达标则免做** | **P2** |

### TrackD — 鉴别器增强(eval-only,绝不回喂生成器)

| # | 方向 | 治哪 | 成本 | 最低成本验证 | 北极星指标 | 防 Goodhart / 防泄露 | 优先级 |
|---|---|---|---|---|---|---|---|
| D1 | **c2st.featurize 加 DGS10 量化 3 特征**(`rounding_0.01 / zero_diff_ratio / unique_value_ratio`) | 鉴别器缺量化指纹 | eval-only | **已跑**:仅这 3 维 LogReg 在 real vs deep_v10 上 **AUC=1.0**,real-vs-real 标定 AUC=0.537(≈0.5 不污染) | AUC(对生成→1.0)+ 标定(→0.5) | 确定性逐窗统计、无近邻/无跨窗 → **零泄露**。借鉴 hw01 `dgs10_fingerprint@244` 但只取确定性 3 维。⚠️ 易被 `np.round` 绕过(见 D5),**不接成生成损失**(否则触发 line1 自己的 `DGS10_QUANTIZE` 规避 OUR 检测器) | **P0** |
| D2 | **新建 `eval/forensic_auc.py`**:口径升级为标定 ROC-AUC + 阈值报表 + **时间 held-out** | 鉴别器量化口径(检出率/p值→AUC) | eval-only | scoreboard 重排 12 模型,看 AUC 排序 vs forensic composite 的 Spearman;消融每特征族边际 AUC | 标定 AUC(real-held-out vs real)≈0.5 + detect AUC→1.0 + 各族边际增益 | ★**命门**:现有 `c2st.py:91` 在 stride=5 的 ~2538 重叠窗上 `train_test_split`,相邻窗共享 >99% 点 → **已带时间邻接泄露**。须按**时间**切(前 80% 训/后 20% 测)+ **缓冲带 gap ≥ `ceil(L/stride)`=410**(复用 `memorization.py:86` guard 范式)。5 折须按时间块切非随机 KFold。**自检:前 80% real vs 后 20% real 标定 AUC 须 ≈0.5** 才放行 | **P0** |
| D3 | **c2st 加特征空间近邻特征**(`nearest_real_ratio / dNN_median`) | 鉴别器缺几何信号(抓 v9/v10 过平滑) | eval-only | `forensic_auc.py` 做"仅统计 vs 统计+近邻"消融 AUC + 偏相关(控制峰度后近邻信号是否存活) | 消融 AUC 增益 + 近邻与统计正交性 | ★高泄露:重叠窗上 `nearest_real_ratio` 会被 99.76% 重叠邻窗压成 ~0 → 伪信号。**必须复用 `memorization.nn_real_ref` 的 guard 排除 `\|i-j\|≤guard`**。借鉴 hw01 头号信号 `nearest_real_ratio@739`,但**先用零训练特征空间近邻,暂不重训 TCN**。⚠️ 近邻惩罚记忆化、与 line1 容许复制反向 → **文件级标注 `NEVER feed to generator loss`,仅作鉴别器只读特征** | **P1** |
| D4 | **signature 修长程盲区**:多尺度 `l_sub ∈ {200,500,1000}` + depth 报表 | 鉴别器长程盲(`l_sub=200` 在 L=2048 仅覆盖 9.8%) | eval-only | 对长程塌模型(v13_a1 concat2048 high_vol 0.27)扫 l_sub,看 SigP 随 l_sub↑ 而↓;对 v10_retrained 不过度误伤 | 长程检出:塌模型 SigP 随 l_sub↑ 而↓;标定 SigP 各 l_sub 仍 ≥0.05 | l_sub 增大时长程子窗可采样位置变少 → 多尺度间子窗更重叠,**须确认标定 SigP 各 l_sub ≥0.05**(否则是采样重叠伪检出)。**M5 铁律:SigP 须全量 `n_perm≥300`,禁 `--quick`**(会毁 sig 标定 0.326→0.046)。借鉴 hw01 `CONTRASTIVE_WINDOWS=(63,126,252)` 多尺度思想 | **P1** |
| D5 | **鉴别器鲁棒性校准**:对抗后处理压测(量化吸附/平滑)+ leave-one-family-out + 多检测器投票 | 鉴别器单腿(D1 易被规避) | eval-only | 对 `deep_v10.csv` 做 DGS10 量化吸附 + 轻平滑生成对抗版,测 D1 单族 AUC 从 1.0 掉到何处(预期掉到统计族 ~0.7-0.8);集成 AUC 须仍 ≥0.9 | leave-one-family-out 最弱 AUC + 对抗后处理下 AUC 保持率 | 正确预见 D1 致命弱点:line1 自己的 `DGS10_QUANTIZE=0.01` 会让 OUR 生成器规避 OUR 检测器。stacker 权重须在 **D2 时间 held-out** 上拟合(借鉴 hw01 `STACK_CALIBRATOR_WEIGHT=0.7`)非重叠窗。依赖 D1/D2 先落地 | **P2** |
| D6 | **(回喂生成器,单独标注)** 用 AUC 作**只读选型**信号 | 双重目标闭环 | eval-only(选型) | 先验证 held-out 切分:鉴别器"前 80% real vs 后 20% real"AUC 必须 ≈0.5;再对 12 模型选型看排序 vs 诚实诊断一致性 | held-out 自检 AUC≈0.5 + 选型排序 vs 峰度/wasserstein 的 Spearman | ★**最高泄露风险**。**硬前置门:必须 D2 held-out 自检 AUC ∈ [0.45,0.55] 才允许启动,否则 reject**。永不做可微损失、永不在 featurize 空间 reweight(项目红线);仅 `scoreboard` 的 `in_forensic_auc` 只读列。选型须 AUC + 诚实诊断**双过滤**防选到检测盲区。鉴别器 train 用前 80% 窗、生成器 eval 用后 20% 窗 + buffer | **P2** |

### 方法学护栏(method 视角,贯穿两 Track)

| # | 方向 | 治哪 | 成本 | 北极星 / 验证 | 优先级 |
|---|---|---|---|---|---|
| M1 | **realism vector 否决式判定** 替代 composite 单标量(新建 `eval/realism_board.py`) | 北极星本身被 Goodhart(composite 70% 权重在新颖/抗复制轴 = line2 北极星) | post-process(零训练) | 见 §5 三栏栏 1。零训练先对 5 候选验证能复现已知排序 | **P0** |
| M2 | **all-vs-novel 污染 gap**(容许复制安全边界) | line1 容许复制→真实度被记忆化污染而不自知 | post-process(零训练) | 见 §G6/§5。对 v10 重训(53.9% 复制)预期暴露 novel 子集峰度/wasserstein 退化;real-vs-real(0% 复制)gap≈0 阴性对照 | **P0** |
| M3 | **防 GAN 泄露协议**(时间 held-out + 缓冲带,否则只离线不回喂) | 2538 重叠窗 + 58M 参数下任何回喂必泄露 | eval-only | 判据=**Δ(无缓冲−有缓冲)c2st 显著收窄**(泄露被消除),残余 >0.5 归因真实时间非平稳并**保留**(勿强压到 0.5,那是 Goodhart) | **P1** |
| M4 | **scoreboard 升级三栏并报**(无主次) | `PRIMARY_METRIC=external_fool` 方向性错误 | post-process(零训练) | 见 §5。任一栏不得 override 另两栏;栏 2/3 异常只触发"解释"不触发"返工" | **P1** |

---

## 3. 廉价先行 —— 零训练快赢(一两天内全部落地,绝不烧 GPU)

### 3.1 numpy 峰度上界扫描(G1 go/no-go,**已跑**)
对 `train_sp500_us10y.csv` 的 `sp500` 列**直接 z-score**(勿 diff/log)后扫 clip:
```
clip{8,12,15,20} → excess_kurt{7.92, 10.98, 12.90, 18.80}    # clip8 精确=v10 实测 7.7
```
**结论铁证:CLIP_RANGE 是 ch0 峰度纯数据管道天花板,与损失/采样/架构无关。** clip15 可达 12.90。
ch1(DGS10 diff)clip12 即 plateau(~6.76)→ ch1 保 8。

### 3.2 DGS10 量化后处理(G2/D1,**已跑,可立即对现有 CSV 验证**)
- `config.py:134 DGS10_QUANTIZE=None` 已定义但 **`generate.py` 全仓零处接线 = 死旋钮**(`grep` 确认)。
- 零训练后处理:对现有任一生成 CSV 的 dgs10 列做 `np.round(x/0.01)*0.01`(在**反归一化后、真实量级**空间,**非标准化空间**)。
  实证:v9 `on_grid 0.0009→1.0`,扰动 mean 0.00058 / max 0.005(半格内),wasserstein 仅 0.0003 冲击、峰度零破坏。
- 一条 CSV <10s。后处理后跑 `diagnostics.py` + `c2st/signature` 对比量化前后,确认 DGS10 通道距离下降且不破坏 SP500 通道。
- ⚠️ **数字纠错(对抗校验抓出的硬伤)**:真实 DGS10 diff `frac_on_grid(0.01) = 1.0`(100% 在网格),**不是**某些草稿里反复写的 `0.0814`(那是 std 0.0917 或 zero-diff-frac 0.0666 的混淆)。**realism vector 的 `frac_on_grid` 自检带目标 = 1.0**,定错会制造假 FAIL 或放过未量化伪造。

### 3.3 给 c2st 加 DGS10 指纹 → 重排 12 模型(D1,**已跑**)
仅 3 维量化特征 LogReg 在 real vs deep_v10 → **AUC=1.0**(完美分离),real-vs-real 标定 0.537。
在 `eval/c2st.py` 的 `featurize()` 末尾对通道 1 追加 3 维:
```python
d = x[1]                                   # dgs10 diff 通道(真实量级)
rounding_ratio_0.01 = mean(abs(d*100 - round(d*100)) < 1e-8)   # 真实=1.0 / fake≈0
zero_diff_ratio     = mean(abs(d) < 1e-12)
unique_value_ratio  = nunique(round(d,8)) / L
```
`forensic_suite.py` 复用 `c2st.featurize` 自动继承。重算 12 模型,看 C2ST_acc 是否全升到 ≈1.0 且标定仍 ≈0.5。

### 3.4 中间 ckpt 轨迹(M1 铁律,零训练判)
重训前**永远**先拉 v10 中间 ckpt(`logs/deep_v10_dit_b/checkpoint_final.pt` + `scaler.pt`)在 ep500/1000/1500 各生成 1 批,用 `diagnostics.py` 看目标指标是否随 epoch / clip 抬升。**质量 ep~1500-2000 即饱和** → 重训只需 ~1500-2000ep,省半;峰度若全程死锁 ~8 则非 epoch 病(clip 天花板),加 epoch 无效。

### 3.5 realism vector + 污染 gap + 三栏记分牌基线(M1/M2/M4)
新建 `eval/realism_board.py`,对 hw01 share 里现有 5 候选 CSV(v9 / v10 采样 / v10 重训 / v11 / v12)跑六族否决式向量 + all-vs-novel gap,固化 real-vs-real 自检带,改 `eval/scoreboard.py` 为三栏。**先复现已知排序**(v10 重训峰度 7.7 最接近真实、波动聚集最强;若 real-vs-real 自己落不进自检带 → 带定义 bug,当场暴露)。

---

## 4. `configs/line1.py` 修正建议(真实度 recipe,替换旧 fool-recipe)

> 当前文件每一条 OVERRIDE 都与修正后北极星 180° 反,**必须整体重写**。
> 这是唯一的 full-retrain(P2),且重训前用 §3.1 / §3.4 廉价 go/no-go。

```python
"""configs/line1.py — 线1【真实度优先 / 直追真实市场 / 同时增强 OUR 鉴别器】配方(修正版)。

承 v9/v10 全窗(L=2048)基座,容许复制(不缩窗、不罚记忆)。放开 CLIP 峰度天花板 + 开 stylized
辅助攻波动聚集 + DGS10 量化吸附补真实结构。主记分牌 = realism vector 否决式判定(§eval/realism_board.py)。
hw01 fool 仅作外部参照并报,绝不优化。依据见 eval/docs/line1_realism_direction.md。

红线:禁 cosine/zero-SNR(打崩尾部)、禁 latent mixup(削尾)、禁把任何鉴别信号回喂训练损失
(除非先过时间 held-out + 缓冲带闸,见 §3 防 GAN 泄露协议)。
"""
OVERRIDES = dict(
    LINE="line1",
    PRIMARY_METRIC="realism_vector",  # ← 改:否决式真实度向量(eval/realism_board.py),非 external_fool
    SEQ_LEN=2048, STRIDE=5,           # 全窗,承 v9/v10 主线,容许复制不缩窗
    CLIP_RANGE=15.0,                  # ← 改:放开 ch0 峰度天花板(clip8→7.92 / clip15→12.90 逼近真实 18.8)
    #                                    ⚠️ 若实现 per-channel clip,则 ch0=15 / ch1=8(DGS10 clip12 即 plateau)
    USE_AUX_LOSS=True,                # ← 改:开 stylized |r|-ACF + 二阶差分能量,攻波动聚集/regime
    #   配套(config.py 已有): AUX_ACF_WEIGHT=0.05 / AUX_ROUGH_WEIGHT=0.05 / AUX_ABAR_MIN 门控
    #   新增 kurt 项: AUX_KURT_WEIGHT(对全量真实四阶矩常数,非 in-batch;见 §G3)
    USE_MIN_SNR=True,                 # 保留
    USE_CONTEXT_COND=False,           # 走 v9/v10 路线,不用富条件(那是 line2)
    USE_BLOCK_BOOTSTRAP=False,
    DGS10_QUANTIZE=0.01,              # 接线 generate.py(§3.2),补真实量化结构;非真实度排序主因
    NUM_EPOCHS=2000,                  # ← 改:M1 实证质量 ~1500-2000ep 饱和,省半(旧 20000 充分过平滑无用)
    # 容许复制:不设复制率阈值、不进判定门;但 realism_board 并报 all-vs-novel 污染 gap(§M2)
)
```

**接线工作量提醒**:
- `CLIP_RANGE` 写在 `dataset.TimeSeriesScaler.transform`;`train.py` 辅助损失内 x0_hat 钳位 `CLIP_RANGE*3`(15*3=45,远超数据范围,无害);GEN 端 `inverse_transform` 不反 clip → **只需改训练侧**。
- `DGS10_QUANTIZE`:`generate.py`(inverse_transform 后、to_csv 前)+ `generate_autoregressive.py` 两处补 `if config.DGS10_QUANTIZE: x_real[:,1,:]=round 到网格`。
- kurt 项:`losses.py stylized_aux` 加 `kurt_l = |((z**4).mean()-3) - REAL_KURT_CONST|`(z=标准化 x0_hat),`train.py` 接线 `loss_kurt = wabar*kurt_l*AUX_KURT_WEIGHT`。
- per-channel clip(可选):`TimeSeriesScaler.transform` 改为按通道用不同 clip 值。

---

## 5. 防 Goodhart / 防 GAN 泄露护栏 + 三栏记分牌

### 5.1 防 Goodhart 护栏
1. **realism vector 否决式,永不压成单标量**(M1):六族任一维落出 real-vs-real 自检带 2x 即该维 FAIL,**不允许其他维补偿**。`forensic_suite.py:248` 的 `composite = 0.30·c2st_novel + 0.25·sig_novel + 0.30·stylized + 0.15·anti_mem`(实测确认权重)= **70% 在新颖/抗复制轴 = line2 北极星**,对 line1(容许复制)是**错的主指标**且单一加权和必被 game → line1 弃用 composite。**代码注释 + scoreboard 固化:严禁任何下游脚本对 realism vector 加权求和当优化目标。**
2. **自检带从 train CSV 时间非重叠半切自举**(前 70% / 后 30% 窗,**非随机切**,否则滑窗重叠污染带宽);加单测断言 real-vs-real 每维落带内,落不进当场报 bug。
3. **容许复制的安全边界**(M2):复制率**只监控不约束**(`copy_rate / copy_rate_p99` 作监控列,绝不进判定门、绝不作优化目标);但每维真实度**同时在全集和 novel 子集各算一份**(novel = `novelty_rerank.py` 复制筛选后非复制子集),报**污染 gap**。若某维 all 落带但 novel 落不进 → 标 **`RESTING ON COPIES`** 红旗(该维真实度是抄来的,不算数)。novel 子集小样本下用 bootstrap 置信区间 + 报 `n_novel`,`n_novel<200` 标 `inconclusive` 防小样本假阳。
4. **DGS10 量化不计真实度排序主因**(G2/D1):它只补真实结构 + 喂 OUR 鉴别器,且易被 `np.round` 绕过 → 鉴别器里仅作低成本筛、非主判别信号。

### 5.2 防 GAN 泄露护栏(M3,红线)
- **线 A(回喂安全)**:任何进入训练损失的鉴别信号,真实数据须按**时间**切(前 70% 训生成器+鉴别器,后 30% 永不入训只验),切点两侧各留 SEQ_LEN **缓冲带**丢弃(消除滑窗重叠泄露)。
- **线 B(评估不回喂)**:`c2st / signature` 永远只离线打分,绝不入损失(固化现状为红线)。
- **线 C(借鉴 hw01 不抄判别器)**:可吸收 hw01 的**特征工程**(量化指纹 / TCN-InfoNCE embedding 思想 / 风格事实)进 OUR `featurize`,但**绝不把 hw01 判别器本身当目标**(line1 北极星已排除 fool)。
- **泄露探针正确判据**(校验修正):判据 = **Δ(无缓冲−有缓冲)c2st 显著收窄**(滑窗重叠泄露被消除),**不是**"有缓冲后绝对值→0.5"。金融时间序列时间非平稳,真实前 30% vs 后 30% 窗本就分布不同,c2st 可合法 >0.5;强压到 0.5 反而 Goodhart(逼鉴别器对真实时间结构变化视而不见)。
- `stylized` 辅助(`|r|-ACF` + 二阶差分能量 + kurt)是对 x̂₀ 自身统计正则、非判别器 → **无 GAN 泄露**;但目标统计量须用**全量真实常数**(非 in-batch,避关键发现 4 Sig-MMD 平凡满足)。

### 5.3 三栏记分牌(改 `eval/scoreboard.py`,无主次)
| 栏 1【line1 真实度】**唯一 go/no-go** | 栏 2【内部新颖度】仅监控 | 栏 3【外部 hw01 参照】仅交叉参照 |
|---|---|---|
| realism vector 六族 PASS/FAIL(M1)+ 污染 gap(M2) | `copy_rate / c2st_novel / sig_novel`(现有内部栏) | `fool_rate / mean_combined`(现有外部栏,**显式标注"非优化目标"**) |
- **三栏物理并排、任一栏不得 override 另两栏**;line1 候选**只看栏 1 判定**,栏 2/3 异常**只触发"解释"不触发"返工"**。
- 代码层硬断言:若有人试图把栏 3 `fool_rate` 写进 `PRIMARY_METRIC` 则**报错**,防回退旧 fool-recipe。
- 正交性自检:应出现"栏 1 真实度好但栏 3 hw01 fool 差"的格子且**不因此返工** → 证明没被 hw01 绑架。

---

## 6. 推荐执行序列

### 阶段一:零训练 P0(一两天内,绝不烧 GPU)
1. **§3.2 DGS10 量化后处理 + §3.3 c2st 加 3 指纹特征**(G2 + D1):对现有 CSV 后处理 → c2st 重算 12 模型 → 确认 AUC≈1.0 且标定 ≈0.5。
2. **§3.5 新建 `eval/realism_board.py`**(M1):六族否决式向量 + real-vs-real 自检带,对 5 候选复现已知排序。⚠️ `frac_on_grid` 目标用 **1.0**(非 0.0814)。
3. **all-vs-novel 污染 gap**(M2)接进 realism_board,v10 重训(53.9% 复制)验证暴露 novel 退化、real-vs-real gap≈0 阴性对照。
4. **新建 `eval/forensic_auc.py`**(D2):时间 held-out(前 80%/后 20% + 缓冲带 `ceil(L/stride)=410`)+ 标定 AUC 自检 ∈[0.45,0.55]。

### 阶段二:零训练协议 + 鉴别器 P1(并行)
5. **改 `eval/scoreboard.py` 三栏**(M4)+ 写防 GAN 泄露闸(M3),代码层断言禁 fool 当 PRIMARY。
6. **§3.4 v10 中间 ckpt 轨迹** + **G4 eta/steps 零训练扫描**:若纯采样把 `mean_run_len` 拉到 ~39 则 G6 长程辅助损失免做。
7. **D3 近邻特征**(带 guard,标注 NEVER feed generator)+ **D4 signature 多尺度 l_sub**(`n_perm≥300`)。

### 阶段三:按需 full-retrain(P0/P1 全部落地、go/no-go 通过后)
8. **重写 `configs/line1.py`**(§4)→ 先 §3.1 numpy clip15 go/no-go + §3.4 中间 ckpt 峰度饱和点。
9. **G3 消融臂**(USE_AUX_LOSS=True + clip 仍 8)隔离 AUX vs clip,再上 **G1+G3 合并**(clip15 + AUX,~2000ep 早停)。
10. 重训后**必跑 M2 污染 gap 复查**:novel 子集峰度/波动聚集是否也进自检带(防 clip15+L2048 只是抄更多真实极端窗充厚尾 = RESTING ON COPIES)。
11. **(可选)G5 v-pred**:仅当 clip15+AUX 后峰度仍 <12 且诊断指向末步均值化才上(先过数值单测)。
12. **(最高风险,最后)D6 只读选型回喂**:硬前置门 D2 held-out 自检 AUC∈[0.45,0.55] 才启动,永不做可微损失。

### 一句话:若只做一件事
> **在 `eval/c2st.py` 的 `featurize()` 末尾给 DGS10 通道加 3 维量化指纹特征(`rounding_0.01 / zero_diff / unique_ratio`)** —— 零训练、确定性、AUC 从未抓→1.0、不污染 real-vs-real 标定(0.537),一次性把 OUR forensic 鉴别器对所有扩散输出推到近完美分离,直接兑现双重目标的鉴别器那一半。(同一行后处理 `np.round` 接到 `generate.py` 又顺手补上 G2 的真实结构。)
