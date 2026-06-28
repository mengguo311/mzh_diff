# C1 改进与低成本验证 — 总分析报告

> 分析师综合三视角(divergence / quality / method)提案 + 对抗批判,经代码与数据实测核实。
> 日期 2026-06-24 · 分支 deep_v10 · 对照基线 `eval/forensic_out/c1_clean/forensic_c1_clean.json`
> 实测核实(本会话):checkpoint 5+final 在盘;loss ep4999=0.00061(单调降未收敛);`c2st.py:65` ku_s/ku_d 为直接判别特征;`signature.py` l_sub 默认 200(L=2048 长程盲区);`config.py:26` CLIP_RANGE=8.0;`generate_autoregressive.py:72` ctx 未 clamp;`forensic_suite.py:116` 发散界 = max(0.5, 30×real_std)、`--quick` 降 n_perm 300→不依赖 max_samples;C1 baseline:copy_rate **2.25%** / C2ST_新颖 **0.770** / SigP_新颖 **0.106** / n_div **2** / frac_div **0.039%** / kurt_fake **8.06** / d2_ratio 0.976x / patch 0.936x。

---

## 0. 执行摘要

**C1 现状一句话**:`deep_v13_c1_ctx`(USE_CONTEXT_COND, COND_DIM=16, DiT-S, L=512 训练 + 自回归 512→2048)已**机制性修好 A1 的两大硬伤**——长程(high_vol_frac 0.49≈真实 0.50、d2_energy 0.976x、|r|-ACF/leverage 曲线几乎重合)与签名半修复(SigP_新颖 0.106,从 A1 的 0.003 大幅回升但仍 < L=2048 标定带 0.326);**余两处病**:(1) 罕见 DDIM 单窗失稳 → 2/5118 行发散(仅 chunk3 爆,frac 0.039%);(2) **欠厚尾**:峰度 8.06 vs 真实 ~17–19、偏度 -0.25 vs -0.76,是 C2ST_新颖 0.770 FAIL 的**主驱动**(c2st 特征直接含 ku_s/ku_d)。复制率 2.25%(史上最低,真新颖)。

**核心结论(哪些方向能用零/低训练成本验证)**:
- **能零训练可靠验证(预测=部署同一改动)**:发散 reject/clamp(divergence 视角①②⑦)、采样旋钮扫描(eta/steps/k/w)。但这些**只回答"症状能否被推理侧消除",不回答"C1 是否更真"**——发散归零对北极星指标按定义无信息量(只动 0.04% 样本)。
- **能零训练高信息量验证**:**中间 checkpoint 指标轨迹**(method M1)——回答"最少需多少训练 / 是否仍在升 / 是否过拟合记忆化",是性价比最高的元方向。
- **欠厚尾(C1 真病)无法零训练根治**:它是 eps-MSE mode-averaging 的过平滑病根 + CLIP_RANGE=8 把 >8σ 极端日硬截断(实测 clip=8→真实 pooled kurt~9.7,clip=15→~18.4,**clip 是峰度天花板**)。推理旋钮(w/eta)只能锐化模型**自身(过平滑)流形**,大概率把"平滑 garbage"换成"毛刺/发散 garbage",**须全量/resume 重训**且廉价验证只能证"天花板存在",证不了"模型学得到"。
- **一句话**:**能用更低成本验证大部分方向(全部 inference-only + 一个零训练轨迹元方向),但"验证发散修复""验证旋钮"≠"验证 C1 达标";C1 真正达标(C2ST_新颖 0.770→PASS)必须正面攻厚尾,且最终决策点禁用低 N/quick、须全量 5120 + n_perm≥300 复测。**

---

## 1. C1 现状诊断与改进靶点

| 病灶 | 实测读数 | 是否 C1 真病 | 机制根因 | 治哪类成本 |
|---|---|---|---|---|
| **发散(罕见单窗 DDIM 失稳)** | 2/5118 行,仅 chunk3 爆(sp std max 14.81 vs 健康 0.021,rowmax 29.9/210.9);chunk0/1/2 std 分布完全相同且健康 | **否**(forensic 已逐行剔除,与生成质量无关) | DDIM ancestral 随机项(eta=1)在某窗 ~0.04% 概率把 x 推离流形;**非系统漂移、非接缝、非上下文反馈**(发散行 3838 chunk2 std=0.003 低波动却 chunk3 爆) | 推理侧零训练 |
| **欠厚尾 → C2ST_新颖 0.770 FAIL** | 峰度 8.06 vs 真实 ~17–19,偏度 -0.25 vs -0.76;逐 chunk 峰度 7.3/7.9/8.2 **均匀偏低**(非接缝伪影) | **是(头号)** | (a) eps-MSE mode-averaging + DDIM 末步均值化的过平滑(关键发现1);(b) CLIP_RANGE=8 训练时硬截断 >8σ 极端日(实测真实 max\|z\|≈17σ、14 天 >8σ),模型从未见 → 峰度结构性封顶 ~8 | **全量重训为主**;采样旋钮只能微调 |
| **SigP_新颖 0.106(签名半修复,WARN)** | 0.106 vs L=2048 标定 0.326,A1 仅 0.003 | 是(次要) | 部分由厚尾不足(签名对增量分布敏感);**l_sub=200 子路径几乎不跨 512 接缝 → 对长程结构盲**,故 0.106 可能"短子路径像真、长程没测出" | 与厚尾耦合;接缝侧推理可试但易过平滑 |

**判读**:发散是卫生问题(降级);**C1 是否达标取决于厚尾**;SigP 与厚尾耦合且其度量本身有长程盲区,不可单独作北极星。

---

## 2. 改进方向全表(三视角合并去重,按 validity×低成本 排序)

去重映射:divergence①≈ 卫生 reject;②/⑦(scheduler)= clamp;③ = ctx-clamp(因果实验);④/E/M2-eta、⑤/M2-steps、quality-E/M2-w = 采样旋钮;⑥/C = overlap 接缝;A = 放宽 clip;B = ctx 特征工程;F/M1 = checkpoint 轨迹;M3 = resume;M4 = 减 epoch 代理;M5 = 稳定性标定护栏。

| 方向 | 治哪个病 | 验证成本类别 | 最便宜验证 | go/no-go 阈值 | 批判后 predictive_validity |
|---|---|---|---|---|---|
| **M5 稳定性标定护栏** | 元/护栏(定可信 N、n_perm 下限) | 推理(~1–1.5h) | 一次 N=5120 候选 × max_samples{256..5120} × quick/全量,记各指标均值±std | 输出"信号必须 > 此 std" 表;SigP 须 n_perm≥300 全量 + N≥2048,否则不采信 | **strong**(但须修两洞:N vs n_perm 混淆、l_sub=200 长程盲) |
| **M1 中间 checkpoint 轨迹** | 元(最少训练量/过拟合记忆化) | 推理(~30–40min) | 5 ckpt 各 N=1024/k4/steps200 生成+forensic,画 copy_rate/C2ST_新颖/峰度/high_vol_frac vs epoch | 证伪用:ep4999 copy_rate↑且 C2ST_新颖不更差 → 缩 epoch 免费赢;平台化(Δ<std)→ 调 epoch 无用 | **plausible**(证伪强、证实弱;Δ0.02 阈 < 自身噪声 0.063,须多 seed) |
| **① reject-resample** | 发散(卫生) | 推理(~40min) | 内层循环加坏行重抽,生成+forensic | n_div 2→0 且北极星不退化(±std) | **plausible 但降级为工程卫生**(对北极星无信息量) |
| **⑦/② x0-clamp(scheduler/生成窗)** | 发散(卫生)+ A 前置安全网 | 推理(~1h,改 1 行) | scheduler x0_pred 后加 clamp(±18–20σ z 空间)+ 生成窗 clamp,重生成 | n_div→0 且**逐行**查被 clamp 2 行无毛刺;干净指标偏移 <0.01 | **strong(改动=验证=部署同一)** 但**不提分只解锁 A** |
| **M2/E/④⑤ 采样旋钮(w/eta/steps/k)** | 欠厚尾(试)+ 发散(试) | 推理(~30–60min 网格) | w{1,1.5,2}×eta{0.8,1}×steps×k 小 N 扫,forensic | **四联**:kurt↑向真实 AND C2ST_新颖**实质下降** AND d2∈[0.9,1.2]x AND frac_div<1%;入选点全量 5120 复测 | **weak**(w/eta 放大自身过平滑流形=换 garbage 高危;峰度⊥C2ST 判据自相矛盾) |
| **③ ctx-clamp** | 发散因果判真伪 | 推理(~40min) | **确定性 seed 重放**那 2 条发散链开/关 ctx-clamp | 确定性阻止爆炸=证实反馈;N=2 计数判因果**无统计功效** | **plausible(仅因果)** 改确定性重放才有效 |
| **⑥/C overlap-averaging 接缝** | SigP(试) | 推理(~40min+实现) | 现 CSV 做 overlap 平滑重跑 forensic;补 FFT bin4/8/12 哨兵 | SigP↑ AND d2 不低于真实 0.95x AND high_vol_frac 不跌 AND 峰度不 <7.5 | **weak**(清洗后接缝已 0.98x/0.94x 达标;overlap=凸组合过平滑高危) |
| **A 放宽 clip(8→15)** | **欠厚尾(头号根因)** | 全量(12.5h)+ 前置 resume | numpy 零成本上界(已复现:clip15 真实 kurt~18.4 vs clip8~9.7);**强制 resume 短续训中间闸门** | go=clip15 真实 kurt≥15 ✓;重训后 kurt≥12 **AND** C2ST_新颖≤0.65 **AND** frac_div<1%(三联) | **strong 但仅证天花板存在**,证不了模型学得到尾部(mode-average 0.2% 极端样本) |
| **G x0 钳位(=⑦,quality 视角命名)** | 发散 + A 前置 | 推理(~1h) | 同⑦ | 同⑦;阈值按实测 17σ 而非 19.4σ,确认 z 空间口径 | **strong(最干净)** |
| **B ctx 特征工程(补尾部维度)** | 欠厚尾(叠加) | 全量(12.5h) | 改"条件干预消融":放大喂回 ctx 的 kurt/std,看生成峰度是否响应 | 生成峰度对条件峰度敏感 → 值得;纹丝不动 → 否决(省 12.5h) | **weak**(R² 验证=Sig-MMD no-op 同型陷阱:信息存在≠模型用) |
| **M3 resume 微调** | 暴露偏差/发散(clamp 型) | resume(~1.5–3h) | 从 final resume +500–1000ep 带改动,forensic 对比 | 向 C2ST PASS 实质移动 AND copy_rate 不升 >0.03 | **weak**(resume 续训=继续压 loss=加重记忆化,污染归因;不可外推到从头训) |
| **M4 减 epoch 代理(改动+1500ep 从头)** | 数据侧改动(A2/B1 等需从头) | 全量代理(~4h) | M1 先证 1500–2000ep 是有效代理点 → 带改动 1500ep,**等-epoch** 对比 | 等-epoch 下向 PASS 移动且无其他北极星恶化 | **plausible(方法最干净:等 epoch+从头)** 但成败压在"代理点成立",增广收益偏后期易假 NO-GO |

---

## 3. 【核心】分层成本验证路线图 Tier 0→3

### Tier 0 — 推理侧零训练(分钟–1h):发散修复 + 旋钮扫描
**回答什么问题**:(a) 发散能否被推理侧消除(卫生);(b) 厚尾能否靠纯采样救(若能则省 A 的 12.5h)。
**时间成本**:reject/clamp 各 ~40min–1h;旋钮网格 ~30–60min。2×A6000 空闲可并行。
**命令(用 checkpoint_final.pt 重生成 + forensic)**:
```bash
# ⑦ x0-clamp(改 scheduler.py x0_pred 后加 clamp ±18σ z 空间 + generate ctx clamp)
nohup conda run --no-capture-output -n ts_diffusion python -u generate_autoregressive.py \
  --model dit-s --checkpoint logs/deep_v13_c1_ctx/checkpoint_final.pt \
  --scaler logs/deep_v13_c1_ctx/scaler.pt --num_samples 5120 --k 4 --seed_ctx real \
  --output output/c1_ar2048_clamp.csv > logs/gen_c1_clamp.log 2>&1 &
conda run -n ts_diffusion python forensic_suite.py --candidate output/c1_ar2048_clamp.csv --label c1_clamp
# 旋钮扫描(小 N 粗筛)
for W in 1.0 1.5 2.0; do for ETA in 0.8 1.0; do
 conda run -n ts_diffusion python generate_autoregressive.py --model dit-s \
  --checkpoint logs/deep_v13_c1_ctx/checkpoint_final.pt --scaler logs/deep_v13_c1_ctx/scaler.pt \
  --num_samples 512 --k 4 -w $W --eta $ETA --num_inference_steps 200 --seed_ctx real \
  --output /tmp/c1_w${W}_e${ETA}.csv
 conda run -n ts_diffusion python forensic_suite.py --candidate /tmp/c1_w${W}_e${ETA}.csv \
  --label c1_w${W}_e${ETA} --no-fig --max-samples 512 --quick
done; done
```
**预测有效性与假信号防护**:
- 发散修复的 predictive_validity = **strong（改动=验证=部署同一）**,但**只预测"发散行消失"这个与质量无关的事实**——把"发散归零"当 C1 胜利 = ddpm_mse 式自指新马甲(批判一致指出)。降级为工程卫生开关,**不计入 C1 达标**。
- 旋钮扫描 **weak**:w/eta 放大模型自身过平滑流形,极易**把欠厚尾 garbage 换成毛刺/发散 garbage**。防护:(1) 删除"峰度↑不升 C2ST"的自相矛盾约束(峰度是 C2ST 判别特征,二者耦合),改为"C2ST_新颖**实质下降**向 PASS 0.595 移动"作唯一峰度 go;(2) 峰度↑必须联查 d2_energy∈[0.9,1.2]x + patch_spike + 发散哨兵,排除假厚尾;(3) **frac_div 必须 N≥5120 全量判**(N=512 期望命中 <0.2 条,"0 发散"是没抽到);(4) 报告**剔除发散行前后峰度差**,防幸存者偏差撑高峰度;(5) force_null 臂对照确认改善非来自 context 失效。

### Tier 1 — 中间 checkpoint 指标轨迹(零训练,~30–40min)
**回答什么问题**:C1 最少需多少训练?是否仍在升?是否已过拟合(复制率随 epoch 上升=记忆化)?给 Tier 2/3 校准 baseline。
**时间成本**:5 点 × (N=1024/k4/steps200 生成 ~2min + forensic ~3min) ≈ 30–40min 串行,双卡 ~20min。
**命令**:
```bash
for E in 0999 1999 2999 3999 4999; do
 conda run -n ts_diffusion python generate_autoregressive.py --model dit-s \
  --checkpoint logs/deep_v13_c1_ctx/checkpoint_epoch_${E}.pt --scaler logs/deep_v13_c1_ctx/scaler.pt \
  --num_samples 1024 --k 4 --num_inference_steps 200 --seed_ctx real --output /tmp/c1_traj_${E}.csv
 conda run -n ts_diffusion python forensic_suite.py --candidate /tmp/c1_traj_${E}.csv \
  --label c1_traj_${E} --no-fig --max-samples 1024
done   # 画 copy_rate_p95 / c2st_novel / kurt_fake / high_vol_frac vs epoch
```
**预测有效性与假信号防护**:
- **证伪用途强**(同权重族零训练同口径,无外推):若 ep4999 copy_rate 明显高于早期点且 C2ST_新颖不更差 → "缩 epoch 免费赢"可信,直接复现关键发现4。
- **三个陷阱**(批判):(1) M5 实测 C2ST_新颖 小 N 抖动 0.707↔0.770 = **Δ0.063 > 平台/回退判据 Δ0.02 一个量级**,**必须每点 3–5 seed 子采样取均值±std**,阈值 > M5 实测 std;(2) **EMA 权重轨迹比 raw loss 平滑**,loss 单调降(实测 ep4999=0.00061 未收敛)不蕴含 EMA 样本回退 → 看不到回退≠没有,**须并记 raw 权重同口径指标对照**;(3) copy_rate 小 N 是稀有事件(N=1024 的 2.2%≈23 条,Poisson ±0.5%),拐点须 N≥2048 复核。
- **外推边界**:M1 只回答 epoch 维度,**不外推到 A/B 机制改动后的从头训练**(不同 loss landscape)。SigP 在低 N/quick 下结构失真(M5 实测 real-real 标定塌到 0.046 vs 0.326),**轨迹判读一律不用 SigP**,北极星限 copy_rate/C2ST_新颖/峰度/high_vol_frac。

### Tier 2 — resume 微调(~1.5–3h):仅限"可 resume 见效"的改动
**回答什么问题**:一个针对性改动(ctx-clamp 防爆型 / 跨接缝)+500–1000ep 续训能否显现可分辨正向增量。
**时间成本**:+500ep ≈ 75min(实测 ~9s/ep),+1000ep ≈ 2.5h,加评估 ~30min。
**命令**:
```bash
nohup conda run --no-capture-output -n ts_diffusion python -u train.py \
  --resume logs/deep_v13_c1_ctx/checkpoint_final.pt --epochs 5500 --run_dir logs/deep_v13_c1_ft \
  --<改动开关> > logs/c1_ft.log 2>&1 &
# 完后 generate_autoregressive(改动臂) + forensic --max-samples 2048
```
**预测有效性与假信号防护(weak,严格限定)**:
- **外推断裂**(批判审问 c 正中要害):resume 从已深度过优化权重(loss 0.0006)续训 = **继续压 loss = 加重记忆化**,任何"改善"都纠缠 copy_rate↑负效应,**无法干净归因于改动**;从头训走完全不同优化轨迹,resume 小增量不能预测从头训方向。
- **只对"推理可先验的防爆型 clamp"有意义**——而那类本就该先走 Tier 0 推理侧验证,不必动训练。
- "自采样 ctx/改数据分布"这类**重塑表征**的改动,**resume 不可外推,必须走 Tier 3 M4**。
- 防护:(1) resume 臂**必同报 copy_rate**,copy_rate↑则改善作废;(2) **frac_div 绝不用 resume 臂小 N 判**(N=2048 期望命中 <1,"降到 0"是假信号);(3) go 门槛从"可分辨增量"上调到"**向 C2ST PASS(0.595)实质移动**"。

### Tier 3 — 减 epoch 代理 / 全量(仅当 Tier 0–2 指向正收益才投入)
**回答什么问题**:必须从头训的改动(A 放宽 clip / A2 block-bootstrap / B1 多资产)在等-epoch 预算下方向是否正确;A 的厚尾天花板模型能否学到。
**时间成本**:M4 代理 1500ep ≈ 3.5–4h + 评估;A 全量 5000ep ≈ 12.5h + 生成 40min + forensic 15min。
**命令**:
```bash
# A 零成本上界(已复现): clip8→真实 kurt~9.7, clip15→~18.4 → 天花板存在
# M4 代理(M1 先证 1500-2000ep 为有效代理点后)
nohup conda run --no-capture-output -n ts_diffusion python -u train.py \
  --epochs 1500 --run_dir logs/deep_v13_X_proxy --<改动开关> > logs/X_proxy.log 2>&1 &
# 等-epoch 对比 C1 ep1999 轨迹点(M1 已产)
```
**预测有效性与假信号防护**:
- **A 仅证天花板存在,证不了模型学得到**(批判核心):解封 14 个 >8σ 样本占训练集 0.2%,eps-MSE/min-SNR + DDIM 末步均值化极可能 mode-average 掉 → 重训峰度可能只 8→9–10,远不到 go 闸门 13。**强制中间闸门**:先 resume 短续训(放宽 clip + 把极端窗高频重采样进 batch)看 kurt 是否对"见过尾部"有响应,再上从头 5000ep。**A 必先落地 ⑦ x0 钳位防发散飙升**。三联 go:kurt≥12 AND C2ST_新颖≤0.65 AND frac_div<1%。
- **M4 plausible(方法最干净:等 epoch + 从头,规避 resume 外推)**,但成败压在"1500ep 是有效代理点":loss 在 ep1500 才 0.0039、到 ep5000 还降 5 倍,**修复可能集中在后期低 loss 区** → 代理点很可能不成立;**对 A2/B1 增广类(收益偏后期),代理点须上调到 2500–3000ep**,否则系统性低估、假 NO-GO。门控:M1 须实证 1500–2000ep 处指标达 ep5000 的 ~90% 才启用。
- **B 弱**:R² 验证是"信息存在≠模型用"陷阱(Sig-MMD no-op 同型),改用"条件干预消融"(放大喂回 ctx 的 kurt 看生成峰度是否响应),不敏感则直接否决省 12.5h;B 永远只作 A 叠加,且在 A 证"解封 clip 有响应"之后。

---

## 4. 廉价验证的【预测有效性与陷阱】

**(a) 减 samples / 减 steps / 减 n_perm 的噪声下限**:
- **SigP 是重灾区**。代码核实:`signature.py` k_sub=2、l_sub 默认 200、n_perm 默认 300;`forensic_suite.py` 标定 c2_cal/sig_cal **只依赖 n_sig/n_perm,不依赖 --max-samples**(批判修正 method M5 的"低 N"混淆)。M5 实测 N=256/steps50/**quick** 下 SigP real-real 标定塌到 0.046(全量 0.326)、SigP_新颖 0.536(全量 0.106)——**真凶是 --quick 降 n_perm,不是样本数**。结论:**SigP 须 n_perm≥300 全量 + N≥2048 才可信**;任何方向仅靠 SigP 翻盘且未全量复现 = 假信号 NO-GO。
- 更深盲区:**l_sub=200 子路径几乎不跨 512 接缝**,SigP 对跨窗长程结构**结构性失盲** → C1 SigP 0.106"部分修复"可能"短子路径像真、长程没测"。即便 N/n_perm 拉满,SigP 仍不能背书长程修复;须补 l_sub 增大(如 800)或专门跨接缝诊断。
- **C2ST_新颖**相对稳但仍有 0.707↔0.770 = **Δ0.063 抖动**(平衡子采样到 min(n_fake, 2538));copy_rate 小 N 稀有事件 Poisson 抖动 ±0.5%;**峰度对 N 不敏感**(7.9↔8.1)是廉价旋钮判读最可靠的轴;high_vol_frac 稳(0.49↔0.50)。

**(b) 推理结论能否外推到重训**:
- **inference-only 方向(reject/clamp/旋钮/k)无"增量外推到从头训"问题**(改动=验证=部署同一)——但其结论只对**推理侧决策**有效;k 推理扫描**不外推**到"针对该 k 重训"(重训会重塑上下文适应)。
- **resume(Tier 2)外推弱**:续训纠缠记忆化 + 不同优化轨迹;只对"防爆型 clamp"勉强可信。
- **减 epoch 代理(Tier 3 M4)**:同配方延长可外推;改 clip/条件维度的从头训**不可由 M1/M4 旧轨迹外推**(不同 landscape)。

**(c) 如何防"修了发散却换成另一种 garbage"**:
- **forensic_suite 哨兵 + 诚实闸门的守与漏**:`divergence_guard` 的 corrupted 判据(bound=30×real_std、d2_ratio、std_ratio、kurt)**全是 MEAN-based**,实测 dg 通道 d2_energy fake_mean 5.46 完全由 2 行驱动(median 0.0202 健康)。**剔除发散行后这些 mean 恒绿**——会**掩盖 clamp/overlap 在被处理少数行引入的边界毛刺**(只在 2 行、被 mean 稀释)。
- **加固守门**:(1) clamp/overlap 必须**逐行/逐位置诊断**被处理的那 2 行(N=2 逐行 d2/patch),并在**不剔除发散行口径**下另跑一次 forensic,才看得到"garbage 换 garbage";(2) 补 **FFT bin4/8/12 哨兵**(当前 patch_spike 只探 bin128 patch-16 谐波,探不到 512 边界谐波);(3) overlap 过平滑表现为**峰度/尾部下降而非 patch_spike 升**,须把峰度/QQ 尾部纳入 go/no-go;(4) 北极星统一:**任何修发散方向必须 copy_rate / C2ST_新颖 / 峰度 / high_vol_frac 不退化(各按 M5 实测 std 而非 ±0.02 硬阈)**;(5) **最致命**:低复制率 2.25% 是"不抄"的真新颖**还是**欠厚尾的平滑 garbage(C2ST_新颖 0.770 FAIL 正印证)——`divergence_guard` 只是 30×std 物理界,**不是新颖性/真实性闸门**,推理侧修发散更可能只是把爆值 garbage 换成平滑 garbage,**C1 达标须正面攻厚尾,本路线图 Tier 0–2 无一能做到**。

---

## 5. 推荐执行序列(最低成本最高信息量优先)

1. **M5 稳定性标定护栏**(~1–1.5h,零训练,**强制最先做**):一次 N=5120 候选 × max_samples{256,512,1024,2048,5120} × quick/全量(并拆 n_sig/n_perm 二维),记各指标均值±std。**预期产出**:可信 N/n_perm 下限表 + 各北极星复现 std。**go/no-go**:SigP 须 n_perm≥300 全量 + N≥2048;C2ST_新颖/copy_rate/峰度/high_vol_frac 的 std 写进后续判据下限;补一条 l_sub=800 长程护栏。**不过此关,后续一切 SigP 读数作废。**

2. **M1 中间 checkpoint 轨迹**(~30–40min,零训练):5 ckpt × N=1024 × **3–5 seed** + forensic,画 copy_rate/C2ST_新颖/峰度/high_vol_frac vs epoch,并并记 raw(非 EMA)对照。**预期产出**:回答"最少训练量 / 是否过拟合记忆化 / epoch* 是否存在"。**go/no-go**:Δ 须 > M5 实测 std;若 ep≥3000 平台化 → 调 epoch 无用,任何改进须来自旋钮或训练侧改动;若 copy_rate 随 epoch 升 → 缩 epoch 免费赢。

3. **⑦ x0 钳位(scheduler + ctx 喂回 clamp)**(~1h,零训练,改 1 行):阈值按实测 17σ 设(z 空间)。**预期产出**:发散卫生归零 + A 的前置安全网。**go/no-go**:n_div→0(N=5120 全量)且**逐行**查被钳 2 行无毛刺、干净指标偏移 <0.01。固化进 generator/scheduler 作默认开关,**不计入 C1 达标**。

4. **M2/E 采样旋钮粗筛**(~30–60min,零训练):w{1,1.5,2}×eta{0.8,1}×steps×k 小 N 网格。**预期产出**:判"纯采样能否救厚尾"——若能则省 A 的 12.5h。**go/no-go(四联,入选点全量 5120 复测)**:kurt↑向真实 AND C2ST_新颖实质下降向 0.595 AND d2∈[0.9,1.2]x AND frac_div<1%;报剔除前后峰度差防幸存者偏差。预期**大概率 NO-GO**(欠厚尾是权重病非采样病)。

5. **(可选)③ ctx-clamp 确定性重放**(~40min):仅对那 2 条确定发散链开/关 ctx-clamp,**确定性**判因果(非 N=2 计数)。**预期产出**:证实/证伪"极端 ctx→失稳"反馈(诊断已强烈反对,大概率证伪)。

6. **若 Tier 0–2 指向"必须重训":A 放宽 clip(8→15)**——先 numpy 上界(已复现 ✓)→ **resume 短续训中间闸门**(放宽 clip + 极端窗高频重采样,看 kurt 是否响应)→ 仅响应才上从头 5000ep(~12.5h);**A 前必落地 ⑦**。**go/no-go**:kurt≥12 AND C2ST_新颖≤0.65 AND frac_div<1%(三联)。**B 仅作 A 叠加**,且先做"条件干预消融"判生成峰度对条件是否敏感。

7. **(数据分布改动如 A2/B1)M4 减 epoch 代理**(~4h):M1 须先证代理点成立(增广类上调到 2500–3000ep),等-epoch 从头对比。**go/no-go**:等-epoch 下向 PASS 移动且无其他北极星恶化。

---

## 6. 一句话回答用户

**能。** C1 的发散修复、采样旋钮、k 扫描全部 **inference-only**(零训练,改动=验证=部署同一),再加一个零训练的**中间 checkpoint 轨迹**元方向(M1,~30min)就能回答"最少需多少训练 / 是否过拟合 / 该不该重训",**几乎所有方向都能在不投入 12.5h 全量重训前用分钟–小时级成本验证**。

**但有一条硬边界必须讲清**:**"低成本验证发散修复/旋钮"≠"验证 C1 达标"**——发散是与质量无关的卫生问题(把它归零当胜利 = ddpm_mse 式自指);C1 的真病是**欠厚尾**(峰度 8 vs 真实 ~18,驱动 C2ST_新颖 0.770 FAIL),它是权重/训练数据 clip 病而非采样病,**低成本验证只能证"天花板存在"(clip15→真实 kurt~18 已复现)、证不了"模型学得到尾部"**,根治须全量/resume 重训。且所有廉价读数的**最终决策点禁用低 N/--quick**(SigP 在 quick 下标定崩、l_sub=200 对长程盲),须全量 5120 + n_perm≥300 复测,信号须 > M5 实测复现 std 方可采信。

---

## 7. 实证结果 — 零训练验证战役(M5 标定 + M1 轨迹,2026-06-24)

> 执行 `eval/c1_validation_campaign.py`(复用 `forensic_suite` + 5 个中间 checkpoint,~1.5h,**零训练**)。
> 产物:`eval/forensic_out/campaign/{m5_calibration.json, m1_trajectory.json, m1_trajectory.png, summary.txt}`。

### 7.1 M5 稳定性标定护栏(20 配置网格,实测)
- **`--quick` 一律把 sig_cal 0.326 → 0.046**(全部配置);SigP_新颖 即使全量 n_perm 仍跨 seed 抖动 ±0.3(ms=512: 0.14–0.76;ms=2048: 0.13–0.84)→ **SigP_新颖 不是可信单值,只能判"塌没塌(<0.05)";禁用 quick,须全量 n_perm≥300 + N≥2048**。
- **C2ST_新颖(±0.02–0.03)/ 复制率(±0.005)/ 峰度(±0.05)稳定** → 低成本可信轴。

### 7.2 M1 中间 checkpoint 指标轨迹(mean±std / 3 seed,N=1024)
| epoch | 复制率 | C2ST_新颖 | 峰度 | high_vol | n_div/1024 |
|---|---|---|---|---|---|
| 0999 | **0.000** | 0.734±0.011 | 8.09 | 0.496 | 7 |
| 1999 | **0.000** | 0.743±0.011 | 8.08 | 0.499 | 16 |
| 2999 | 0.000 | 0.749±0.029 | 8.35 | 0.499 | 19 |
| 3999 | 0.006 | 0.749±0.005 | 8.05 | 0.488 | **1** |
| 4999 | 0.022 | 0.738±0.009 | 8.00 | 0.491 | **0** |

### 7.3 实证结论(四条铁律,全部坐实)
1. **质量 ep999 即饱和**:C2ST_新颖 / 峰度 / high_vol 全程不动 → 后 ~3000–4000ep(~8–10h)对质量零贡献。
2. **峰度全程死锁 ~8(真实~18),epoch 完全推不动** → **欠厚尾 = clip/损失的硬结构天花板,加 epoch 必然无效**,须走方向 A(放宽 clip)。
3. **复制率随训练上升(0%→2.2%)** → 最抗记忆的是**早期** checkpoint(ep999–2999 全 0%)。
4. **发散随训练下降(n_div 7→16→19→1→0)** → 晚期自回归最稳;早期爆窗多。存在权衡:早=更新颖但更易发散,晚=略记忆但更稳。

### 7.4 决策(基于实证)
- **C1 未来重训只需 ~1500–2000ep(~5h)而非 5000ep(12.5h),省 >一半**;质量在 ep1000–2000 已满且复制率仍 ~0%。
- **甜点配置**:ep~1500–2000(零记忆 + 质量满)+ ⑦ 推理钳位(零训练治发散,不必为降爆窗多训 3000ep)。
- **破峰度天花板须方向 A**(放宽 clip 8→15),已实证"加 epoch 无效";方向 A 先 numpy 上界(clip15→真实 kurt~18,已复现)+ 短 resume 廉价预判,再决定是否上全量。
- **"能否低成本验证改进方向"= 能,且本战役自身即证明**:5 个现存 checkpoint + ~1.5h 零训练,得出全部上述结论。
