> **⚠️ 实证更新(2026-06-28,见 `research/E0_results.md`)**:E0(FHS GJR-GARCH-t)+ E0b(component-GARCH)**已跑完并评分**。**关键负结果**:强计量先验作为**独立生成器**在 C2ST 主闸门上被**决定性反驳** —— FHS **C2ST_新颖 0.946**、CGARCH 0.944,而训练 DiT(clip11)**0.601**。复制 0.0%/真厚尾(kurt 28)/签名 PASS(0.166)/无接缝虽 training-free 可达,但 **GARCH 波动律(几何衰减)抓不住真实长记忆 → 深网在波动动力学上反而赢**。**故下方 E2(白化外包波动给更差的 GARCH)降级、E3(GARCH 合成 teacher 偏弱)需谨慎;最有据的剩余杠杆 = E4 加真数据(唯一治本)或给 DiT 注入厚尾噪声先验。** 详见 E0_results.md §5-7。

# Top 推荐 → 可执行的下一步实验序列

> 把 README §5 的 5 个 Top 推荐落成有序、可跑、用诚实闸门把关的实验。优先级 = **数据效率高 × 避记忆化 × 复用现有 DiT/signature/c2st/forensic_suite 基建**。
> **统一验收闸门(每个实验产 CSV → 同口径)**:`forensic_suite.py <候选CSV>`(复制率[vs real-vs-real 底噪 0.0%] + C2ST_新颖[vs 标定,L2048 ~0.496 / L512 ~0.496] + SigP_新颖[vs L2048 0.116/0.326] + 诊断[峰度/high_vol_frac/mean_run_len/d2_energy] + 逐行发散哨兵)。**SigP 须全量 n_perm≥300,`--quick` 毁 sig 标定。**
> **环境**:`conda activate ts_diffusion`;长任务 `nohup conda run --no-capture-output -n ts_diffusion python -u ... > logs/xxx.log 2>&1 &`。
> **依赖现状**:`arch` 未装(可 `pip install arch` 或纯 numpy/scipy 手撸);`torchsde/sigkernel/nflows/zuko/pyro/numpyro` 未装;`signature.py / losses.py(可微 depth-2 sig_mmd + recover_x0)/ forensic_suite.py / memorization.py / c2st.py` 已就位。

---

## 实验序总览(有序纲领:先验底座 → 破 floor → 抬分母)

| # | 实验 | 推荐源 | 成本 | 前置依赖 | 攻什么 floor |
|---|---|---|---|---|---|
| **E0** | FHS/GARCH-t 零训练强基线 | Top-1 | 极低(1-2 天) | numpy/scipy(已有) | 立即给"真厚尾+零复制+无接缝"参照线 |
| **E1** | 签名矩匹配主损失(held-out Sig-MMD) | Top-2 | 低(改 losses+train) | signature.py(已有) | SigP_新颖(v11 no-op→真信号) |
| **E2** | GARCH×扩散白化杂交 | Top-3 | 中(改 dataset+generate) | E0 的 GARCH 滤波器 | **C2ST_新颖 ~0.60 floor + 复制率(双攻)** |
| **E3** | B2 合成 surrogate 预训→真实微调 | (Top-1+3 合流) | 中(两阶段训练) | E0 的 DCC/GJR 模拟器 | 抬"可学结构"分母 |
| **E4** | B1 多资产同构对跨市场预训 | Top-4 | 中-高(联网取数) | FRED/Stooq 取数 | 抬真·独立窗(7→~140) |
| **E*** | 强正则+早停 hygiene 包裹 | Top-5 | 零(改 train 旋钮) | — | 每个实验默认外壳,巩固低复制+省半算力 |

> **E\*(强正则+早停)不是独立实验,而是 E1–E4 训练时的默认包裹**:开 dit1d attention/MLP dropout(当前=0.0)、给 Linear 套谱归一、按 ckpt 轨迹早停 ~1500-2000ep(质量 ep999 即饱和)、保持 WEIGHT_DECAY=1e-2/EMA。零成本、直击头号杀手记忆化。

---

## E0 · FHS / GARCH-t 零训练强基线(Top-1,先跑)

- **假设**:在 ~7 独立窗硬约束下,学 ~5-8 参低维递归(GARCH-t)+ 经验残差自助,可一次性拿下"真峰度~19 + 复制率≈0 + 无接缝长程",给出**比任何 58M DiT 配置更真实且更新颖**的零训练基线 —— 证伪/锚定"深网在 7 窗上的天花板"。
- **机制(新建 1 文件,不碰 DiT)**:
  - 新建 `eval/fhs_baseline.py`(~80 行纯 numpy/scipy):① 对 `train_sp500_us10y.csv` 两通道各拟合 GJR-GARCH-t(`scipy.optimize.minimize` MLE,~5-7 参/通道);② 收益除条件波动得标准化残差;③ 联合自助残差向量(保同期跨相关)+ GARCH 递归注回波动 → 批量生成 N×L=2048×2 路径;④ 导出 CSV 同 generate.py 格式。
  - **复用**:直接喂 `forensic_suite.py`;无需训练、无需 GPU、无需 arch 包。
- **最低成本验证(诚实闸门)**:`python forensic_suite.py fhs_baseline.csv` → 期望 **复制率≈0%**(新 innovation)、**峰度 17-19**(经验残差,直击 C1/line2 头号 FAIL)、**无 patch_spike**(递归连续,对照 A2 block-bootstrap 的 40x 接缝失败)。
- **数据/依赖前置**:无(numpy/scipy 已装)。可选 `pip install arch` 加速但非必需。
- **预期对 C2ST_新颖 floor 的突破**:**不破 floor 但移动基线**——FHS C2ST_新颖很可能仍被 long-range/run_len(GARCH 几何衰减 vs 真实幂律)微弱检出,**但这是"模型误设"而非"过平滑/记忆",且峰度大胜**。它定义了"零数据的强先验能到哪",是后续 E1-E4 的对照锚。
- **风险**:GARCH(1,1) 几何衰减波动 ACF 难捕长记忆 |r|-ACF 双曲慢衰减 → run_len/high_vol_frac 可能仍贴带或微塌。缓解:叠 component-GARCH(下一旋钮,见 candidates §1.7)。

## E1 · 签名矩匹配主损失(Top-2,与 E0 并行)

- **假设**:把仓库 signature.py 从 no-op 辅助损失**升为对固定 held-out 真实签名库的 Sig-MMD/Sig-W1 主损失**,可让 SigP_新颖 从 v11 的 no-op(0.003-0.007)回到有信号区,并补 DiT 最弱的 SP500↔DGS10 跨通道相关(签名 level-2 Lévy area 原生编码)。这是关键发现 4 开出、**v11 没做对**(in-batch + 权重 0.05 = no-op)的正确修复版。
- **机制(改 2 文件,复用 DiT)**:
  - `losses.py`:把现有可微 depth-2 `sig_mmd_loss`(167-176 行,当前对比**同 batch** x̂0 vs x0)改为对比**固定 held-out 真实签名库**(训练前从真实数据预算一批签名、`.detach()` 冻结);权重升为**主损失**量级;扩 depth 2→3;`recover_x0` 已在可直接用。
  - `train.py`:加载 held-out 签名库;**E\* 包裹**(dropout+谱归一+早停)。
  - 生成器**复用现有 DiT**;评估 `forensic_suite.py`(SigP 全量 n_perm≥300)。
  - 参考实现:PyTorch repo `luchungi/Generative-Model-Signature-MMD`(loss.py + generators.py + gaussianize.py)。
- **最低成本验证**:先做**廉价消融**——固定 batch 大小,只改"in-batch→held-out 库 + 主权重",看 `SigP_新颖` 是否从 ~0.003 升向标定 0.116/0.326;同时盯 `复制率`(矩匹配应不奖励逐点抄)。
- **数据/依赖前置**:无(signature.py 纯 numpy 已在,losses.py 可微版已在)。
- **预期对 C2ST_新颖 floor 的突破**:**主要动 SigP_新颖,不太可能破 C2ST_新颖 floor**——depth-3 签名对边缘峰度~19 相对盲视(须叠 clip20/EVT 边缘层),且 held-out 库有效自由度仍≈7。定位=**修签名律 + 跨通道相关**,与 E0/E2 的厚尾互补。
- **风险**:作 DiT 主损失时签名项与驱动记忆化的 eps-MSE 竞争,可能拉回过平滑;最干净版是小生成器(MA 噪声+浅网)纯 Sig 训练=换模型,工作量升中等。**先做 DiT 辅损改主损的廉价版,不行再考虑小生成器。**

## E2 · GARCH×扩散白化杂交(Top-3,E0 之后)

- **假设**:把厚尾/聚集/杠杆**外包给 E0 的数据高效 GARCH**,DiT 只在近 iid 的 GARCH 标准化残差上学(目标更平滑→记忆动机降),生成端乘回条件波动 —— **唯一跳出"模型侧三杠杆(clip/eta/block-bootstrap)已耗尽"面的结构侧正交杠杆**,最有希望**同时压 C2ST_新颖 floor 与复制率**。
- **机制(改 2 文件,复用 DiT+E0 模拟器)**:
  - `dataset.py`:加 GARCH 滤波层 —— 用 E0 拟合的 GJR-GARCH-t 把训练收益除条件波动 → DiT 训练目标变为标准化残差(近 iid、近平稳)。
  - `generate.py`:生成端把 DiT 输出的残差**乘回 GARCH 条件波动**(须采样/条件化一条一致的波动路径,可用 E0 的递归仿真 vol)。
  - **E\* 包裹** + `forensic_suite.py` 评估;**关键对照**:`--force_null` 类消融,确认增益非来自 context 坍缩 / 非 GARCH 单独贡献。
- **最低成本验证**:对比三条 CSV —— 纯 DiT(现状)、纯 E0-FHS、E2 杂交。看 E2 是否 **C2ST_新颖 < 0.62(破 floor 迹象)且 复制率仍≈0**;若 C2ST_新颖 仅等于 E0 则说明 DiT 残差项无增量(白化未起效)。
- **数据/依赖前置**:E0 的 GARCH 滤波器(numpy);无新包。
- **预期对 C2ST_新颖 floor 的突破**:**这是 5 个推荐里最有机会真正推动 floor 的模型侧实验**——把欠厚尾外包给经验波动后,DiT 在更简单残差上若能学到跨通道高阶结构,C2ST_新颖 有望从 0.62 下探。但诚实地说,残差仍是 7 窗的残差,floor 只会被推动不会消除。
- **风险**:① GARCH 滤波不完美则残差留结构→DiT 仍记忆;② 白化路须生成一致波动路径(vol 采样不当则失真);③ 增广变体可能把 GARCH 模型偏误(规整指数衰减/无长记忆)教给 DiT 而非真高阶结构(模型误设泄露)。缓解:force_null 对照 + 同时报 E0 基线隔离 GARCH 单独贡献。

## E3 · B2 合成 surrogate 预训→真实微调(Top-1+3 合流)

- **假设**:用 DCC/GJR-GARCH-t(+必要 FIGARCH 长记忆)生成**无限合成 2 通道**先训现有 DiT,把"可学结构"分母从 7 抬到无限,真实 14734 行**仅做轻量微调校准** → 模型先学"律"后贴"例",**防记忆最强**(真实 7 窗占训练极小份额)。DELPHYNE(2025)实证 GARCH-only 合成预训 NLL 0.0865 优于混合;Chronos-2 报"纯合成≈全量"。
- **机制(改 dataset + 两阶段训练,复用 DiT)**:
  - 复用 E0/E2 的 GARCH 模拟器,扩为 **DCC-GARCH**(+2 标量建跨通道时变相关)→ `dataset.py` 加合成预训练流(无限批);
  - 阶段1:DiT 在合成上预训(大批);阶段2:真实 14734 上 **LoRA/早停轻微调**(E\* 包裹,微调步数极少防过拟合)。
  - 评估 `forensic_suite.py`;**DELPHYNE 配方**(any-variate + Student-t 混合厚尾头 + GARCH+wavelet 合成)可注入。
- **最低成本验证**:先验证"纯合成预训(零真实微调)"的 forensic 分数——若已接近全量(印证 Chronos-2/DELPHYNE),则微调只做轻校准;盯 复制率(应远低于从零训 DiT)。
- **数据/依赖前置**:DCC/GJR-GARCH 模拟器(numpy,~150 行,E0 扩展);全离线无联网。
- **预期对 floor 的突破**:**抬分母但 floor 被移动非消除**——救不了 SP500↔DGS10 市场特异残差(那部分仍只有 7 窗)。预期复制率史低 + stylized 由合成保证。
- **风险**:易把 GARCH 伪影(规整指数衰减聚集、缺长记忆、香草无杠杆)当真相灌进权重,7 窗微调难纠。缓解:**必须用 GJR(杠杆)+FIGARCH(长程)+DCC(跨通道)精心构造合成器**,naive GARCH 在杠杆/长程必塌(恰是现存弱族)。

## E4 · B1 多资产同构对跨市场预训(Top-4,抬真·独立窗)

- **假设**:用 ~20 国同构对(股指 log 收益,10Y 国债日差)预训现有 DiT,把**真实独立窗 7→~140**,再微调 SP500↔DGS10 —— **唯一加真·独立信息**的方向,CLAUDE.md 钉死的治本方向,格式与现 DiT 100% 同构。
- **机制(取数管线 + 两阶段训练,复用 DiT)**:
  - 取数:复用已就位的 `eval/fetch_fred_treasury.py`(FRED 多国 10Y)+ 新增 Stooq/Yahoo 取股指(DAX/FTSE/Nikkei/CAC 等);per-market z-score 归一 + 交易日历/缺失对齐。
  - 阶段1:DiT 在 ~20 市场同构对上预训;阶段2:SP500↔DGS10 微调(E\* 包裹)。
  - 评估 `forensic_suite.py`(注意:跨市场后须用 SP500↔DGS10 自己的标定)。
- **最低成本验证**:先验证"取数+对齐"管线产出干净的 N 市场 2 通道 CSV;再看预训后微调的 **复制率**(应随真实独立窗增多而进一步降)与 **C2ST_新颖**(唯一可能真·下探的方向)。
- **数据/依赖前置**:**需联网**取多国股指(Stooq/Yahoo)+ FRED 多国 10Y;交易日历对齐(有界工作量)。
- **预期对 floor 的突破**:**最有原理依据真·破 floor 的方向**——但诚实:全球危机(2008/2020)同步命中所有市场→有效独立宏观 regime 全局共享,真实独立窗远不到 20×;且各国股债相关符号随 regime 翻转异质,跨市场平均可能**稀释** SP500↔DGS10 特异跨通道相关。
- **风险**:负迁移(异市场制度异质)、取数/对齐工程量、"7→140"被高估。缓解:优先同域(金融股债对,非通用 TSFM),per-market 归一,微调阶段保留足够 SP500↔DGS10 信号。

---

## 落地顺序与决策树

```
E0 (FHS 强基线, 1-2天, 必跑)  ──┐
E1 (签名主损失, 与E0并行)      ──┤── 两者廉价、零/低依赖、立即出锚
                                 │
        ┌────────────────────────┘
        ▼
E2 (GARCH×扩散白化杂交)  ← 用 E0 的 GARCH;最可能破 C2ST floor 的模型侧实验
        │
        ├─ C2ST_新颖 < 0.62 且 复制率≈0  →  成功,固化为 line2/v14 主配方
        │
        └─ C2ST_新颖 仍 ≈0.62           →  确证模型侧已尽,转 E3/E4 抬分母
                                              │
                          E3 (B2 合成预训, 离线)  ←先做,无需联网
                          E4 (B1 多资产, 需联网)  ←唯一真·加独立窗,终极治本

[E* 强正则+早停: 包裹 E1/E2/E3/E4 每次训练, 零成本默认开]
```

## 横贯诚实护栏(与 CLAUDE.md 铁律一致)

- **没有一个实验能"破"** C2ST_新颖 ≈0.60 信息论 floor —— 它是数据稀缺下界。E0-E3 在"模型侧/先验侧"只能**推动**它(尤其 E2 杂交、E3 合成);**唯有 E4(注入真新信息)**与 E0/E2 的"正确参数律"(若真过程近 GARCH-t)有可能真·下探。
- **每个实验报三栏**(真实度/新颖度/hw01 参照),互不替代;**外部 hw01 非地面真值**(罚真实厚尾、奖过平滑),绝不 Goodhart。
- **SigP 须全量 n_perm≥300**(`--quick` 毁 sig 标定 0.326→0.046);复制率/C2ST/峰度低成本可信。
- **峰度天花板**:加 epoch 推不动(全程死锁 ~8),靠放宽 clip(clip20→峰度20)或外包给 GARCH 经验残差(E0/E2),非靠训练。
