# 实验序 E0→E4 总结论(数据稀缺金融双序列生成)

> 承接 `research/`(选型调研)的可执行实验序。逐实验细节见 `E0_results.md` / `E4_results.md`。
> 评分一律用 `forensic_suite.py` 诚实闸门(复制率/C2ST_新颖/SigP_新颖, 按 L 重标定)。

## 执行总览

| 实验 | 内容 | 结论 | 状态 |
|---|---|---|---|
| **E0** | FHS / GJR-GARCH(1,1)-t 零训练强基线 | 强计量先验**被反驳**(C2ST 0.946 vs DiT 0.601) | ✅ |
| **E0b** | component-GARCH 富化(长记忆波动) | C2ST 0.944 纹丝不动, **家族反驳坐实** | ✅ |
| **E4** | 多资产(US/JP/UK/EU)跨市场预训 + US 微调 | **SigP 0.36→0.95 决定性修复; native C2ST 推到史上最佳 0.548 PASS; L2048 floor 守** | ✅ |
| E1 | 签名矩匹配主损失 | **被 E4 subsume**(E4 已把 SigP 修到 0.95) | ⊘ 降级 |
| E2 | GARCH×扩散白化杂交 | **被 E0 证据降级**(GARCH 波动劣于 DiT, 外包不智) | ⊘ 降级 |
| E3 | GARCH 合成 surrogate 预训 | **被 E0 证据降级**(GARCH teacher 误设); 真数据(E4)优于合成 | ⊘ 降级 |

## 三个核心科学发现

### 1. 强计量先验在主真实度闸门上输给训练 DiT(E0/E0b)
研究调研的 Top-1 假设"数据稀缺下强先验小模型 > 深网"**被实证反驳**。6 参 GJR-GARCH(1,1)(及 component-GARCH 富化)零训练即得零复制+真厚尾+签名 PASS,但 **C2ST_新颖 0.94 远输训练 DiT 0.60** —— GARCH 几何衰减的波动律抓不住真实的长记忆 + 窗间过同质,24 维联合判别器轻松分离。**深网在波动动力学上反而赢**;这把后续 E2/E3(GARCH 基)证据性降级,指向唯一真杠杆 = 加真数据。

### 2. 真实度 gap 可分解为"普适律"+"市场特异例"两半(E4,最大价值)
注入 2.7× 真·独立窗(跨市场)后:
- **普适签名律 = 跨市场可迁移 → 已解**:SigP_新颖 0.362→**0.947**(几乎与真实不可分),v11/A1 长期签名塌陷的真正修复。
- **US 市场特异 C2ST 残差 = 不可迁移 → floor 守**:L=2048 C2ST 0.629(vs clip11 0.601,持平);外国市场各自 stylized 细节不同,补不了 SP500↔DGS10 的边缘/联合残差。
- **C2ST floor 锁的不是"窗数"而是"市场特异结构"** —— 即便 2.7× 真独立窗也破不了 L=2048 floor。这比"E4 破/不破 floor"的二元判断信息量大得多。

### 3. 跨市场预训产出项目首个【全 PASS】模型(E4 native)
密集早微调 **ep99 = 项目首个三诚实闸门同时 PASS**(`logs/e4/finetune_dense/checkpoint_epoch_0099.pt`):
copy **0.018 PASS** + **C2ST_新颖 0.539(史上最低/最真)PASS** + SigP **0.591 PASS** + 综合 **92.9(史上最高)**。
多资产预训【确实帮到 C2ST】(史上首次 C2ST_新颖 PASS, 非靠复制——在剔复制后的新颖子集上算),
只是幅度有限且仅在 native L=512(L=2048 floor 0.629 仍守)。**比 E0 朴素 prior、line2 已耗尽的 clip 杠杆都更进一步。**

## 落到模型/工程的产物

- **代码**:`eval/fhs_baseline.py`(E0 引擎, 复用为 E2/E3 备件)、`eval/fetch_multiasset.py`(4 市场取数)、`dataset.MultiAssetDataset`(per-market z-score 池化)、`train.py --init_from`(权重级微调)、`configs/e4_{pretrain,finetune,finetune_dense}.py`、`run_e4_chain.sh`(自动链)。
- **★ 最先进模型 = E4 native L=512 finetune_dense ep99**(`logs/e4/finetune_dense/checkpoint_epoch_0099.pt`)= **项目首个三诚实闸门全 PASS**:copy 0.018 / C2ST_新颖 0.539(史上最低)/ SigP 0.591 / 综合 **92.9**,真厚尾 + 长程/无接缝干净。
- **诚实定位**:无任何方法"凭空破" C2ST 数据稀缺 floor;能动它的只有真·独立信息(E4 已证对"律"有效, 对"市场特异例"有限)。下一步真治本 = 更多/更独立市场(降 UK-EU 0.88 冗余, 加亚太/新兴市场)或更长 US 历史。

## 一句话

> 在 ~7 独立窗硬墙前:**强计量先验(E0)被反驳——深网波动更强;真·跨市场数据(E4)决定性修好普适签名律(SigP 0.36→0.95)并产出项目首个三诚实闸门全 PASS 模型(native C2ST_新颖 0.539 史上最佳 / 综合 92.9),但 US 市场特异的 L=2048 C2ST floor 仍是数据稀缺下界。真实度 gap = 可迁移的"律" + 不可迁移的"市场特异例"。**
