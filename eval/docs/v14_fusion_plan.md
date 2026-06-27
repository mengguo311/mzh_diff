# v14-fusion — 多通道(收益率曲线)融合模型 · 正式训练前的优化与验证

> 新线(**非 line1/line2**),git tag **`v14-fusion`**,配置档案 **`configs/fusion.py`**(`CONFIG_PROFILE=fusion`)。
> 目标:用 FRED 美债数据扩通道,**同时**攻真实度+复制率,并完善配套鉴别器。
> 本文件记录:正式训练前已完成的【廉价证伪 + 代码 wiring + 冰烟验证 + 鉴别器升级】,以及训练/判读的可执行步骤。

## 0. 诚实论题(不夸大、反 Goodhart)

记忆化↔过平滑 Pareto 被**数据稀缺**(~6-7 个独立宏观段)锁死。本模型**不声称解决数据稀缺**,
而是**在不同的轴上各放一根经实证的独立杠杆**,把锁死 Pareto 的**两端同时推到更好工作点**:

| 轴 | 杠杆 | 来源 | 是"真增益"还是"只挪Pareto" |
|---|---|---|---|
| 真实度(跨资产) | +DGS2_diff 通道 = 2s10s 收益率曲线斜率(regime/衰退信号) | 新 | **真增益**(单通道物理上不可能有的维度) |
| 真实度(厚尾) | CLIP_RANGE=20(解除峰度天花板) | line1 | **真增益**(sp500 数据侧达原生 kurt 18.8) |
| 新颖/低复制 | 联合 block-bootstrap(造新宏观次序) | v13-A2 | **真增益**(copy≈0 同时保结构) |
| 新颖/低复制 | context-cond 富条件 + 自回归 512→2048 | v13-C1 | **真增益**(修签名/长程) |
| (病根) | 多通道**本身** | — | **只挪Pareto**:同一日历**不增独立窗**(3ch indep512 反 28.8→23.8) |

**关键诚实标注**:复制率的改善**全部**来自 joint-boot + context-cond + L=512 缩窗这三根独立杠杆,
**不能**归功于"加通道"。多通道只加真实度/取证维度。

## 1. 五路并行预检(零训练廉价证伪,均跑通可复现)

脚本在 `~/.claude/jobs/22bfa227/tmp/preflight_a{1..4}_*.py`。

| 杠杆 | 决定 | 依据(真实数字) |
|---|---|---|
| **通道集 3ch** `[sp500, DGS10, DGS2_diff]` | **GO** | DGS2-DGS10 corr **0.81**(强联动但 ~34%独立方差=2s10s斜率;非>0.9近重复);仅 -18% 窗(7111→5844),起始 1976-06,indep512 28.8→23.8 |
| T10YIE 入主集 | **NO-GO** | 2003 硬边界砍 62% 行 → indep512 28.8→**11.0**,把缩窗独立段一次性吐回。仅留 2003+ 消融臂 |
| DGS30 入主集 | **NO-GO** | DGS2-DGS30 corr 0.725 冗余;可选 4ch 全曲线消融 |
| **联合 block-bootstrap** | **GO** | copy_rate **0.00%** + 跨通道相关 0.799→**0.810**(保持)+ 块内 stylized 退化≤**6.9%**;**独立**重采样对照崩到 0.001 → 实证**必须联合** |
| **clip=20 统一** | **GO** | clip20 > 全通道 max\|z\|(≤19.43)→ 每通道命中原生峰度(sp500 18.8 / DGS10 8.4 / DGS2 19.2),无削尾无过冲。per-channel clip 零收益(反削 DGS2 厚尾)→ NO-GO |
| **context-cond** | **GO** | C1 实证:复制率 2.2%、SigP 0.003→0.106、long-range 修(high_vol 0.27→0.49);消融门过 |
| **鉴别器升级** | **GO**(进 forensic/我方,不进诚实闸门) | 受控演示:毁跨通道结构保边际的 fake,marginal AUC **0.515→加cross 1.000** |

> ⚠️ **更正**(对抗核验发现):预检 A1 报 DGS2-DGS10 corr=0.536 是把【已差分的 DGS10 列又 diff 一次】(二阶差分)的产物;
> 主CSV 的 `DGS10` 列**本就是日差分**(mean≈0, std 0.067, kurt 8.4)。真实 corr≈**0.81**(A2/实测一致)。结论不变(仍 GO),措辞更正为"强联动+独立斜率"。

## 2. 最终配方(`configs/fusion.py`)

```
CHANNELS=3  CHANNEL_COLS=[sp500, DGS10, DGS2_diff]  (DGS2_diff 来自 FRED, 1976-06+)
QUANTIZE_GRID={DGS10:0.01, DGS2_diff:0.01}  OUTPUT_PREFIX={...DGS2_diff:dgs2}  (sp500 永不量化)
SEQ_LEN=512  STRIDE=2  CLIP_RANGE=20.0  USE_AUX_LOSS=True
USE_BLOCK_BOOTSTRAP=True  BLOCK_LEN=192  BOOT_FRAC=0.3
USE_CONTEXT_COND=True  N_CTX_FEAT=8  → COND_DIM=24
USE_SIG_MMD=False  WEIGHT_DECAY=1e-2  NUM_EPOCHS=2000  CHECKPOINT_EVERY=500  模型=DiT-S(32.66M, in_channels=3)
```

## 3. 代码 wiring(全程保 line1/line2 的 2ch 复现 —— base 不动,fusion 档案注入)

- `config.py`:新增 `FRED_PATH/CHANNEL_COLS/CHANNEL_SOURCES/QUANTIZE_GRID/OUTPUT_PREFIX`(base 默认=2ch);
  `COND_DIM` 两处 `2*N_CTX_FEAT`→`CHANNELS*N_CTX_FEAT`。
- `dataset.py`:多通道加载(主CSV + 按需 FRED join,按 CHANNEL_COLS 选列,截到全通道非NaN起始,起点后 ffill);
  scaler 打印名/断言泛化;`_cond_from` 的 null 维 + cat 改 `range(CHANNELS)`;block-bootstrap 已天然联合(整块全列搬运)。
- `train.py`:model_builders **补 `in_channels=config.CHANNELS`**(原漏传 → 多通道必修)。
- `generate.py` / `generate_autoregressive.py`:输出泛化 N 通道循环 + per-channel 量化(float64 保网格)+ **兼容列名**(ch0→`sp500_`、ch1→`dgs10_`、新通道真名小写)以不破坏下游 eval 解析。
- `utils.py`:config.json 快照增 `channel_cols/quantize_grid/output_prefix`。

## 4. 鉴别器升级(双重目标的鉴别器侧,已验证)

- `eval/c2st.py`:`featurize(..., cross_channel=False)` 新开关 + `_cross_row`(10 维跨通道/曲线取证特征:
  股债/利率联动相关、危机期尾部协动、2s10s 斜率变化 std/kurt、同号率、lead-lag)。**默认关,不污染诚实闸门**。
- `eval/forensic_cross.py`(新,forensic_auc 的 3ch 版):real(主CSV+FRED join)vs 候选(3ch宽表)的
  marginal vs marginal+cross AUC + real-vs-real 标定自检。
- **验证**:A4 式受控 fake(毁跨通道结构保边际)→ marginal AUC **0.500 → +cross 1.000**(跨通道增益 +0.500);
  真实自身当候选 → 全 0.500(无误报)。证多通道给鉴别器**边际盲区外**的取证力。

## 5. 冰烟验证(全部通过)

- 4 档案(base/line1/line2/fusion)dataset+model forward/backward:fusion=3ch/COND_DIM 24/12198行/5844窗,
  joint-boot 窗 (512,3),DiT-S(4,3,512),stylized 3ch;**2ch 三档案数值不变 → 向后兼容确认**。
- 完整 `train.py` 2-epoch:config 快照 channels=3、in_channels=3、EMA/checkpoint OK,loss 0.48→0.18,aux 激活。
- `generate_autoregressive` 3ch:输出 sp500_/dgs10_/dgs2_ 各 1024 列、两利率通道量化触发。

## 6. 正式训练(Step D)

```bash
cd ~/src && CONFIG_PROFILE=fusion nohup conda run --no-capture-output -n ts_diffusion \
  python -u train.py --model dit-s --run_name fusion/v14_3ch_clip20_boot_ctx \
  > logs/run_fusion_v14.log 2>&1 &
# DiT-S 2000ep, CHECKPOINT_EVERY=500。clip20 解锁厚尾后须复核 M1 式中间 ckpt 轨迹定 epoch*。
```
训练后(自回归 + 终裁):
```bash
CONFIG_PROFILE=fusion conda run --no-capture-output -n ts_diffusion python -u generate_autoregressive.py \
  --model dit-s --checkpoint logs/fusion/v14_3ch_clip20_boot_ctx/checkpoint_final.pt \
  --scaler logs/fusion/v14_3ch_clip20_boot_ctx/scaler.pt --num_samples 5120 --k 4 --seed_ctx real \
  --output output/fusion_v14_ar2048.csv
# 鉴别器升级交叉验证:
CONFIG_PROFILE=fusion conda run --no-capture-output -n ts_diffusion python -u eval/forensic_cross.py \
  --fakes fusion=output/fusion_v14_ar2048.csv --json eval/forensic_cross_v14.json
```

## 7. ⚠️ Step C — 诚实闸门必须对 3ch+L512 重标定(绝不套旧标尺)

- 复制率底噪:3ch 非重叠 real-vs-real 已实证 **0.00%**(A2)。
- C2ST_cal / SigP_cal:**须在 3ch+L512 真实数据上重算**,绝不套旧 0.0%/0.496/0.116(2ch 口径)。
  SigP 须全量 `n_perm≥300`(`--quick` 毁 sig 标定)。
- 当前 `memorization.py`/`forensic_suite.py`/`novelty_rerank.py` 经 `load_changes` 读 **2ch 子空间**(sp500+DGS10),
  对 3ch 候选会**忽略 dgs2_ 通道**(不崩,但只判 2ch)。Step C 须把诚实闸门接到 3ch(或显式声明判 2ch 子空间口径)后再重标定。
- 跨通道特征**只进 forensic_cross / 可选报告,绝不进 go/no-go 诚实闸门**(防移动球门 + 防 Goodhart);hw01 三栏只作外部参照。

## 8. go/no-go(终裁,全量 N=5120,盲测 holdout 特征)

峰度≥12(sp500) **AND** C2ST_新颖≤地板上沿 **AND** 复制率<3ch地板上沿 **AND** 长程不退 **AND** frac_div<1%
**AND** forensic_cross 的 marginal+cross AUC 不显著高于纯 stylized(即跨通道结构也逼真)。

## 9. 未决风险

- 多通道**不增独立窗**(indep512 28.8→23.8),数据稀缺铁律不被打穿;复制率全靠三杠杆,诚实判读。
- 2ch 旧 ckpt **不可复用**(patch_embed/unpatchify 维度随 C 变)→ 全新训练。scaler/标定全须重算。
- clip20 是厚尾**必要非充分**条件:sp500 生成侧峰度仍可能因 AUX/损失不达真实 18.8,须看生成侧诊断。
- joint-boot(30% null-ctx)与 context-cond(70% real-ctx)混训配比未实证最优;BOOT_FRAC/BLOCK_LEN 训后看诊断微调。
- DGS2_diff 真实 kurt 19.2(利率非薄尾),3ch 新通道厚尾对 DiT-S 容量是新负担;欠厚尾须单独诊断。
