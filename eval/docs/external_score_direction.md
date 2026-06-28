# line1 训练方向 —— 最大化第三方交叉评分（external fool/combined）

> ⛔ **已取代(2026-06-24)**:用户澄清 **hw01_discriminator 不是基准/优化目标**,**绝不 Goodhart 去 game 它的 fool 率**。
> 本文(最大化 fool = 过平滑迎合 hw01)框架**作废**,仅留作"hw01 奖励什么"的归因参考。
> **line1 正确方向 = 逼近真·真实市场(诚实诊断+分布距离)+ 同时提升我们自己的鉴别器**,见 **`line1_realism_direction.md`**。

> (历史)旧用户目标:可适当容许复制率,换取大幅提高真实性 = 最大化队友 v5 鉴别器的 fool/combined。
> 依据:`thirdparty/THIRDPARTY_CLAUDE.md` + 11-agent 归因(test_AUC 0.9999,与上报 Spearman 0.809)。
> 配方落地 = `configs/line1.py`;对比 = `eval/scoreboard.py`;结构 = `dual_track_structure.md`。

## 0. 一句话方向
**过平滑贴近真实窗 TCN 质心(nn_k5 小)+ L=2048 + DGS10 量化指纹 + 反厚尾;容许复制率。** 承 v9/v10。

## 1. 该评分奖励什么(量化归因,combined=0.75·p_real+0.25·sf_prob)
| 杠杆 | 与 fool 相关 | 方向 |
|---|---|---|
| TCN-embedding 贴近真实窗近邻 `nn_k5↓` | **−0.839(头号)** | 过平滑/像"平均脸"→ 落进真实簇判真 |
| 峰度(厚尾) | 厚尾**有害**(负类含 GARCH-t/SABR 厚尾合成) | **低峰度~2 fool 最高**;真实厚尾被推向合成侧 |
| 窗长 | L=512 崩到 4.3%,concat→2048 回 34% | **L≥1260(用 2048)** |
| DGS10 量化到 0.01 | 真实 rounding=0.0814,全 12 模型=0 | **匹配它(最大未开发杠杆)** |
| 逐点复制率 | **−0.15(解耦)** | 容许复制不直接帮 fool;真杠杆是过平滑贴质心 |

实证:v9_20k fool **57%**(≈真实窗自身 p_real 0.643)居首,v13_a1_L512 垫底 4.3%。**项目既往(抗记忆+攻厚尾+缩窗)恰是该评分最讨厌的 → line1 方向需翻转。**

## 2. line1 方向全表(按 优先级×成本)
| # | 方向 | 杠杆 | 成本 | 最低成本验证 | 泄露风险 | 预期 fool 增益 |
|---|---|---|---|---|---|---|
| **P0-1** | **DGS10 量化到 0.01**(后处理→训练) | DGS 指纹 | **后处理零训练** | 对 v9_20k CSV 的 DGS10 列 `np.round(x/0.01)*0.01` + ~8% 零差分 → 重跑 `discriminate_mguo.py` | 低(匹配真实属性,非编码泄露) | 全模型普涨(0.25 权重 sf + 分类器特征) |
| **P0-2** | **换基座 v9_20k/v10 + 确认 L=2048** | nn_k5↓+窗长 | 零训练(用现有 CSV) | 直接重跑 scoreboard/discriminate | — | 立得 fool 上限~57% |
| **P1** | **专用配方 v14-extfool 重训** | 过平滑+低峰度 | full(按 M1 缩 epoch) | 见下 | **中**:不对其 TCN 编码器直接训(单裁判泄露) | 边际 |
| P2 | 关 stylized 厚尾辅助(USE_AUX_LOSS=False) | 不向合成簇漂 | 已在 line1 档案 | resume 对照 | — | v10_retrained 正是被该损失出卖 |

## 3. 廉价先行序列(零训练优先)
1. **DGS10 量化后处理 + 重打分**(P0-1):对 v9_20k 现有 CSV 改 DGS10 列 → `discriminate_mguo.py` → 看 fool 是否跳升。**头号快赢,零训练。**
2. **确认基座**:v9_20k / v10(已 fool ~57%)为 line1 起点;`eval/scoreboard.py` 落表。
3. 若零训练触顶仍想更高 → **v14-extfool 重训**(见 §4)。

## 4. v14-extfool 配方(`configs/line1.py` 已固化)
- SEQ_LEN=2048, CLIP_RANGE=8(过平滑), **USE_AUX_LOSS=False**(关厚尾辅助), USE_CONTEXT_COND=False, **DGS10_QUANTIZE=0.01**(生成端接线), NUM_EPOCHS≈v9_20k 量级(可按 M1 式轨迹定 epoch*)。
- go/no-go:外部 fool 较 v9_20k 提升 **AND** 用 held-out 真实窗/多裁判验证(**不对队友 TCN 编码器直接优化**,防过拟合单裁判)。

## 5. 护栏(必须并存)
- **该评分非地面真值**:它惩罚真实厚尾、奖励过平滑 → 优化它=让模型"更像它的经典-合成训练分布",非更接近真实市场。用户已接受(作业评分标准)。
- **防单裁判泄露**:只优化底层真实属性(低阶矩贴质心/DGS 量化/长窗),**绝不对其 TCN 编码器训练**(不可推广;= GAN 式同流形自欺,项目红线)。
- **双记分牌**:`eval/scoreboard.py` 同时报内部诚实闸门(C2ST_新颖/复制率)与外部 fool,line1 优化外部但内部并报作参考,**两线互不摧毁**。
