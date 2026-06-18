# 仓库目录结构 (STRUCTURE.md)

> 金融时间序列扩散模型项目。本文件说明各目录职责与"活跃/归档"分区。
> **深度分包待办**(eval 分 tools/results、源码分包、output/outputs 改名)见 [`REORG_TODO.md`](REORG_TODO.md),
> 需在 v11 评估流水线跑完、无运行进程时执行(因需改文件内 import/路径)。
> 真实训练数据在 `/home/u00134/data/train_sp500_us10y.csv`(**不在**本仓库)。`*.pt`/`*.csv` 已 gitignore。

## 顶层
| 路径 | 职责 |
|---|---|
| `train.py` | 训练入口(`python train.py ...`) |
| `generate.py` | 采样生成入口 |
| `config.py` | 全局超参 + Phase-2/v11 损失开关 |
| `dit1d.py` / `unet1d.py` | DiT-B 主干 / 旧 U-Net |
| `scheduler.py` | DDPM + DDIM(含 eta)调度器 |
| `losses.py` | min-SNR 加权 + stylized-fact 辅助损失 + 可微 Sig-MMD |
| `dataset.py` / `utils.py` | 数据管道(z-score+clip+滑窗) / EMA·checkpoint·种子 |
| `eval/` `logs/` `output/` `outputs/` `data/` | 见下 |

> 注:核心源码为扁平布局,被 `eval/*.py` 与 `train/generate.py` 以 `import config`、`from dataset import ...`
> 直接引用;**入口与模块路径被 CLAUDE.md/handoff 文档及 eval 流水线固定,勿随意搬动**(否则需同步改全部 import 与文档命令)。

## logs/ —— 训练运行产物
- **活跃/在用**:
  - `deep_v11_dit_b_val/` — **v11 验证训练(运行中)**
  - `deep_v9_dit_b_20k/` — v9 基线,**取证打分的固定参考模型**
  - `deep_v10_dit_b/` — v10 重训
  - `deep_training_v8_b32/` — legacy(被 `eval/score_mixed.py` 引用,故保留)
  - `run_dit_v11_val.log`(训练日志)、`auto_eval_v11.log`(步骤F自动评估)、`*.pid`
- `_archive/` — 历史 run 目录(deep_v7 / deep_v9_dit_b)+ 早期 .log(v6/v7/v8/v9 及根目录散落日志)

## output/ —— 生成样本 CSV
- **在用**:`deep_v10_retrained.csv`(v10 重训,取证基线)、`deep_v9_20k.csv`/`deep_v10.csv`(脚本/文档引用),即将产出 `deep_v11_val.csv`
- `_archive/` — 早期版本(deep_v7/v8*)+ 采样扫描临时(`sw_*`)+ 冰烟(`smoke_eta05`)

## outputs/figures/ —— 绘图工具输出 PNG
注意与 `output/` 区分:`output/`=数据 CSV,`outputs/figures/`=图片(被 `plot_*`/`show1`/`diagnostics --fig` 等硬编码引用)。

## eval/ —— 评估 / 取证
- **核心(被 v11 自动评估流水线依赖,勿移动)**:
  `score.py`(10项 fidelity;`ddpm_mse` 已降权 0.05 并标注自指)、`c2st.py`/`signature.py`(非自指鉴别器)、
  `diagnostics.py`(roughness/regime/PSD)、`metrics.py`(Wasserstein)、`verdict_v11.py`(步骤F判据)
- **v11 流水线脚本**:`auto_eval_v11.sh`(训练后自动评估,**运行中轮询**)、`rescore_v11.sh`
- **分析/绘图工具**:`plot_compare.py` `plot_detectors.py` `aggregate.py` `select_best.py` `show1.py`
  `eval_distribution.py` `score_mixed.py` `price_table_converter.py` `run_sweep.sh`
  (彼此有 `eval.score`/`eval.eval_distribution`/`eval.score2` 交叉依赖,若要归类到子目录需一并改 import)
- `score2/` — 高级打分子包(`advanced_scorer.py`,被 `score_mixed.py` 引用)
- `docs/` — 改进方案与路线图(`v11_task_handoff.md` 等)
- `*.json` — 评估结果(`v11rw_*` / `diag_*` / `c2st_*` / `signature_*`)

## data/ —— 辅助/中间数据
`mixed_data_1st/` `processed/`(混合资产实验等;真实训练数据见顶部说明,不在此)。
