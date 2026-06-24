# 双线并行开发结构（dual-track）

> 两条目标【正交甚至冲突】的开发线在【单分支】上并行,用【配置档案】隔离配方、【统一记分牌】对比。
> 建立 2026-06-24。依据 `THIRDPARTY_CLAUDE.md`(外部评分归因)+ `C1_improvement_analysis.md`(内部方向)。

## 两条线

| | **line1 — 真实度优先** | **line2 — 内部诚实/新颖优先** |
|---|---|---|
| 目标(主记分牌) | 最大化队友 v5 鉴别器 **fool/combined** | 内部诚实闸门 **C2ST_新颖/复制率/SigP_新颖** |
| 基座/方向 | 承 **v9/v10**:过平滑贴近真实窗 TCN 质心 + L=2048 + DGS10 量化指纹;**容许复制率** | 续 **v13-c1**:富条件+自回归 + 攻厚尾(放宽 clip) |
| 配方档案 | `configs/line1.py` | `configs/line2.py` |
| 关键配置 | SEQ_LEN=2048, CLIP=8, AUX=False, CTX=False, DGS10_QUANTIZE=0.01 | SEQ_LEN=512, CLIP=15, AUX=True, CTX=True, NUM_EPOCHS=2000 |

**为何对立**:外部鉴别器奖励过平滑/贴近真实质心、**惩罚真实厚尾**(负类含厚尾合成);内部闸门奖励新颖/不抄/厚尾。**neither 是地面真值** → 两线并行、双记分牌并报,互不替代。

## 结构(单分支 + 配置档案 + 命名空间 + 统一记分牌)

```
configs/{line1,line2}.py      # 两线配方(OVERRIDES dict), 同时存在于版本控制
config.py                     # 共享默认 + 档案加载器: CONFIG_PROFILE=lineX → 应用 configs/lineX.OVERRIDES
                              #   (重算 COND_DIM 等派生; 不设档案=base 默认)
output/{line1,line2}/         logs/{line1,line2}/   # 产物命名空间(csv/pt 仍 gitignore)
eval/scoreboard.py + scoreboard.csv   # ★统一双记分牌: 任一 CSV → 内部闸门 + 外部fool → 一张对比表
eval/forensic_suite.py …      # 整套 eval 量具【两线共享】(同一把尺才可比)
```

## 怎么用

**训练某条线**(档案经 `CONFIG_PROFILE` 环境变量加载,零侵入现有 `import config`):
```bash
# line1 (真实度优先)
CONFIG_PROFILE=line1 nohup conda run --no-capture-output -n ts_diffusion python -u train.py \
  --model dit-s --run_name line1/l1_<name> > logs/line1/l1_<name>.log 2>&1 &
# line2 (续 v13)
CONFIG_PROFILE=line2 nohup conda run --no-capture-output -n ts_diffusion python -u train.py \
  --model dit-s --run_name line2/l2_<name> > logs/line2/l2_<name>.log 2>&1 &
```
两线可在**同一 checkout 上同时双卡训练**(各自 CONFIG_PROFILE + run_name + CUDA_VISIBLE_DEVICES),配置读入内存互不污染。

**生成**:`CONFIG_PROFILE=lineX ... generate.py --output output/lineX/<name>.csv`(line1 自回归用 generate_autoregressive.py;DGS10_QUANTIZE 需在生成端接线,见下)。

**对比两条线**(随时):
```bash
conda run -n ts_diffusion python eval/scoreboard.py --candidate output/line1/xxx.csv   # 自动推 line
conda run -n ts_diffusion python eval/scoreboard.py --candidate output/line2/yyy.csv
column -s, -t < eval/scoreboard.csv      # 看横向对比表
```
每个 run 都被**两套标准都打一遍**:内部(copy_rate/c2st_novel/sig_p_novel/kurt/composite,forensic_suite 现算)+ 外部(fool_rate/p_real/combined,读 `mguo/log/` 缓存;无缓存先跑 `discriminate_mguo.py`)。

## 待接线(line1 首个开发任务)
- `DGS10_QUANTIZE`(config 已声明,默认 None):line1 头号杠杆。需在 `generate.py`/`generate_autoregressive.py` 生成后对 DGS10 通道 `np.round(x/q)*q` 吸附到网格 + 制造 ~8% 零差分,匹配真实量化指纹(全模型当前 rounding≈0,真实=0.0814)。**亦可零训练后处理现有 v9_20k CSV 先验证**(改 DGS10 列后重跑 discriminate_mguo)。

## git
单分支;里程碑按线打 tag(`l1-v1` / `l2-v14a` …)。CSV/checkpoint 由 `.gitignore` 忽略,对比靠 `eval/scoreboard.csv`(进库)。
