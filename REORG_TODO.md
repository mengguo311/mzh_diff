# REORG_TODO — 深度目录分包待办清单

> 第一波"安全归档"已完成(见 `STRUCTURE.md` + commit `6eb538f`):仅搬动无引用的旧 log/run/CSV。
> 本清单是**第二波**——需要改文件内 import/路径的"分类分包",**必须在 v11 评估流水线全部跑完、无任何运行进程时**执行。
> 已把每一处引用关系查清并写进步骤,照做即可。所有路径相对 `/home/u00134/src`。

---

## ☐ Phase 0 — 前置门禁(必做,缺一不可)

- [ ] 确认 v11 训练 + 步骤F评估已结束:`tail -n 5 logs/auto_eval_v11.log` 末尾出现 `DONE`。
- [ ] 确认无运行进程钉住文件:`pgrep -fa "train.py|generate.py|auto_eval_v11|score.py|c2st.py|signature.py"` 应为空。
- [ ] `git status` 干净(或先提交)。在 `deep_v10` 分支操作。
- [ ] 上保险:`git tag pre-reorg-v2`(出错可 `git reset --hard pre-reorg-v2`)。

> 若仍有进程在跑 → **停手**,只能做无引用文件的归档。

---

## ☐ Phase 1 — eval/ 子分类(高价值、中等改动;**推荐做**)

目标布局(核心取证脚本**留在 eval/ 顶层**,因其本就是该目录的主职能,可见性是好事):
```
eval/
├── score.py c2st.py signature.py diagnostics.py metrics.py verdict_v11.py   # 核心(不移)
├── auto_eval_v11.sh rescore_v11.sh run_sweep.sh                             # 流水线(不移)
├── score2/  docs/                                                           # 不移
├── tools/      # ← 新增:分析/绘图/转换工具
│   plot_compare.py plot_detectors.py aggregate.py select_best.py
│   show1.py eval_distribution.py score_mixed.py price_table_converter.py
└── results/    # ← 新增:所有评估结果 *.json
```

### ✅ 1A. 建 `eval/tools/` 并搬入 8 个工具 —— 已完成 (2026-06-20)
- [x] `git mv` 8 个工具到 `eval/tools/`(plot_compare/plot_detectors/aggregate/select_best/show1/eval_distribution/score_mixed/price_table_converter)
- [x] 修 3 个 `sys.path` 深度:`parent.parent` → `parents[2]`(select_best/eval_distribution/score_mixed)
- [x] 修 `score_mixed`:`from eval.eval_distribution` → `from eval.tools.eval_distribution`(score/score2 不变)
- [x] 修 `eval/run_sweep.sh:55` → `eval/tools/aggregate.py`;更新 `~/CLAUDE.md` 关键文件表
- [x] 冒烟全绿:8 工具语法 OK;6 个核心取证模块 import OK;移动后的工具(含跨模块 score_mixed)import OK;`verdict_v11.py` 端到端 OK

### ⏸ 1B. 建 `eval/results/` 并搬入所有 JSON —— **暂缓**(引用面大、风险>收益)
> 复查发现 `eval/*.json` 被多处硬编码/glob 引用:`tools/aggregate.py`(glob `eval/score_sw_*.json`/`eval/diag_sw_*.json`)、
> `tools/plot_compare.py`/`tools/plot_detectors.py`(多个 `eval/*.json`)、`verdict_v11.py`、`auto_eval_v11.sh`、`rescore_v11.sh`。
> 全改风险高、收益(eval/ 根去 json 杂)有限,**暂不做**;若要做,按下列改点一次性改全:
- [ ] `mkdir -p eval/results && git mv eval/*.json eval/results/`
- [ ] 改 `eval/verdict_v11.py`:`HERE` 拼接处加 `results/` 子目录
  (`os.path.join(HERE, p)` → `os.path.join(HERE, "results", p)`)。
- [ ] 改写后续会**重跑**的脚本里写/读 JSON 的路径(已跑过的历史 .sh 仅影响未来重跑):
  - `eval/auto_eval_v11.sh`:`eval/c2st_v11.json`、`eval/signature_v11.json`、`eval/diag_v11_val.json`、`eval/v11rw_v11_val.json` → `eval/results/...`
  - `eval/rescore_v11.sh`:`eval/v11rw_*.json` → `eval/results/...`
  - `eval/tools/plot_detectors.py`:`eval/c2st_results.json`、`eval/signature_results.json` → `eval/results/...`
- [ ] 冒烟:`python eval/verdict_v11.py` 仍能读到 `results/` 下的 v10/v11 JSON(打印判据,不报"缺失")。

---

## ☐ Phase 2 — 核心源码分包(可选,**高改动**:会改文档命令)

> 现状:9 个源码 `.py` 扁平在根目录,被 `import config` / `from dataset import ...` 直接引用,
> 且 `CLAUDE.md`/`v11_task_handoff.md` 所有命令写的是 `python train.py`、`python eval/score.py`。
> **入口 `train.py`/`generate.py` 建议留在根**(否则要改全部文档命令)。仅当确实想分包时再做。

可选目标(把"非入口库模块"收进 `core/`,入口留根):
```
~/src/
├── train.py generate.py            # 入口留根(文档命令不变)
└── core/  config.py dataset.py dit1d.py unet1d.py scheduler.py losses.py utils.py
```
若执行,需要:
- [ ] `mkdir -p core && git mv {config,dataset,dit1d,unet1d,scheduler,losses,utils}.py core/` + 加空 `core/__init__.py`。
- [ ] 全仓改 import:`import config` → `from core import config`;`from dataset import X` → `from core.dataset import X`;
      `dit1d/unet1d/scheduler/losses/utils` 同理。涉及:`train.py`、`generate.py`、`core/` 内部互相 import、
      `eval/score.py`(`import config` / `from dataset import` / `from unet1d import` / `from dit1d import` / `from scheduler import`)、
      其余 eval 脚本。建议 `grep -rn "^import config\|^from \(dataset\|dit1d\|unet1d\|scheduler\|losses\|utils\) import"`
      列全清单逐一改。
- [ ] `config.py` 里 `LOG_DIR/OUTPUT_DIR` 用 `os.path.expanduser("~/src/...")` 是绝对计算,**不受搬动影响**,无需改。
- [ ] 冒烟:`python train.py --help`、`python generate.py --help`、`python eval/score.py --help` 全部 import 通过;
      再跑一遍 `losses` 单元梯度自检(见下"通用冒烟")。
- [ ] 同步更新 `CLAUDE.md` 关键文件表的路径、`STRUCTURE.md`。

> 评估:根目录归档后已较清爽(9 源码 + 少量 md + 5 目录),Phase 2 收益有限而改动面大,**非必要可不做**。

---

## ☐ Phase 3 — 消除 output/ vs outputs/ 命名混淆(可选,小改动)

> `output/`=生成数据 CSV;`outputs/figures/`=绘图 PNG。两名相近易混。可把 `outputs/figures` 顶到 `figures/`。
- [ ] `git mv outputs/figures figures && rmdir outputs`(确认 `outputs/` 仅含 `figures/`)。
- [ ] 全仓替换硬编码 `outputs/figures` → `figures`(共 ~7 处):
      `eval/tools/plot_compare.py`(2)、`eval/tools/show1.py`(1)、`eval/tools/eval_distribution.py`(默认值)、
      `eval/tools/select_best.py`(默认值)、`eval/tools/plot_detectors.py`(2)、`eval/diagnostics.py`(仅 docstring 示例,可顺手改)。
      `grep -rn "outputs/figures" --include=*.py .` 验证替换干净(返回空)。
- [ ] 改 `STRUCTURE.md` 对应说明。

---

## ☐ 通用冒烟与收尾(每个 Phase 后)
- [ ] 语法:`python -c "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('**/*.py',recursive=True)];print('OK')"`
- [ ] 取证链 import:`python eval/score.py --help && python eval/c2st.py --help && python eval/signature.py --help`
- [ ] losses 可微自检(确保没被分包改坏):跑一遍 git 历史里 v11 提交用过的 `sig_mmd_loss` 梯度方向检查
      (固定子窗起点下 Adam 使 MMD² 单调下降)。
- [ ] 端到端抽检:`python eval/verdict_v11.py` 正常输出判据。
- [ ] 更新 `STRUCTURE.md`,提交:`git commit`(信息说明搬了什么、改了哪些 import),`git push origin deep_v10`。
- [ ] 全绿后删标签:`git tag -d pre-reorg-v2`。

---

## 一句话优先级
**Phase 1(eval/ 分 tools/+results/)收益最高、风险可控,优先做;Phase 2/3 视意愿,可不做。**
