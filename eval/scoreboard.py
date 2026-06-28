#!/usr/bin/env python3
"""
eval/scoreboard.py — 双线统一对比【三栏】记分牌(dual-track 3-column scorecard)。

任一候选 CSV → 三栏并报, 三栏物理并排、任一栏不得 override 另两栏:
  栏1【真实度】realism_board 六族否决式 PASS/FAIL + 污染 gap  —— line1 的【唯一 go/no-go】
  栏2【新颖度】copy_rate / c2st_novel / sig_p_novel(forensic_suite)—— 仅监控(line2 北极星)
  栏3【外部 hw01 参照】fool_rate / mean_combined(队友 v5 鉴别器缓存)—— 仅交叉参照,【非优化目标】

护栏(路线图 §5.3): hw01 fool【绝不】作 PRIMARY_METRIC(下方硬断言); 栏2/3 异常只触发"解释"不触发"返工"。
三栏正交甚至冲突(真实度奖厚尾, hw01 罚厚尾): 出现"栏1好但栏3 fool 差"是正常, 不返工。

用法:
  conda run -n ts_diffusion python eval/scoreboard.py --candidate output/line1/xxx.csv --line line1
  column -s, -t < eval/scoreboard.csv
"""
import argparse
import os
import sys
import csv
import json
import time
import subprocess

import numpy as np
import pandas as pd

REPO = "/home/u00134/src"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
import forensic_suite as F  # noqa: E402
import config  # noqa: E402

# ── 护栏: hw01 fool 绝不可作 PRIMARY(路线图红线) ──
assert getattr(config, "PRIMARY_METRIC", "") != "external_fool", \
    "hw01 fool 不可作 PRIMARY_METRIC(hw01 非基准, 仅外部参照)。请改 configs/lineX.py。"

SCOREBOARD = os.path.join(REPO, "eval", "scoreboard.csv")
MGUO_LOG = "/home/u00134/data/generation/mguo/log"
TMP = "/home/u00134/.claude/jobs/22bfa227/tmp"
EXT_THRESHOLD = 0.5108
COLUMNS = ["timestamp", "line", "run", "L",
           # 栏1 真实度(唯一 go/no-go)
           "rl_realism", "rl_fail", "rl_resting",
           # 栏2 新颖度(仅监控)
           "in_copy_rate", "in_c2st_novel", "in_sig_p_novel", "in_kurt",
           # 栏3 外部 hw01 参照(非优化目标)
           "ext_fool_rate", "ext_mean_combined", "ext_source"]


def infer_line(path, explicit):
    if explicit and explicit != "auto":
        return explicit
    p = path.replace("\\", "/")
    if "/line1/" in p:
        return "line1"
    if "/line2/" in p:
        return "line2"
    return "?"


def model_key(path):
    b = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_top1000", "_ar2048", "_realctx2048", "_clamp", "_clean"):
        b = b.replace(suf, "")
    return b


# ── 栏1: 真实度(realism_board 子进程, 读 json) ──
def realism_scorecard(candidate, real):
    out = os.path.join(TMP, f"_sb_realism_{os.getpid()}.json")
    try:
        subprocess.run(["conda", "run", "--no-capture-output", "-n", "ts_diffusion", "python",
                        os.path.join(REPO, "eval", "realism_board.py"),
                        "--real", real, "--fakes", f"c={candidate}", "--json", out],
                       check=True, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        d = json.load(open(out)); det = d["detect"]["c"]
        fams = [k for k in d["bands"]]
        fail = [k for k in fams if not det["all"][k]["pass"]]
        return {"rl_realism": "PASS" if det["realism_pass"] else "FAIL",
                "rl_fail": "|".join(fail) if fail else "-",
                "rl_resting": "|".join(det.get("resting_on_copies") or []) or "-"}
    except Exception as e:
        return {"rl_realism": f"err:{str(e)[:20]}", "rl_fail": "-", "rl_resting": "-"}
    finally:
        if os.path.exists(out):
            os.remove(out)


# ── 栏2: 新颖度(forensic_suite 内部诚实闸门) ──
def novelty_scorecard(candidate, real):
    rep, _ = F.run_forensic(candidate, real, label=os.path.basename(candidate))
    return {"L": rep["L"],
            "in_copy_rate": round(rep["memorization"]["copy_rate_p95"], 4),
            "in_c2st_novel": round(rep["novelty_rerank"]["c2st_novel"], 4),
            "in_sig_p_novel": round(rep["novelty_rerank"]["sig_p_novel"], 4),
            "in_kurt": round(rep["stylized_facts"]["sp500"]["fake"]["kurt"], 3)}


# ── 栏3: 外部 hw01 参照(缓存) ──
def external_scorecard(candidate):
    key = model_key(candidate)
    summ = os.path.join(MGUO_LOG, "discrimination_summary.csv")
    if os.path.isfile(summ):
        df = pd.read_csv(summ); hit = df[df["model"] == key]
        if len(hit):
            r = hit.iloc[0]
            return {"ext_fool_rate": round(float(r["fool_rate"]), 4),
                    "ext_mean_combined": round(float(r["mean_combined"]), 4), "ext_source": "cache"}
    return {"ext_fool_rate": "pending", "ext_mean_combined": "pending",
            "ext_source": "未缓存(跑 mguo/discriminate_mguo.py)"}


def append_row(row):
    new = not os.path.isfile(SCOREBOARD)
    with open(SCOREBOARD, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def print_board():
    if not os.path.isfile(SCOREBOARD):
        return
    df = pd.read_csv(SCOREBOARD)
    print("\n===== eval/scoreboard.csv 三栏(栏1真实度=唯一go/no-go | 栏2新颖度=监控 | 栏3 hw01=仅参照) =====")
    show = ["line", "run", "L", "rl_realism", "rl_fail", "in_copy_rate", "in_c2st_novel",
            "in_kurt", "ext_fool_rate"]
    show = [c for c in show if c in df.columns]
    with pd.option_context("display.max_rows", None, "display.width", 220, "display.max_colwidth", 28):
        print(df[show].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="双线统一三栏对比记分牌")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--line", default="auto")
    ap.add_argument("--real", default=F.REAL_DEFAULT)
    ap.add_argument("--run", default=None)
    ap.add_argument("--no-realism", action="store_true", help="跳过栏1(realism_board, 较慢)")
    args = ap.parse_args()

    run = args.run or os.path.splitext(os.path.basename(args.candidate))[0]
    line = infer_line(args.candidate, args.line)
    print(f"[scoreboard] line={line} run={run} | PRIMARY={config.PRIMARY_METRIC}(hw01 fool 仅参照非目标)")

    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M"), "line": line, "run": run}
    t0 = time.time()
    row.update(novelty_scorecard(args.candidate, args.real))      # 栏2(顺带给 L)
    print(f"  栏2 新颖度({time.time()-t0:.0f}s): copy={row['in_copy_rate']} "
          f"C2ST_新颖={row['in_c2st_novel']} 峰度={row['in_kurt']}")
    if not args.no_realism:
        t1 = time.time(); row.update(realism_scorecard(args.candidate, args.real))
        print(f"  栏1 真实度({time.time()-t1:.0f}s): {row['rl_realism']}  FAIL族={row['rl_fail']}  "
              f"RESTING={row['rl_resting']}")
    row.update(external_scorecard(args.candidate))
    print(f"  栏3 hw01参照({row['ext_source']}): fool={row['ext_fool_rate']} combined={row['ext_mean_combined']}")

    append_row(row)
    print(f"  → 追加到 {SCOREBOARD}")
    print_board()


if __name__ == "__main__":
    main()
