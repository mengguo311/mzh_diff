#!/usr/bin/env python3
"""
eval/scoreboard.py — 双线统一对比记分牌 (dual scorecard)。

两条并行开发线(line1 真实度优先 / line2 内部诚实优先)用【同一把尺】横向对比:
任一候选 CSV → ① 内部诚实闸门(forensic_suite, 总是计算) + ② 外部 fool(队友 v5 鉴别器,
读 mguo 缓存日志; 无缓存则标 pending, 用 discriminate_mguo.py 现跑) → 追加一行到 eval/scoreboard.csv。

- 内部(快, ~3min/文件): copy_rate / c2st_novel / sig_p_novel / kurt / composite / n_div
- 外部(读缓存): fool_rate / mean_p_real / mean_combined  (越高越能骗过队友鉴别器)
两套指标【正交甚至冲突】(内部奖励新颖+不抄+厚尾; 外部奖励过平滑+贴近真实质心, 罚厚尾) —— 并报, 互不替代。

用法:
  conda run -n ts_diffusion python eval/scoreboard.py --candidate output/line1/xxx.csv --line line1
  conda run -n ts_diffusion python eval/scoreboard.py --candidate output/line2/yyy.csv   # 自动从路径推 line
  # 看汇总: column -s, -t < eval/scoreboard.csv   (或直接读 eval/scoreboard.csv)
"""
import argparse
import os
import sys
import csv
import time

import numpy as np
import pandas as pd

REPO = "/home/u00134/src"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
import forensic_suite as F  # noqa: E402

SCOREBOARD = os.path.join(REPO, "eval", "scoreboard.csv")
MGUO_LOG = "/home/u00134/data/generation/mguo/log"
EXT_THRESHOLD = 0.5108   # 队友鉴别器判真阈值(THIRDPARTY_CLAUDE.md)
COLUMNS = ["timestamp", "line", "run", "L", "n",
           "in_copy_rate", "in_c2st_novel", "in_sig_p_novel", "in_kurt", "in_composite", "in_n_div",
           "ext_fool_rate", "ext_mean_p_real", "ext_mean_combined", "ext_source"]


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
    """候选文件名 → 用于匹配 mguo 外部缓存的模型名。"""
    b = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_top1000", "_ar2048", "_realctx2048", "_clamp", "_clean"):
        b = b.replace(suf, "")
    return b


def internal_scorecard(candidate, real):
    rep, _ = F.run_forensic(candidate, real, label=os.path.basename(candidate))
    v = rep["verdict"]
    return {
        "L": rep["L"], "n": rep["n_candidate"],
        "in_copy_rate": round(rep["memorization"]["copy_rate_p95"], 4),
        "in_c2st_novel": round(rep["novelty_rerank"]["c2st_novel"], 4),
        "in_sig_p_novel": round(rep["novelty_rerank"]["sig_p_novel"], 4),
        "in_kurt": round(rep["stylized_facts"]["sp500"]["fake"]["kurt"], 3),
        "in_composite": (round(v["composite_score"], 1) if v["composite_score"] is not None else "NA"),
        "in_n_div": rep["divergence_guard"]["n_diverged_rows"],
    }


def external_scorecard(candidate):
    """读队友 v5 鉴别器缓存(mguo log): 优先 summary.csv, 退回逐样本 discriminate_<model>.csv。"""
    key = model_key(candidate)
    summ = os.path.join(MGUO_LOG, "discrimination_summary.csv")
    if os.path.isfile(summ):
        df = pd.read_csv(summ)
        hit = df[df["model"] == key]
        if len(hit):
            r = hit.iloc[0]
            return {"ext_fool_rate": round(float(r["fool_rate"]), 4),
                    "ext_mean_p_real": round(float(r["mean_p_real"]), 4),
                    "ext_mean_combined": round(float(r["mean_combined"]), 4),
                    "ext_source": "cache:summary"}
    per = os.path.join(MGUO_LOG, f"discriminate_{key}.csv")
    if os.path.isfile(per):
        df = pd.read_csv(per)
        fool = float((df["p_real_classifier"] >= EXT_THRESHOLD).mean())
        return {"ext_fool_rate": round(fool, 4),
                "ext_mean_p_real": round(float(df["p_real_classifier"].mean()), 4),
                "ext_mean_combined": round(float(df["combined_real_score"].mean()), 4),
                "ext_source": "cache:per-sample"}
    return {"ext_fool_rate": "pending", "ext_mean_p_real": "pending",
            "ext_mean_combined": "pending",
            "ext_source": "未缓存(跑 mguo/discriminate_mguo.py 后再 --refresh)"}


def append_row(row):
    new = not os.path.isfile(SCOREBOARD)
    with open(SCOREBOARD, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def print_board():
    if not os.path.isfile(SCOREBOARD):
        return
    df = pd.read_csv(SCOREBOARD)
    print("\n===== eval/scoreboard.csv (双线对比; 内部诚实↑越新颖/外部 fool↑越骗过) =====")
    show = ["line", "run", "L", "in_copy_rate", "in_c2st_novel", "in_kurt", "in_composite",
            "ext_fool_rate", "ext_mean_combined"]
    show = [c for c in show if c in df.columns]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df[show].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="双线统一对比记分牌")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--line", default="auto", help="line1/line2/auto(从路径推)")
    ap.add_argument("--real", default=F.REAL_DEFAULT)
    ap.add_argument("--run", default=None, help="run 标签(默认取文件名)")
    ap.add_argument("--no-internal", action="store_true", help="跳过内部闸门(只补外部缓存)")
    args = ap.parse_args()

    run = args.run or os.path.splitext(os.path.basename(args.candidate))[0]
    line = infer_line(args.candidate, args.line)
    print(f"[scoreboard] line={line} run={run} candidate={args.candidate}")

    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M"), "line": line, "run": run,
           "L": "", "n": "", "in_copy_rate": "", "in_c2st_novel": "", "in_sig_p_novel": "",
           "in_kurt": "", "in_composite": "", "in_n_div": ""}
    if not args.no_internal:
        t0 = time.time()
        row.update(internal_scorecard(args.candidate, args.real))
        print(f"  内部诚实闸门 ({time.time()-t0:.0f}s): copy={row['in_copy_rate']} "
              f"C2ST_新颖={row['in_c2st_novel']} SigP_新颖={row['in_sig_p_novel']} "
              f"峰度={row['in_kurt']} 综合={row['in_composite']} n_div={row['in_n_div']}")
    row.update(external_scorecard(args.candidate))
    print(f"  外部 fool({row['ext_source']}): fool_rate={row['ext_fool_rate']} "
          f"p_real={row['ext_mean_p_real']} combined={row['ext_mean_combined']}")

    append_row(row)
    print(f"  → 已追加到 {SCOREBOARD}")
    print_board()


if __name__ == "__main__":
    main()
