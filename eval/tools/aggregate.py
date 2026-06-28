"""
eval/aggregate.py — 汇总采样扫描结果为一张对比表。

读取 eval/score_sw_*.json (score.py 输出) + eval/diag_sw_*.json (diagnostics.py 输出)，
打印统一表格: score.py 加权总分及关键分项 + 诊断族的 gap(fake−real)。
按总分降序。可选把 v9 基线 (eval/v9_20k_score.json) 一并列出。

用法:  python3 eval/aggregate.py
"""

import glob
import json
import os


def load_score(p):
    s = json.load(open(p))["scores"]
    return s["batch_total"], s["batch_components"]


def load_diag_gaps(name):
    p = f"eval/diag_sw_{name}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {
        "d2_gap": d["roughness_sp"]["d2_energy"]["gap_mean"],
        "racf1_gap": d["roughness_sp"]["ret_acf1"]["gap_mean"],
        "run_gap": d["regime"]["mean_run_len"]["gap_mean"],
        "vol_gap": d["regime"]["max_rolling_vol"]["gap_mean"],
    }


def main():
    rows = []

    # v9 基线 (若存在)
    if os.path.exists("eval/v9_20k_score.json"):
        tot, c = load_score("eval/v9_20k_score.json")
        rows.append(("v9_20k_baseline", tot, c, None))

    for sp in sorted(glob.glob("eval/score_sw_*.json")):
        name = os.path.basename(sp)[len("score_sw_"):-len(".json")]
        tot, c = load_score(sp)
        rows.append((name, tot, c, load_diag_gaps(name)))

    # 按总分降序 (基线保持参与排序)
    rows.sort(key=lambda r: -r[1])

    sc = ["ddpm_mse", "sp_acf", "dg_acf", "uncond_corr", "tail_corr", "wasserstein", "sp_kurt"]
    h = (f"{'config':18s} {'TOTAL':>6s} " + " ".join(f"{k[:6]:>6s}" for k in sc)
         + "  | " + f"{'d2_gap':>9s} {'run_gap':>8s} {'vol_gap':>9s}")
    print(h)
    print("-" * len(h))
    for name, tot, c, g in rows:
        line = f"{name:18s} {tot:6.2f} " + " ".join(f"{c[k]:6.2f}" for k in sc)
        if g:
            line += f"  | {g['d2_gap']:+9.2e} {g['run_gap']:+8.2f} {g['vol_gap']:+9.2e}"
        else:
            line += "  | " + " " * 28 + "(无诊断)"
        print(line)

    print("\n说明: gap = fake − real。d2_gap<0=偏平滑; run_gap>0=regime 太黏; vol_gap<0=波动爆发不足。")
    print("      真实自检总分 ≈ 52.47 为上限参照。")


if __name__ == "__main__":
    main()
