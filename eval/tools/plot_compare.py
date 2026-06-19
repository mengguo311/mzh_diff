"""
eval/plot_compare.py — cross-version comparison (v9 baseline / v10 sampling / v10 retrained)

Reads existing score JSONs + diagnostics JSONs, draws a 3-panel figure:
  A) score.py component bars (higher = better)
  B) score.py total (all scored by the SAME v9 model) + real self-score reference
  C) diagnostic |gap| normalized to v9 (lower = better): roughness / regime / burst

Labels in English (matplotlib default font lacks CJK glyphs).
Usage:  python eval/plot_compare.py   ->  outputs/figures/v10_compare.png
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (name, score_json, diag_json) -- scores all by the same v9 model for comparability
VERSIONS = [
    ("v9 baseline", "eval/v9_20k_score.json", "eval/diag_v9_20k.json"),
    ("v10 sampling", "eval/v10_score.json", "eval/diag_v10.json"),
    ("v10 retrained", "eval/v10_retrained_byV9_score.json", "eval/diag_v10_retrained.json"),
]
COLORS = ["#888888", "#33aa99", "#ee7755"]
REAL_SELF = 52.47


def load_scores(p):
    s = json.load(open(p))["scores"]
    return s["batch_total"], s["batch_components"]


def load_diag(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    totals, comps, diags = [], [], []
    for _, sp, dp in VERSIONS:
        t, c = load_scores(sp)
        totals.append(t); comps.append(c); diags.append(load_diag(dp))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    w = 0.26

    # Panel A: score.py components
    metrics = ["ddpm_mse", "sp_acf", "dg_acf", "uncond_corr", "tail_corr", "wasserstein", "sp_kurt"]
    x = np.arange(len(metrics))
    for i, (name, _, _) in enumerate(VERSIONS):
        axes[0].bar(x + (i - 1) * w, [comps[i][m] for m in metrics], w, label=name, color=COLORS[i])
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics, rotation=40, ha="right")
    axes[0].set_title("A. score.py components (higher=better, scored by v9)")
    axes[0].set_ylabel("score (0-100)"); axes[0].legend(fontsize=9)
    axes[0].annotate("ddpm_mse: rewards proximity\nto over-smooth model manifold",
                     xy=(0, comps[2]["ddpm_mse"]), xytext=(0.6, 12),
                     fontsize=8, color="#b00000",
                     arrowprops=dict(arrowstyle="->", color="#b00000", lw=0.8))

    # Panel B: totals
    names = [v[0] for v in VERSIONS]
    axes[1].bar(names, totals, color=COLORS)
    for i, t in enumerate(totals):
        axes[1].text(i, t + 0.15, f"{t:.2f}", ha="center", fontweight="bold")
    axes[1].axhline(REAL_SELF, color="red", ls="--", lw=1.5, label=f"real self-score {REAL_SELF}")
    axes[1].set_ylim(44, 54); axes[1].set_ylabel("total score")
    axes[1].set_title("B. score.py total (incl. ddpm_mse, w=0.20)"); axes[1].legend(fontsize=9)

    # Panel C: diagnostic |gap| normalized to v9 (lower=better)
    diag_keys = [("roughness_sp", "d2_energy"), ("roughness_sp", "ret_acf1"),
                 ("regime", "max_rolling_vol"), ("regime", "mean_run_len"), ("regime", "vol_of_vol")]
    labels = ["d2_energy\n(rough)", "ret_acf1\n(persist)", "max_roll_vol\n(burst)",
              "mean_run_len\n(regime)", "vol_of_vol"]
    x = np.arange(len(diag_keys))
    base = [abs(diags[0][b][k]["gap_mean"]) for b, k in diag_keys]
    for i, (name, _, _) in enumerate(VERSIONS):
        if diags[i] is None:
            continue
        vals = [abs(diags[i][b][k]["gap_mean"]) / (base[j] + 1e-12) for j, (b, k) in enumerate(diag_keys)]
        axes[2].bar(x + (i - 1) * w, vals, w, label=name, color=COLORS[i])
    axes[2].axhline(1.0, color="#888", ls=":", lw=1)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("|gap| relative to v9 (lower=better)")
    axes[2].set_title("C. diagnostic |fake-real| gap (normalized to v9)")
    axes[2].legend(fontsize=9)

    fig.suptitle("v9 -> v10 sampling -> v10 retrained   (real ceiling = 52.47)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("outputs/figures", exist_ok=True)
    out = "outputs/figures/v10_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[plot_compare] saved {out}")


if __name__ == "__main__":
    main()
