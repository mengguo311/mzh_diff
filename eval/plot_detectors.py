"""eval/plot_detectors.py — 鉴别器结果可视化 (C2ST + Sig-MMD)。
读取 eval/c2st_results.json 与 eval/signature_results.json, 出 outputs/figures/detector_compare.png。
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c2 = json.load(open("eval/c2st_results.json"))
sg = json.load(open("eval/signature_results.json"))

order = ["v9", "v10_sampling", "v10_retrained"]
colors = ["#888888", "#33aa99", "#ee7755"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: C2ST accuracy (lower = more realistic)
accs = [c2["detect"][k]["test_acc"] for k in order]
cal_acc = c2["calibration"]["real_vs_real"]["test_acc"]
axes[0].bar(order, accs, color=colors)
for i, a in enumerate(accs):
    axes[0].text(i, a + 0.01, f"{a:.3f}", ha="center", fontweight="bold")
axes[0].axhline(0.5, color="red", ls="--", lw=1.5, label=f"calibration floor (real-vs-real={cal_acc:.3f})")
axes[0].set_ylim(0.45, 1.0); axes[0].set_ylabel("C2ST test accuracy")
axes[0].set_title("A. C2ST detectability (LOWER = more realistic)")
axes[0].legend(fontsize=9)

# Panel B: Sig-MMD p-value (higher = more realistic / passes)
ps = [sg["detect"][k]["p_value"] for k in order]
cal_p = sg["calibration"]["real_vs_real"]["p_value"]
axes[1].bar(order, ps, color=colors)
for i, p in enumerate(ps):
    axes[1].text(i, p + 0.01, f"{p:.3f}", ha="center", fontweight="bold")
axes[1].axhline(0.05, color="red", ls="--", lw=1.5, label="detection threshold p=0.05")
axes[1].axhline(cal_p, color="gray", ls=":", lw=1.2, label=f"real-vs-real p={cal_p:.3f}")
axes[1].set_ylim(0, max(0.5, max(ps) * 1.2)); axes[1].set_ylabel("Sig-MMD permutation p-value")
axes[1].set_title("B. Signature-MMD test (p>0.05 = indistinguishable from real)")
axes[1].legend(fontsize=9)

fig.suptitle("Non-self-referential fake detectors agree: v10 retrained is the most realistic",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs("outputs/figures", exist_ok=True)
fig.savefig("outputs/figures/detector_compare.png", dpi=130, bbox_inches="tight")
print("[plot_detectors] saved outputs/figures/detector_compare.png")
