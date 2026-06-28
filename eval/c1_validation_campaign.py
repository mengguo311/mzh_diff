#!/usr/bin/env python3
"""
eval/c1_validation_campaign.py — C1 低成本验证战役驱动 (零训练, Tier 0/1)

按 eval/docs/C1_improvement_analysis.md 推荐执行序列:
  步骤1 M5 稳定性标定护栏: 在 C1 已生成的 realctx2048.csv 上, 跑 forensic 的
        (max_samples × quick/full × seed) 网格, 量出各北极星指标的【复现 mean±std】,
        定出"信号必须 > 此 std 才可信"的下限 (尤其暴露 quick/低 N 对 SigP 的破坏)。
  步骤2 M1 中间 checkpoint 指标轨迹: 对 ep{0999,1999,2999,3999,4999} 各自回归生成
        N=1024 × 3 seed, forensic 打分, 画 copy_rate/C2ST_新颖/峰度/high_vol_frac vs epoch,
        回答"最少需多少训练 / 是否仍在升 / 是否过拟合记忆化"。

全程零训练 (复用 checkpoint + 现有 CSV)。增量落盘, 异常隔离。
输出: eval/forensic_out/campaign/{m5_calibration.json, m1_trajectory.json, m1_trajectory.png, summary.txt}
"""
import os
import sys
import json
import time
import subprocess

import numpy as np

REPO = "/home/u00134/src"
sys.path.insert(0, REPO)
import forensic_suite as F  # noqa: E402

OUT = os.path.join(REPO, "eval", "forensic_out", "campaign")
os.makedirs(OUT, exist_ok=True)
TMP = "/home/u00134/.claude/jobs/22bfa227/tmp"
os.makedirs(TMP, exist_ok=True)
C1_CSV = os.environ.get("C1_CSV_OVERRIDE",
                        os.path.join(REPO, "output", "deep_v13_c1_ctx_realctx2048.csv"))
REAL = F.REAL_DEFAULT
RUNDIR = os.path.join(REPO, "logs", "deep_v13_c1_ctx")
SCALER = os.path.join(RUNDIR, "scaler.pt")
GPU = os.environ.get("CAMPAIGN_GPU", "0")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(rep):
    return {
        "copy_rate": rep["memorization"]["copy_rate_p95"],
        "c2st_novel": rep["novelty_rerank"]["c2st_novel"],
        "c2st_all": rep["novelty_rerank"]["c2st_all"],
        "sig_p_novel": rep["novelty_rerank"]["sig_p_novel"],
        "kurt": rep["stylized_facts"]["sp500"]["fake"]["kurt"],
        "high_vol_frac": rep["diagnostics"]["regime"]["high_vol_frac"]["fake_mean"],
        "d2_ratio": rep["divergence_guard"]["d2_energy_ratio"],
        "c2_cal": rep["calibration_L_specific"]["c2st_real_vs_real"],
        "sig_cal": rep["calibration_L_specific"]["sig_p_real_vs_real"],
        "n_div": rep["divergence_guard"]["n_diverged_rows"],
        "n_kept": rep["divergence_guard"]["n_kept"],
    }


def agg(runs, keys):
    """runs: list[dict] → {key: (mean, std)}。"""
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}


# ════════════════════════ 步骤 1: M5 稳定性标定护栏 ════════════════════════
def run_m5():
    log("===== 步骤1 M5 稳定性标定护栏 开始 =====")
    grid = []
    configs = []
    for ms in [512, 1024, 2048, None]:
        for quick in [False, True]:
            configs.append((ms, quick))
    for ms, quick in configs:
        nsig, nperm = (150, 150) if quick else (300, 300)
        seeds = [1, 2, 3] if ms is not None else [1]   # 全量无子采样 → 单 seed
        runs = []
        for sd in seeds:
            try:
                rep, _ = F.run_forensic(C1_CSV, REAL, label="m5", n_sig=nsig, n_perm=nperm,
                                        max_samples=ms, seed=sd)
                m = metrics(rep)
                runs.append(m)
                grid.append({"max_samples": ms or "full", "quick": quick, "seed": sd, **m})
                log(f"[M5] ms={ms or 'full'} quick={quick} seed={sd} | "
                    f"copy={m['copy_rate']:.3f} C2ST_nov={m['c2st_novel']:.3f} "
                    f"SigP_nov={m['sig_p_novel']:.3f} sig_cal={m['sig_cal']:.3f} kurt={m['kurt']:.2f}")
            except Exception as e:
                log(f"[M5][ERR] ms={ms} quick={quick} seed={sd}: {e}")
        # 聚合该 config 跨 seed 的 std
        if runs:
            a = agg(runs, ["copy_rate", "c2st_novel", "sig_p_novel", "kurt", "high_vol_frac", "sig_cal"])
            log(f"[M5][AGG] ms={ms or 'full'} quick={quick} (n={len(runs)}): "
                f"C2ST_nov {a['c2st_novel'][0]:.3f}±{a['c2st_novel'][1]:.3f} | "
                f"SigP_nov {a['sig_p_novel'][0]:.3f}±{a['sig_p_novel'][1]:.3f} | "
                f"sig_cal {a['sig_cal'][0]:.3f} | copy {a['copy_rate'][0]:.3f}±{a['copy_rate'][1]:.3f}")
        # 增量落盘
        json.dump(grid, open(os.path.join(OUT, "m5_calibration.json"), "w"), indent=2, ensure_ascii=False)
    log("===== 步骤1 M5 完成 =====")
    return grid


# ════════════════════════ 步骤 2: M1 中间 checkpoint 轨迹 ════════════════════════
def generate(ckpt, out_csv, n=1024):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = GPU
    cmd = ["conda", "run", "--no-capture-output", "-n", "ts_diffusion", "python", "-u",
           os.path.join(REPO, "generate_autoregressive.py"), "--model", "dit-s",
           "--checkpoint", ckpt, "--scaler", SCALER,
           "--num_samples", str(n), "--k", "4", "--num_inference_steps", "200",
           "--seed_ctx", "real", "--output", out_csv]
    subprocess.run(cmd, env=env, check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_m1():
    log("===== 步骤2 M1 中间 checkpoint 轨迹 开始 =====")
    epochs = ["0999", "1999", "2999", "3999", "4999"]
    keys = ["copy_rate", "c2st_novel", "c2st_all", "sig_p_novel", "kurt", "high_vol_frac", "d2_ratio", "n_div"]
    traj = {}
    for ep in epochs:
        ckpt = os.path.join(RUNDIR, f"checkpoint_epoch_{ep}.pt")
        if not os.path.exists(ckpt):
            log(f"[M1][SKIP] {ckpt} 不存在"); continue
        runs = []
        for sd in [1, 2, 3]:
            tmp = os.path.join(TMP, f"c1_traj_{ep}_{sd}.csv")
            try:
                t0 = time.time()
                generate(ckpt, tmp)
                rep, _ = F.run_forensic(tmp, REAL, label=f"traj_{ep}_{sd}",
                                        n_sig=300, n_perm=300, max_samples=None)
                m = metrics(rep); runs.append(m)
                log(f"[M1] ep{ep} seed{sd} ({time.time()-t0:.0f}s) | "
                    f"copy={m['copy_rate']:.3f} C2ST_nov={m['c2st_novel']:.3f} "
                    f"SigP_nov={m['sig_p_novel']:.3f} kurt={m['kurt']:.2f} "
                    f"high_vol={m['high_vol_frac']:.3f} n_div={m['n_div']}")
            except Exception as e:
                log(f"[M1][ERR] ep{ep} seed{sd}: {e}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        if runs:
            a = agg(runs, keys)
            traj[ep] = {"n": len(runs), "agg": a, "runs": runs}
            log(f"[M1][AGG] ep{ep} (n={len(runs)}): "
                f"copy {a['copy_rate'][0]:.3f}±{a['copy_rate'][1]:.3f} | "
                f"C2ST_nov {a['c2st_novel'][0]:.3f}±{a['c2st_novel'][1]:.3f} | "
                f"SigP_nov {a['sig_p_novel'][0]:.3f}±{a['sig_p_novel'][1]:.3f} | "
                f"kurt {a['kurt'][0]:.2f}±{a['kurt'][1]:.2f} | "
                f"high_vol {a['high_vol_frac'][0]:.3f} | n_div {a['n_div'][0]:.1f}")
        json.dump(traj, open(os.path.join(OUT, "m1_trajectory.json"), "w"), indent=2, ensure_ascii=False)
    plot_m1(traj)
    log("===== 步骤2 M1 完成 =====")
    return traj


def plot_m1(traj):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[M1][plot] 跳过: {e}"); return
    eps = sorted(traj.keys())
    if not eps:
        return
    x = [int(e) + 1 for e in eps]
    panels = [("copy_rate", "copy rate (memorization, ↓)", None),
              ("c2st_novel", "C2ST_novel (↓=real, cal~0.495)", 0.495),
              ("sig_p_novel", "SigP_novel (↑=real, cal~0.326)", 0.326),
              ("kurt", "kurtosis (real~18.8)", 18.8),
              ("high_vol_frac", "high_vol_frac (real~0.50)", 0.50),
              ("d2_ratio", "d2_energy fake/real (real=1.0)", 1.0)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (k, title, ref) in zip(axes.ravel(), panels):
        mean = [traj[e]["agg"][k][0] for e in eps]
        std = [traj[e]["agg"][k][1] for e in eps]
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=4, color="#1f77b4")
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", alpha=0.6, label="real/calib")
            ax.legend(fontsize=8)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    fig.suptitle("C1 metric trajectory vs training epoch (N=1024, 3 seeds, mean±std)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "m1_trajectory.png"); fig.savefig(p, dpi=130); plt.close(fig)
    log(f"[M1][plot] {p}")


def main():
    t0 = time.time()
    log(f"C1 验证战役开始 (GPU={GPU}); 输出 {OUT}")
    m5 = run_m5()
    traj = run_m1()
    # 汇总
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("C1 低成本验证战役 汇总\n" + "=" * 60 + "\n\n")
        f.write("【M5 稳定性标定】关键: full vs quick 对 SigP 标定的破坏 + 低 N 噪声\n")
        for row in m5:
            f.write(f"  ms={row['max_samples']} quick={row['quick']} seed={row['seed']}: "
                    f"C2ST_nov={row['c2st_novel']:.3f} SigP_nov={row['sig_p_novel']:.3f} "
                    f"sig_cal={row['sig_cal']:.3f} copy={row['copy_rate']:.3f} kurt={row['kurt']:.2f}\n")
        f.write("\n【M1 轨迹】(mean±std over 3 seeds)\n")
        for ep in sorted(traj.keys()):
            a = traj[ep]["agg"]
            f.write(f"  ep{ep}: copy={a['copy_rate'][0]:.3f}±{a['copy_rate'][1]:.3f} "
                    f"C2ST_nov={a['c2st_novel'][0]:.3f}±{a['c2st_novel'][1]:.3f} "
                    f"SigP_nov={a['sig_p_novel'][0]:.3f}±{a['sig_p_novel'][1]:.3f} "
                    f"kurt={a['kurt'][0]:.2f} high_vol={a['high_vol_frac'][0]:.3f} "
                    f"n_div={a['n_div'][0]:.1f}\n")
    log(f"C1 验证战役完成, 总耗时 {(time.time()-t0)/60:.1f} min; 汇总见 {OUT}/summary.txt")


if __name__ == "__main__":
    main()
