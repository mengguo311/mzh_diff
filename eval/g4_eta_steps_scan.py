#!/usr/bin/env python3
"""
eval/g4_eta_steps_scan.py — line1 路线图 G4: eta/steps 零训练扫描(治 regime 黏滞)。

用 v10 base(logs/deep_v10_dit_b, DiT-B, L=2048)重采样, 扫 eta×steps, 看纯采样能否把
regime 指标(mean_run_len → 真实~33 / high_vol_frac → 0.5 / switch_rate)拉向真实。
若能 → 长程辅助损失(G6)免做、省一次重训; 若不能 → regime 病须训练侧解决。
约束: 选点不破坏峰度(eta 增纹理但勿打崩尾部)。

用法: CAMPAIGN_GPU=0 conda run -n ts_diffusion python eval/g4_eta_steps_scan.py
"""
import os
import sys
import json
import time
import subprocess

import numpy as np

REPO = "/home/u00134/src"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
from diagnostics import load_changes, regime_block, VOL_WINDOW  # noqa: E402
from scipy.stats import kurtosis as _kurt  # noqa: E402

CKPT = os.path.join(REPO, "logs", "deep_v10_dit_b", "checkpoint_final.pt")
SCAL = os.path.join(REPO, "logs", "deep_v10_dit_b", "scaler.pt")
REAL = "/home/u00134/data/train_sp500_us10y.csv"
TMP = "/home/u00134/.claude/jobs/22bfa227/tmp"
GPU = os.environ.get("CAMPAIGN_GPU", "0")
N = 512
COMBOS = [(1.0, 200), (1.0, 500), (0.5, 200), (0.5, 500), (0.8, 300)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def generate(eta, steps, out):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = GPU
    env["CONFIG_PROFILE"] = "line1"   # v9/v10 recipe: COND_DIM=2 匹配 v10 checkpoint(无 context)
    cmd = ["conda", "run", "--no-capture-output", "-n", "ts_diffusion", "python", "-u",
           os.path.join(REPO, "generate.py"), "--model", "dit-b",
           "--checkpoint", CKPT, "--scaler", SCAL, "--num_samples", str(N),
           "--num_inference_steps", str(steps), "--eta", str(eta), "--output", out]
    subprocess.run(cmd, env=env, check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    real = load_changes(REAL, target_seq_len=2048)
    rr = real[:, 0, :]
    reg_real, _ = regime_block(rr, rr, VOL_WINDOW)
    real_runlen = reg_real["mean_run_len"]["real_mean"]
    real_hv = reg_real["high_vol_frac"]["real_mean"]
    log(f"G4 扫描开始 (v10 base, GPU={GPU}); 真实: mean_run_len={real_runlen:.1f} high_vol={real_hv:.3f}")
    log(f"基准对照: v10_retrained.csv(eta1/steps200)在 realism_board 上 run_len=42.6 / high_vol=0.46 / 峰度7.68")

    rows = []
    for eta, steps in COMBOS:
        out = os.path.join(TMP, f"g4_e{eta}_s{steps}.csv")
        try:
            t0 = time.time()
            generate(eta, steps, out)
            fake = load_changes(out); rf = fake[:, 0, :]
            reg, _ = regime_block(rf, rr, VOL_WINDOW)
            runlen = reg["mean_run_len"]["fake_mean"]
            hv = reg["high_vol_frac"]["fake_mean"]
            sw = reg["switch_rate"]["fake_mean"]
            kurt = float(_kurt(rf.ravel()))
            rows.append({"eta": eta, "steps": steps, "mean_run_len": runlen,
                         "high_vol_frac": hv, "switch_rate": sw, "kurt": kurt})
            log(f"  eta={eta} steps={steps} ({time.time()-t0:.0f}s): "
                f"run_len={runlen:.1f}(真实{real_runlen:.1f}) high_vol={hv:.3f}(0.5) "
                f"switch={sw:.4f} 峰度={kurt:.2f}")
        except Exception as e:
            log(f"  [ERR] eta={eta} steps={steps}: {e}")
        finally:
            if os.path.exists(out):
                os.remove(out)
        json.dump({"real": {"mean_run_len": real_runlen, "high_vol_frac": real_hv}, "scan": rows},
                  open(os.path.join(REPO, "eval", "g4_scan.json"), "w"), indent=2, default=float)

    # 判读
    if rows:
        best = min(rows, key=lambda r: abs(r["mean_run_len"] - real_runlen))
        log(f"\nG4 结论: 最接近真实 run_len({real_runlen:.1f}) 的工作点 = eta{best['eta']}/steps{best['steps']} "
            f"→ run_len={best['mean_run_len']:.1f} (峰度{best['kurt']:.2f})")
        if abs(best["mean_run_len"] - real_runlen) < 5 and best["kurt"] > 6:
            log("  → 纯采样可把 regime 拉近真实且不破坏峰度 → G6 长程辅助损失可免, regime 用此工作点")
        else:
            log("  → 纯采样未能把 regime 拉到真实(或破坏峰度) → regime 病须训练侧(AUX/v-pred)解决")
    log("G4 完成。结果 eval/g4_scan.json")


if __name__ == "__main__":
    main()
