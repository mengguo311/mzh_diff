#!/usr/bin/env python3
"""eval/clip20_regime_scan.py — clip20 模型 eta/steps 采样扫描, 攻 regime(零训练)。
clip20 已学到厚尾(峰度20.15)但 regime 黏滞(mean_run_len 63.6 vs 真实33)。看纯采样能否把
run_len 拉回 ~33 + high_vol 拉到 0.5, 且【不破坏峰度】(须仍 ≥17.5 进带)。若不能→regime须训练侧。"""
import os, sys, json, time, subprocess
import numpy as np
from scipy.stats import kurtosis as _kurt
sys.path.insert(0, "/home/u00134/src/eval")
from diagnostics import load_changes, regime_block, VOL_WINDOW  # noqa: E402

REPO = "/home/u00134/src"; RUN = f"{REPO}/logs/line1/l1_clip20"
CKPT = f"{RUN}/checkpoint_final.pt"; SCAL = f"{RUN}/scaler.pt"
REAL = "/home/u00134/data/train_sp500_us10y.csv"
TMP = "/home/u00134/.claude/jobs/22bfa227/tmp"
GPU = os.environ.get("CAMPAIGN_GPU", "1")
COMBOS = [(1.0, 200), (1.0, 500), (0.5, 500), (1.0, 1000)]   # eta, steps


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gen(eta, steps, out, n=512):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = GPU; env["CONFIG_PROFILE"] = "line1_clip20"
    subprocess.run(["conda", "run", "--no-capture-output", "-n", "ts_diffusion", "python", "-u",
                    f"{REPO}/generate.py", "--model", "dit-b", "--checkpoint", CKPT, "--scaler", SCAL,
                    "--num_samples", str(n), "--num_inference_steps", str(steps), "--eta", str(eta),
                    "--output", out], env=env, cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    real = load_changes(REAL, target_seq_len=2048); rr = real[:, 0, :]
    reg_r, _ = regime_block(rr, rr, VOL_WINDOW)
    real_rl = reg_r["mean_run_len"]["real_mean"]; real_hv = reg_r["high_vol_frac"]["real_mean"]
    log(f"clip20 regime 采样扫描; 真实 run_len={real_rl:.1f} high_vol={real_hv:.3f}; "
        f"clip20默认(eta1/steps500) run_len=63.6 峰度20.15. 目标: run_len→33 且 峰度≥17.5")
    rows = []
    for eta, steps in COMBOS:
        out = f"{TMP}/clip20_e{eta}_s{steps}.csv"
        try:
            t0 = time.time(); gen(eta, steps, out)
            f = load_changes(out); rf = f[:, 0, :]
            reg, _ = regime_block(rf, rr, VOL_WINDOW)
            rl = reg["mean_run_len"]["fake_mean"]; hv = reg["high_vol_frac"]["fake_mean"]
            k = float(_kurt(rf.ravel()))
            rows.append({"eta": eta, "steps": steps, "run_len": rl, "high_vol": hv, "kurt": k})
            log(f"  eta{eta}/steps{steps} ({time.time()-t0:.0f}s): run_len={rl:.1f}(真{real_rl:.0f}) "
                f"high_vol={hv:.3f}(0.5) 峰度={k:.2f}(须≥17.5)")
        except Exception as e:
            log(f"  [ERR] eta{eta}/steps{steps}: {e}")
        finally:
            if os.path.exists(out): os.remove(out)
        json.dump({"real_run_len": real_rl, "scan": rows},
                  open(f"{REPO}/eval/clip20_regime_scan.json", "w"), indent=2, default=float)
    # 判读: 找峰度仍进带(≥17.5)且 run_len 最接近真实的点
    ok = [r for r in rows if r["kurt"] >= 17.5]
    if ok:
        best = min(ok, key=lambda r: abs(r["run_len"] - real_rl))
        log(f"\n判读: 厚尾保住(峰度≥17.5)中 run_len 最近真实 = eta{best['eta']}/steps{best['steps']} "
            f"→ run_len={best['run_len']:.1f}(真{real_rl:.0f})")
        if abs(best["run_len"] - real_rl) < 6:
            log("  ✅ 纯采样可把 regime 拉近真实且不丢厚尾 → 用此工作点, regime 免训练侧")
        else:
            log("  ❌ 纯采样不足以修 regime(run_len 仍远) → regime 须训练侧(提 AUX 权重/regime 损失)")
    else:
        log("\n判读: 所有采样点都破坏了厚尾(峰度<17.5) → 采样换不来 regime, 须训练侧")
    log("完成。eval/clip20_regime_scan.json")


if __name__ == "__main__":
    main()
