#!/usr/bin/env python3
"""line1 clip15 重训 go/no-go: 峰度轨迹(ep499/999/1499/final)+ 存 final 大批量供三栏验收。
对照: 旧模型(clip8) ~8 封顶 | clip15 数据天花板 12.9 | 真实 ~19。峰度明显超 8 → clip15 起效。"""
import os, sys, subprocess, json, time
import numpy as np
from scipy.stats import kurtosis, skew
sys.path.insert(0, "/home/u00134/src/eval")
from diagnostics import load_changes, regime_block, VOL_WINDOW  # noqa: E402

REPO = "/home/u00134/src"; RUN = f"{REPO}/logs/line1/l1_clip15"
REAL = "/home/u00134/data/train_sp500_us10y.csv"
TMP = "/home/u00134/.claude/jobs/22bfa227/tmp"
GPU = os.environ.get("CAMPAIGN_GPU", "1")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gen(ckpt, out, n, steps):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = GPU; env["CONFIG_PROFILE"] = "line1"
    subprocess.run(["conda", "run", "--no-capture-output", "-n", "ts_diffusion", "python", "-u",
                    f"{REPO}/generate.py", "--model", "dit-b", "--checkpoint", ckpt,
                    "--scaler", f"{RUN}/scaler.pt", "--num_samples", str(n),
                    "--num_inference_steps", str(steps), "--eta", "1.0", "--output", out],
                   env=env, cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    real = load_changes(REAL, target_seq_len=2048); rr = real[:, 0, :]
    real_k = float(kurtosis(rr.ravel()))
    log(f"line1 clip15 go/no-go; 真实峰度={real_k:.1f}; 旧clip8封顶~8; clip15数据天花板12.9")
    plan = [("0499", 512, 200, None), ("0999", 512, 200, None), ("1499", 512, 200, None),
            ("final", 1024, 500, f"{REPO}/output/line1/l1_clip15.csv")]
    traj = []
    for ep, n, st, save in plan:
        ckpt = f"{RUN}/checkpoint_{'final' if ep == 'final' else 'epoch_' + ep}.pt"
        out = save or f"{TMP}/l1_{ep}.csv"
        try:
            t0 = time.time(); gen(ckpt, out, n, st)
            f = load_changes(out); sp = f[:, 0, :]
            k = float(kurtosis(sp.ravel())); sk = float(skew(sp.ravel())); std = float(sp.std())
            reg, _ = regime_block(sp, rr, VOL_WINDOW)
            rl = reg["mean_run_len"]["fake_mean"]; hv = reg["high_vol_frac"]["fake_mean"]
            traj.append({"ep": ep, "kurt": k, "skew": sk, "std": std, "run_len": rl, "high_vol": hv})
            log(f"  ep{ep} ({time.time()-t0:.0f}s): 峰度={k:.2f} 偏度={sk:.2f} std={std:.4f} "
                f"run_len={rl:.1f}(真33) high_vol={hv:.3f}(0.5)")
        except Exception as e:
            log(f"  [ERR] ep{ep}: {e}")
        finally:
            if save is None and os.path.exists(out):
                os.remove(out)
    json.dump({"real_kurt": real_k, "traj": traj}, open(f"{REPO}/eval/line1_clip15_check.json", "w"),
              indent=2, default=float)
    if traj:
        kf = traj[-1]["kurt"]
        log(f"\n判读: 峰度轨迹 {[round(t['kurt'],1) for t in traj]} (旧clip8封顶~8, clip15天花板12.9)")
        if kf > 9.5:
            log(f"  ✅ GO: final 峰度 {kf:.1f} 明显超旧 clip8 封顶 → clip15 让模型学到更厚尾, 厚尾战役有效")
        elif kf > 8.3:
            log(f"  ◐ 边界: final 峰度 {kf:.1f} 轻微超封顶 → clip15 有效但有限, 看三栏 tail_kurt 是否进带")
        else:
            log(f"  ❌ NO-GO: final 峰度 {kf:.1f} 仍卡~8 → eps-MSE 吞极端样本, clip 非充分 → 转 G3 kurt损失/G5 v-pred")
    log("完成。final 样本存 output/line1/l1_clip15.csv (供 realism_board/scoreboard 三栏验收)")


if __name__ == "__main__":
    main()
