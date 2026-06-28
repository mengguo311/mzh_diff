#!/usr/bin/env bash
# eval_v13_run.sh <run_name> [gpu] [model] — 评估一个【per-512 窗】v13 run (A1/A2 类, cond_dim=2)。
# L=512 诚实闸门(对标 calib_*_L512) + concat2048 长程基线。C1(cond_dim>2)请用 auto_eval_v13_c1.sh。
# 用法: bash eval/eval_v13_run.sh deep_v13_a2_boot 1
set -uo pipefail
cd /home/u00134/src
RUN="$1"; GPU="${2:-0}"; MODEL="${3:-}"
if [ -z "$MODEL" ]; then case "$RUN" in *dit_b*|*ditb*) MODEL=dit-b;; *) MODEL=dit-s;; esac; fi
CKPT=logs/$RUN/checkpoint_final.pt; SCAL=logs/$RUN/scaler.pt
OUT=output/${RUN}.csv; CONCAT=output/${RUN}_concat2048.csv
R="conda run --no-capture-output -n ts_diffusion python"
export CUDA_VISIBLE_DEVICES=$GPU

echo "[eval_v13] $RUN model=$MODEL GPU=$GPU $(date)"
$R -u generate.py --model $MODEL --checkpoint "$CKPT" --scaler "$SCAL" \
   --num_samples 5120 --cond_mode dataset --num_inference_steps 200 --guidance_scale 1.0 --eta 1.0 --output "$OUT"
echo "[eval_v13] $RUN —— L=512 诚实闸门"
$R eval/memorization.py   --fakes ${RUN}="$OUT" --json eval/mem_${RUN}.json
$R eval/novelty_rerank.py --fakes ${RUN}="$OUT" --json eval/rerank_${RUN}.json
$R eval/diagnostics.py    --fake "$OUT" --json eval/diag_${RUN}_L512.json --label ${RUN}
echo "[eval_v13] $RUN —— 长程基线 concat 4x512=2048"
$R eval/concat_windows.py --inp "$OUT" --out "$CONCAT" --k 4
$R eval/diagnostics.py    --fake "$CONCAT" --json eval/diag_${RUN}_concat2048.json --label ${RUN}_concat2048
echo "[eval_v13] $RUN DONE $(date)"
