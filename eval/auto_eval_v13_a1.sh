#!/usr/bin/env bash
# auto_eval_v13_a1.sh — A1(缩窗 L=512)自动评估 (durable, nohup)。
# 轮询等训练完成 → 生成 L=512 样本 → 【L=512 诚实闸门】memorization + novelty_rerank + diagnostics
#   → 长程基线: 朴素拼接 4x512=2048 → diagnostics vs real-2048。
# 判据(对标 L=512 标定: 复制地板 0.0% / C2ST_cal 0.496 / SigP_cal 0.116):
#   合格 = 复制率 << 36.5% 基线; C2ST_novel < 0.755 且趋近 0.496; SigP_novel 不低于 ~0.116;
#         diagnostics@512 roughness/regime 不退化。长程 concat2048 仅作 C1 待修量化, 非合格门。
set -uo pipefail
cd /home/u00134/src
CKPT=logs/deep_v13_a1_L512/checkpoint_final.pt
SCAL=logs/deep_v13_a1_L512/scaler.pt
OUT=output/deep_v13_a1_L512.csv
CONCAT=output/deep_v13_a1_concat2048.csv
RUN="conda run --no-capture-output -n ts_diffusion python"
export CUDA_VISIBLE_DEVICES=0

echo "[ae_v13_a1] 等 A1 训练完成 (checkpoint_final.pt) ... $(date)"
while [ ! -f "$CKPT" ]; do sleep 120; done
sleep 60
echo "[ae_v13_a1] 生成 L=512 样本 (5120, eta=1 steps=200 w=1) $(date)"
$RUN -u generate.py --model dit-s --checkpoint "$CKPT" --scaler "$SCAL" \
  --num_samples 5120 --cond_mode dataset --num_inference_steps 200 --guidance_scale 1.0 --eta 1.0 --output "$OUT"

echo "[ae_v13_a1] ===== L=512 诚实闸门 (对标 L=512 标定) ====="
$RUN eval/memorization.py  --fakes a1_L512="$OUT" --json eval/mem_deep_v13_a1.json
$RUN eval/novelty_rerank.py --fakes a1_L512="$OUT" --json eval/rerank_deep_v13_a1.json
$RUN eval/diagnostics.py --fake "$OUT" --json eval/diag_deep_v13_a1_L512.json --label v13_a1_L512

echo "[ae_v13_a1] ===== 长程基线: 朴素拼接 4x512=2048 → diagnostics vs real-2048 ====="
$RUN eval/concat_windows.py --inp "$OUT" --out "$CONCAT" --k 4
$RUN eval/diagnostics.py --fake "$CONCAT" --json eval/diag_deep_v13_a1_concat2048.json --label v13_a1_concat2048

echo "[ae_v13_a1] DONE $(date)"
