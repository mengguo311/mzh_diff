#!/usr/bin/env bash
# auto_eval_v11.sh — 步骤 F 自动评估链 (durable, nohup 后台抗 SSH 断)。
# 轮询等待 12h 验证训练写出 checkpoint_final.pt, 一旦就绪自动:
#   生成 5120 样本 → 非自指鉴别器(c2st/signature) → 诊断 → 重设计 score.py(--forensic)
#   → verdict_v11.py 给出步骤 F 判据初判 (步骤 G 依据)。
# 绝不用 ddpm_mse 总分判定; 判据 = 鉴别器 + 诊断 + wasserstein/峰度。
set -uo pipefail
cd /home/u00134/src

CKPT=logs/deep_v11_dit_b_val/checkpoint_final.pt
SCAL=logs/deep_v11_dit_b_val/scaler.pt
REAL=/home/u00134/data/train_sp500_us10y.csv
V9CK=logs/deep_v9_dit_b_20k/checkpoint_final.pt
V9SC=logs/deep_v9_dit_b_20k/scaler.pt
V10=output/deep_v10_retrained.csv
OUT=output/deep_v11_val.csv
RUN="conda run --no-capture-output -n ts_diffusion python"
export CUDA_VISIBLE_DEVICES=0   # 训练结束后 GPU0 空闲

echo "[auto_eval_v11] 等待训练完成 (checkpoint_final.pt) ... $(date)"
while [ ! -f "$CKPT" ]; do sleep 120; done
sleep 60   # 确保 checkpoint/scaler 落盘完成
echo "[auto_eval_v11] checkpoint 就绪, 开始评估 $(date)"

echo "[auto_eval_v11] (1/5) 生成 5120 样本 (eta=1 steps=200 w=1 cond=dataset) ..."
$RUN -u generate.py --model dit-b --checkpoint "$CKPT" --scaler "$SCAL" \
  --num_samples 5120 --cond_mode dataset --num_inference_steps 200 \
  --guidance_scale 1.0 --eta 1.0 --output "$OUT"

echo "[auto_eval_v11] (2/5) C2ST 鉴别器 (v11 vs v10_retrained) ..."
$RUN eval/c2st.py --fakes v11="$OUT" v10_retrained="$V10" --json eval/c2st_v11.json

echo "[auto_eval_v11] (3/5) Sig-MMD 鉴别器 (v11 vs v10_retrained) ..."
$RUN eval/signature.py --fakes v11="$OUT" v10_retrained="$V10" --json eval/signature_v11.json

echo "[auto_eval_v11] (4/5) 诊断 diagnostics ..."
$RUN eval/diagnostics.py --fake "$OUT" --json eval/diag_v11_val.json --label v11_val

echo "[auto_eval_v11] (5/5) 重设计 score.py (固定 v9 参考) + forensic ..."
$RUN eval/score.py --model dit-b --checkpoint "$V9CK" --scaler "$V9SC" --real "$REAL" \
  --fake "$OUT" --json eval/v11rw_v11_val.json --forensic

echo "[auto_eval_v11] ===== 步骤 F 判据汇总 ====="
$RUN eval/verdict_v11.py

echo "[auto_eval_v11] DONE $(date)"
