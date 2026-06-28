#!/usr/bin/env bash
# auto_eval_v13_c1.sh <c1_run> [gpu] — C1 自回归评估 + 必做消融门 (cond_dim=16 模型)。durable。
# 流程: 自回归生成 real-ctx 与 zero-ctx 各 2048 → ctx_ablation 判 context 是否被用上
#       → (用上才有意义的) 长程诚实闸门 @2048 (对标 L=2048 标定 + v10_retrained)。
# 用法: nohup bash eval/auto_eval_v13_c1.sh deep_v13_c1_ctx 0 > logs/ae_v13_c1.log 2>&1 &
set -uo pipefail
cd /home/u00134/src
RUN="$1"; GPU="${2:-0}"
CKPT=logs/$RUN/checkpoint_final.pt; SCAL=logs/$RUN/scaler.pt
RC=output/${RUN}_realctx2048.csv; ZC=output/${RUN}_zeroctx2048.csv
R="conda run --no-capture-output -n ts_diffusion python"
export CUDA_VISIBLE_DEVICES=$GPU

echo "[c1_eval] 等 $RUN 训练完成 ... $(date)"
while [ ! -f "$CKPT" ]; do sleep 120; done
sleep 60

echo "[c1_eval] 自回归生成 real-ctx 与 zero-ctx 各 2048 (k=4)"
$R -u generate_autoregressive.py --model dit-s --checkpoint "$CKPT" --scaler "$SCAL" \
   --num_samples 5120 --k 4 --seed_ctx real --output "$RC"
$R -u generate_autoregressive.py --model dit-s --checkpoint "$CKPT" --scaler "$SCAL" \
   --num_samples 5120 --k 4 --force_null --output "$ZC"

echo "[c1_eval] 消融门: context 是否被用上 (否则停 C1)"
$R eval/ctx_ablation.py --real-ctx "$RC" --zero-ctx "$ZC" --json eval/ctx_ablation_${RUN}.json

echo "[c1_eval] 长程诚实闸门 (real-ctx 2048, 对标 L=2048 标定 + v10_retrained)"
$R eval/memorization.py   --fakes ${RUN}_ar="$RC" v10_retrained=output/deep_v10_retrained.csv --json eval/mem_${RUN}.json
$R eval/novelty_rerank.py --fakes ${RUN}_ar="$RC" v10_retrained=output/deep_v10_retrained.csv --json eval/rerank_${RUN}.json
$R eval/diagnostics.py    --fake "$RC" --json eval/diag_${RUN}_ar2048.json --label ${RUN}_ar2048
echo "[c1_eval] $RUN DONE $(date)"
