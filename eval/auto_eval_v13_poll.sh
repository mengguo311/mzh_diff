#!/usr/bin/env bash
# auto_eval_v13_poll.sh <run_name> [gpu] — 通用轮询器: 等 per-512 run 训练完成 → eval_v13_run.sh。
# durable (nohup 抗 SSH 断)。用于 A2 及任何 cond_dim=2 的 v13 run。
# 用法: nohup bash eval/auto_eval_v13_poll.sh deep_v13_a2_boot 0 > logs/ae_v13_a2.log 2>&1 &
set -uo pipefail
cd /home/u00134/src
RUN="$1"; GPU="${2:-0}"
CKPT=logs/$RUN/checkpoint_final.pt
echo "[poll_v13] 等 $RUN 训练完成 (checkpoint_final.pt) ... $(date)"
while [ ! -f "$CKPT" ]; do sleep 120; done
sleep 60
bash eval/eval_v13_run.sh "$RUN" "$GPU"
echo "[poll_v13] $RUN ALL DONE $(date)"
