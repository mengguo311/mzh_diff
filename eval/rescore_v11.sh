#!/usr/bin/env bash
# A.4 — 用 v11 新权重的 score.py 对 v9 / v10采样 / v10重训 重打分 (固定 v9 模型做参考)。
# 全部用同一 v9 checkpoint 作为参考流形, 跨候选可比。
set -uo pipefail
cd /home/u00134/src

CKPT=logs/deep_v9_dit_b_20k/checkpoint_final.pt
SCAL=logs/deep_v9_dit_b_20k/scaler.pt
REAL=/home/u00134/data/train_sp500_us10y.csv
RUN="conda run --no-capture-output -n ts_diffusion python eval/score.py --model dit-b --checkpoint $CKPT --scaler $SCAL --real $REAL"

echo "############ [1/3] v9_20k ############"
$RUN --fake output/deep_v9_20k.csv        --json eval/v11rw_v9_20k.json

echo "############ [2/3] v10_sampling ############"
$RUN --fake output/deep_v10.csv           --json eval/v11rw_v10_sampling.json

echo "############ [3/3] v10_retrained (+forensic 集成自检) ############"
$RUN --fake output/deep_v10_retrained.csv --json eval/v11rw_v10_retrained.json --forensic

echo "############ DONE rescore ############"
