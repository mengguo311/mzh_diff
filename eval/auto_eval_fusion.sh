#!/usr/bin/env bash
# eval/auto_eval_fusion.sh — v14-fusion 终裁评估 (镜像 C1 协议 + 新增 forensic_cross 跨通道鉴别)。
# 用法: bash eval/auto_eval_fusion.sh <RUN> [GPU] [CKPT_TAG]
#   RUN=fusion/v14_3ch_clip20_boot_ctx  CKPT_TAG=final|epoch_1499|...
# honest gates 经 load_changes 读 sp500+DGS10 子空间 → 对标 L=2048 2ch 标尺, 与 C1 直接可比;
# DGS2/收益率曲线/跨通道结构 由 forensic_cross 单独评 (新鉴别器)。
set -uo pipefail
cd /home/u00134/src
RUN="${1:-fusion/v14_3ch_clip20_boot_ctx}"; GPU="${2:-0}"; TAG="${3:-final}"
CKPT="logs/$RUN/checkpoint_${TAG}.pt"; [ "$TAG" = "final" ] && CKPT="logs/$RUN/checkpoint_final.pt"
SCAL="logs/$RUN/scaler.pt"
SAFE=$(echo "${RUN}_${TAG}" | tr '/' '_')
RC="output/${SAFE}_realctx2048.csv"; ZC="output/${SAFE}_zeroctx2048.csv"
RG="conda run --no-capture-output -n ts_diffusion python"
export CUDA_VISIBLE_DEVICES=$GPU
echo "[fusion_eval] RUN=$RUN TAG=$TAG CKPT=$CKPT  $(date)"
[ -f "$CKPT" ] || { echo "[fusion_eval] 缺 checkpoint $CKPT"; exit 1; }

echo "[fusion_eval] 自回归生成 real-ctx + zero-ctx 各 5120 (k=4 → 2048), x0钳位±20σ"
CONFIG_PROFILE=fusion $RG -u generate_autoregressive.py --model dit-s --checkpoint "$CKPT" --scaler "$SCAL" \
   --num_samples 5120 --k 4 --seed_ctx real --x0_clamp 20 --output "$RC"
CONFIG_PROFILE=fusion $RG -u generate_autoregressive.py --model dit-s --checkpoint "$CKPT" --scaler "$SCAL" \
   --num_samples 5120 --k 4 --force_null --x0_clamp 20 --output "$ZC"

echo "[fusion_eval] 消融门: context 是否被用上 (否则 context-cond=no-op)"
$RG eval/ctx_ablation.py --real-ctx "$RC" --zero-ctx "$ZC" --json eval/ctx_ablation_${SAFE}.json

echo "[fusion_eval] 诚实闸门 (2ch子空间 sp500+DGS10, 对标 L=2048 标尺 + v10_retrained)"
$RG eval/memorization.py   --fakes ${SAFE}="$RC" v10_retrained=output/deep_v10_retrained.csv --json eval/mem_${SAFE}.json
$RG eval/novelty_rerank.py --fakes ${SAFE}="$RC" v10_retrained=output/deep_v10_retrained.csv --json eval/rerank_${SAFE}.json
$RG eval/diagnostics.py    --fake "$RC" --json eval/diag_${SAFE}_ar2048.json --label ${SAFE}_ar2048

echo "[fusion_eval] forensic_suite 单文件终裁 (复制率/C2ST_新颖/SigP_新颖/峰度/逐行发散, n_perm=300)"
$RG forensic_suite.py --candidate "$RC" --label ${SAFE} --no-fig

echo "[fusion_eval] forensic_cross 跨通道鉴别器 (新, 3ch 收益率曲线取证)"
CONFIG_PROFILE=fusion $RG eval/forensic_cross.py --fakes ${SAFE}="$RC" --json eval/forensic_cross_${SAFE}.json

echo "[fusion_eval] DONE $(date)"
