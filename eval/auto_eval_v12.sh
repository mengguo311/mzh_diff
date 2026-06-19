#!/usr/bin/env bash
# auto_eval_v12.sh — v12 实验自动评估链 (durable, nohup 抗 SSH 断)。
# 对每个 run: 轮询等 checkpoint_final.pt → 生成 5120 样本 → 【诚实闸门】memorization(复制率)
#   + novelty_rerank(新颖子集 C2ST/Sig-MMD) + diagnostics(roughness/regime/PSD) → 对照 v10_retrained/v11。
# 判模型【不看 all 口径,只看 C2ST_新颖 / SigP_新颖 + 复制率】。
set -uo pipefail
cd /home/u00134/src
REAL=/home/u00134/data/train_sp500_us10y.csv
V10=output/deep_v10_retrained.csv
V11=output/deep_v11_val.csv
RUN="conda run --no-capture-output -n ts_diffusion python"

eval_run () {
  local name="$1"; local gpu="$2"
  local ckpt="logs/$name/checkpoint_final.pt"
  local scal="logs/$name/scaler.pt"
  local out="output/${name}.csv"
  local model="dit-b"; case "$name" in *dits*) model="dit-s";; esac

  echo "[auto_eval_v12] 等待 $name 训练完成 (checkpoint_final.pt) ... $(date)"
  while [ ! -f "$ckpt" ]; do sleep 120; done
  sleep 60   # 确保落盘
  echo "[auto_eval_v12] $name 就绪 (model=$model), GPU$gpu 生成+评估 $(date)"

  CUDA_VISIBLE_DEVICES=$gpu $RUN -u generate.py --model $model --checkpoint "$ckpt" --scaler "$scal" \
    --num_samples 5120 --cond_mode dataset --num_inference_steps 200 --guidance_scale 1.0 --eta 1.0 \
    --output "$out"

  echo "[auto_eval_v12] $name —— 记忆化复制率 (vs v10_retrained, v11) ..."
  $RUN eval/memorization.py  --fakes ${name}="$out" v10_retrained="$V10" v11="$V11" --json eval/mem_${name}.json
  echo "[auto_eval_v12] $name —— 诚实排名 新颖子集鉴别器 ..."
  $RUN eval/novelty_rerank.py --fakes ${name}="$out" v10_retrained="$V10" v11="$V11" --json eval/rerank_${name}.json
  echo "[auto_eval_v12] $name —— 诊断 roughness/regime/PSD ..."
  $RUN eval/diagnostics.py --fake "$out" --json eval/diag_${name}.json --label $name
  echo "[auto_eval_v12] ===== $name 评估完成 $(date) ====="
}

# DiT-S 先完成(~5h)→ 用 GPU1(其训练一结束即空); 消融后完成(~12h)→ 用 GPU0。
# 顺序调用: dits 先评(不被消融阻塞), 之后等消融完成再评。
eval_run deep_v12_dits_antimem 1
eval_run deep_v11_noSig 0
echo "[auto_eval_v12] ALL DONE $(date)"
