#!/bin/bash
# run_e4_chain.sh — E4 自动链: 等预训完成 → US 微调(--init_from) → 生成(native512 + AR2048) → forensic 评分。
# 全程 GPU1。用法: CUDA_VISIBLE_DEVICES=1 nohup bash run_e4_chain.sh > logs/e4_chain.log 2>&1 &
set -u
cd /home/u00134/src
RUN() { echo "[chain $(date +%H:%M:%S)] $*"; }
CONDA="conda run --no-capture-output -n ts_diffusion python -u"

PRE=logs/e4/pretrain_multiasset/checkpoint_final.pt
RUN "等待预训完成: $PRE"
while [ ! -f "$PRE" ]; do sleep 60; done
sleep 30
RUN "预训完成 → 阶段2 US 微调 (--init_from)"

CONFIG_PROFILE=e4_finetune $CONDA train.py --model dit-s --batch_size 128 \
  --init_from "$PRE" --run_name e4/finetune_us > logs/e4_finetune.log 2>&1
FT=logs/e4/finetune_us
if [ ! -f "$FT/checkpoint_final.pt" ]; then RUN "微调失败, 无 checkpoint_final; 退出"; exit 1; fi
RUN "微调完成 → 生成"

mkdir -p output/e4
# native L=512 (real-context 条件)
CONFIG_PROFILE=e4_finetune $CONDA generate.py --model dit-s \
  --checkpoint $FT/checkpoint_final.pt --scaler $FT/scaler.pt \
  --num_samples 5120 --cond_mode dataset --output output/e4/e4_native512.csv > logs/e4_gen_native.log 2>&1
RUN "native512 生成完成 → forensic"
CONFIG_PROFILE=e4_finetune $CONDA forensic_suite.py \
  --candidate output/e4/e4_native512.csv --label e4_multiasset_native512 > logs/e4_forensic_native.log 2>&1

# AR 512→2048 (与 line2 clip11 L=2048 同口径对标; x0 钳位防发散)
CONFIG_PROFILE=e4_finetune $CONDA generate_autoregressive.py --model dit-s \
  --checkpoint $FT/checkpoint_final.pt --scaler $FT/scaler.pt \
  --num_samples 5120 --k 4 --seed_ctx real --x0_clamp 11 \
  --output output/e4/e4_ar2048.csv > logs/e4_gen_ar.log 2>&1
RUN "AR2048 生成完成 → forensic"
CONFIG_PROFILE=e4_finetune $CONDA forensic_suite.py \
  --candidate output/e4/e4_ar2048.csv --label e4_multiasset_ar2048 > logs/e4_forensic_ar.log 2>&1

RUN "E4 全链完成。forensic: eval/forensic_out/e4_multiasset_{native512,ar2048}/"
grep -E "复制率|C2ST_新颖|SigP_新颖|综合诚实" logs/e4_forensic_native.log logs/e4_forensic_ar.log 2>/dev/null
