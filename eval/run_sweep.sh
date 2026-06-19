#!/usr/bin/env bash
# eval/run_sweep.sh — 零成本采样扫描
#   对每个 (eta, steps, w) 配置: generate → score.py → diagnostics.py
#   产物: output/sw_<name>.csv, eval/score_sw_<name>.json, eval/diag_sw_<name>.json
#   生成已存在则跳过 (可断点续跑)。最后调 eval/aggregate.py 汇总。
#
# 用法:   bash eval/run_sweep.sh            # 默认 N=512
#          N=5120 bash eval/run_sweep.sh     # 全量复核
#
# 注: 不用 set -e —— 单个配置失败不应中断整个扫描。

cd "$(dirname "$0")/.."   # 切到 ~/src

CKPT="logs/deep_v9_dit_b_20k/checkpoint_final.pt"
SCAL="logs/deep_v9_dit_b_20k/scaler.pt"
REAL="/home/u00134/data/train_sp500_us10y.csv"
N="${N:-512}"
CONDA="conda run -n ts_diffusion python"

# name        eta  steps  w     —— eta=0 为现状(确定性), eta=1 接近 DDPM
CONFIGS=(
  "e0_s50_w3    0.0  50   3.0"
  "e1_s50_w3    1.0  50   3.0"
  "e1_s100_w3   1.0  100  3.0"
  "e1_s200_w3   1.0  200  3.0"
  "e1_s100_w1.5 1.0  100  1.5"
  "e1_s200_w1   1.0  200  1.0"
  "e1_s400_w1   1.0  400  1.0"
)

for cfg in "${CONFIGS[@]}"; do
  read -r name eta steps w <<< "$cfg"
  csv="output/sw_${name}.csv"

  if [[ -f "$csv" ]]; then
    echo "[skip gen] $csv 已存在"
  else
    echo "[gen]  $name  (eta=$eta steps=$steps w=$w N=$N)"
    $CONDA generate.py --model dit-b --checkpoint "$CKPT" --scaler "$SCAL" \
      --num_samples "$N" --num_inference_steps "$steps" --guidance_scale "$w" --eta "$eta" \
      --cond_mode dataset --output "$csv" 2>&1 | grep -E "Total time|Output:" || true
  fi

  echo "[score] $name"
  $CONDA eval/score.py --model dit-b --checkpoint "$CKPT" --scaler "$SCAL" \
    --real "$REAL" --fake "$csv" --json "eval/score_sw_${name}.json" 2>&1 | grep -E "总分" || true

  echo "[diag]  $name"
  $CONDA eval/diagnostics.py --fake "$csv" --json "eval/diag_sw_${name}.json" --label "$name" \
    2>&1 | grep -vE "FutureWarning|warnings.warn" | grep -E "mean_run_len|d2_energy" || true
done

echo ""
echo "================== 汇总 =================="
python3 eval/tools/aggregate.py
