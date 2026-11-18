#!/usr/bin/env bash
# N5：14B QLoRA 在 NSCC 上重跑（DEVLOG S35）。在登录节点运行，只提交作业：4 卡训练 → 调参集与测试集打分（4bit 基座 + 适配器）。
#   SEED=42 bash scripts/pbs/run_n5.sh       # RUN 默认 n5-14b-s42
#   SMOKE=1 bash scripts/pbs/run_n5.sh       # 只训练 30 步，看显存和速度，不打分
# 打分协议与 NSCC 上的 E2a 对照组（S30/S31）相同：4bit、不开 double quantization（S27：对 7B 没有影响）。
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
SEED="${SEED:-42}"
MODEL="${MODEL:-Qwen2.5-14B-Instruct}"
CONFIG=configs/sft/n5_qwen14b_qlora_4gpu.yaml
[ -s "$MODEL_ROOT/$MODEL/config.json" ] || { echo "先下载模型：bash scripts/download_model.sh Qwen/$MODEL" >&2; exit 1; }
if [ -n "${SMOKE:-}" ]; then
  RUN="${RUN:-n5-smoke}"
  export TRAIN_OVERRIDES="seed=$SEED max_steps=30 eval_steps=15 save_steps=15"
  scripts/pbs/submit.sh train_full TRAIN_OVERRIDES CONFIG=$CONFIG MODEL="$MODEL" RUN="$RUN"
  exit
fi
RUN="${RUN:-n5-14b-s$SEED}"
export TRAIN_OVERRIDES="seed=$SEED"
T=$(scripts/pbs/submit.sh train_full TRAIN_OVERRIDES CONFIG=$CONFIG MODEL="$MODEL" RUN="$RUN")
echo "train        $T"
for part in tune test; do
  case $part in tune) F="$DATA_ROOT/esci_tune/splits/tune.jsonl";; test) F="$DATA_ROOT/esci_test2500/splits/test.jsonl";; esac
  S=$(AFTER=$T scripts/pbs/submit.sh label_scores MODEL="$MODEL" RUN="$RUN" LOAD_4BIT=1 TEST="$F" NAME="$RUN-$part")
  echo "score $part   $S"
done
