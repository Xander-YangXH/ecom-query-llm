#!/usr/bin/env bash
# N3：7B 全参数微调（DEVLOG S32）。在登录节点运行，只提交作业：4 卡训练 → 调参集与测试集打分（bf16 完整模型）。
#   LR=1e-5 SEED=42 bash scripts/pbs/run_n3.sh        # RUN 默认 n3-full-lr1e-5-s42
#   SMOKE=1 bash scripts/pbs/run_n3.sh                # 只训练 30 步，看显存和速度，不打分
# 对照组是 NSCC 上的 E2a 三个种子（S31，n0-e2a-s{42,43,44}）。
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
LR="${LR:-1e-5}"; SEED="${SEED:-42}"
MODEL="${MODEL:-Qwen2.5-7B-Instruct}"
[ -s "$DATA_ROOT/e1_relevance/llamafactory/dataset_info.json" ] || { echo "先运行 scripts/pbs/submit.sh prepare_data" >&2; exit 1; }
if [ -n "${SMOKE:-}" ]; then
  RUN="${RUN:-n3-smoke}"
  export TRAIN_OVERRIDES="learning_rate=$LR seed=$SEED max_steps=30 eval_steps=15 save_steps=15 save_total_limit=1"
  scripts/pbs/submit.sh train_full TRAIN_OVERRIDES CONFIG=configs/sft/n3_full_7b.yaml MODEL="$MODEL" RUN="$RUN"
  exit
fi
RUN="${RUN:-n3-full-lr$LR-s$SEED}"
export TRAIN_OVERRIDES="learning_rate=$LR seed=$SEED"
T=$(scripts/pbs/submit.sh train_full TRAIN_OVERRIDES CONFIG=configs/sft/n3_full_7b.yaml MODEL="$MODEL" RUN="$RUN")
echo "train        $T"
for part in tune test; do
  case $part in tune) F="$DATA_ROOT/esci_tune/splits/tune.jsonl";; test) F="$DATA_ROOT/esci_test2500/splits/test.jsonl";; esac
  S=$(AFTER=$T scripts/pbs/submit.sh label_scores MODEL_PATH="$OUTPUT_ROOT/$RUN/best" TEST="$F" NAME="$RUN-$part")
  echo "score $part   $S"
done
