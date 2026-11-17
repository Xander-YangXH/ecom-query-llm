#!/usr/bin/env bash
# N0：在 NSCC 上复现 E2a seed 42（训练 → 调参集与测试集的标签打分）。在登录节点运行，只提交作业，不做计算。
# 三个作业用 PBS 依赖串起来，提交完就可以退出登录。打分协议与原 E2a 相同（4bit，不开 double quantization）。
#   bash scripts/pbs/run_n0.sh            # RUN 默认 n0-e2a-s42
#   SEED=43 bash scripts/pbs/run_n0.sh    # 其他种子（只改 seed，同 E5），RUN 默认 n0-e2a-s43
# 之后在本机取回打分并做先验调整：
#   REMOTE_HOST=<user>@<host> REMOTE_ROOT=scratch/ecomq bash scripts/local_cache.sh n0-e2a-s42
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
SEED="${SEED:-42}"
RUN="${RUN:-n0-e2a-s$SEED}"
MODEL="${MODEL:-Qwen2.5-7B-Instruct}"
[ -s "$DATA_ROOT/e1_relevance/llamafactory/dataset_info.json" ] || { echo "先运行 scripts/pbs/submit.sh prepare_data" >&2; exit 1; }
[ -s "$MODEL_ROOT/$MODEL/config.json" ] || { echo "先下载模型：bash scripts/download_model.sh Qwen/$MODEL" >&2; exit 1; }

export TRAIN_OVERRIDES="seed=$SEED"
T=$(scripts/pbs/submit.sh train TRAIN_OVERRIDES CONFIG=configs/sft/e2_relevance_qlora_1ep_best.yaml MODEL="$MODEL" RUN="$RUN" \
      DATASET_DIR="$DATA_ROOT/e1_relevance/llamafactory")
echo "train        $T"
S1=$(AFTER=$T scripts/pbs/submit.sh label_scores MODEL="$MODEL" RUN="$RUN" LOAD_4BIT=1 \
      TEST="$DATA_ROOT/esci_tune/splits/tune.jsonl" NAME="$RUN-tune")
echo "score tune   $S1"
S2=$(AFTER=$T scripts/pbs/submit.sh label_scores MODEL="$MODEL" RUN="$RUN" LOAD_4BIT=1 \
      TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl" NAME="$RUN-test")
echo "score test   $S2"
