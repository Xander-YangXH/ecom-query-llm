#!/usr/bin/env bash
# E4：数据量曲线（DEVLOG S13）。训练集嵌套 20k ⊂ 50k ⊂ 100k；20k 即 E2a（复用已有模型与打分）。
# 每个新点：训练（E2a 设定）→ 在调参集与 2500 query 测试集上计算四个标签的对数概率；偏置搜索与对比在本机完成。
#   setsid nohup bash scripts/slurm/run_e4.sh >> logs/chain_e4.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
log() { echo "$(date +%T) [e4] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}
prep() {  # prep <out_dir> <train_queries> <grow_from>
  export PREP_ARGS="--out-dir $1 --train-queries $2 --grow-from $3 --exclude-queries-from $TUNE --skip-test"
  local J; J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch)
  unset PREP_ARGS
  local S; S=$(wait_job "$J"); log "prep $(basename "$1") job=$J state=$S"
  [ "$S" = "COMPLETED" ] || { log "prep failed, stop"; exit 1; }
  cmp -s "$DATA_ROOT/e1_relevance/splits/val.jsonl" "$1/splits/val.jsonl" || { log "val differs from E1, stop"; exit 1; }
}

# 1) 数据：20k(E1) -> 50k -> 100k，嵌套扩充；val 与 E1 逐字节一致
prep "$DATA_ROOT/esci_train50k"  12500 "$DATA_ROOT/e1_relevance/splits/train.jsonl"
prep "$DATA_ROOT/esci_train100k" 25000 "$DATA_ROOT/esci_train50k/splits/train.jsonl"

# 2) 训练 + 打分；评测频率随数据量等比例放大，每次训练都评测约 6 次
for spec in "50k:250" "100k:500"; do
  IFS=: read -r size steps <<< "$spec"
  export TRAIN_OVERRIDES="eval_steps=$steps save_steps=$steps"
  RUN=e4-$size CONFIG=$CFG DATASET_DIR="$DATA_ROOT/esci_train$size/llamafactory" EVALS=none bash scripts/slurm/chain.sh
  unset TRAIN_OVERRIDES
  for part in tune test; do
    if [ "$part" = tune ]; then F="$TUNE"; else F="$TEST"; fi
    J=$(sbatch --parsable --export=ALL,MODEL=$MODEL,RUN=e4-$size,TEST=$F,NAME=e4-$size-$part,LOAD_4BIT=1 scripts/slurm/label_scores.sbatch)
    S=$(wait_job "$J"); log "scores e4-$size-$part job=$J state=$S"
    [ "$S" = "COMPLETED" ] || { log "scoring failed, stop"; exit 1; }
  done
done
log "e4 all done"
