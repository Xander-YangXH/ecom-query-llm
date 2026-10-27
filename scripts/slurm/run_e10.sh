#!/usr/bin/env bash
# E10（DEVLOG S24）：E2a 20k + 1,500 条 complement + 1,500 条 irrelevant 真实样本，不用软标签，3 个种子。
# 回答 E8 留下的问题：只加 complement 时 irrelevant 的区分能力下降，同时按比例补 irrelevant 能否保住它。
# 抽样种子与排除集合和 E7a、E8 相同，所以：
#   - E10 的 1,500 条 complement 与 E8-c1500 完全相同（E10 与 E8-c1500 只差 1,500 条 irrelevant）；
#   - E10 的追加样本是 E7a（C、I 各 3,000）追加样本的子集（作业里断言）。
#   setsid nohup bash scripts/slurm/run_e10.sh >> logs/chain_e10.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
E1="$DATA_ROOT/e1_relevance/splits"
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
SCORES="$OUTPUT_ROOT/label_scores/e7-auditor-on-e2a-train.jsonl"   # --threshold 1.01 时不会用到，但参数必填
log() { echo "$(date +%T) [e10] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}
prep() {
  export PREP_MODULE=ecomq.data.build_e7 PREP_ARGS="$*"
  local J S; J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch)
  unset PREP_MODULE PREP_ARGS
  S=$(wait_job "$J"); log "prep job=$J state=$S"
  [ "$S" = "COMPLETED" ] || { log "prep failed, stop"; exit 1; }
}
score() {
  local J S; J=$(sbatch --parsable --export=ALL,MODEL=$MODEL,RUN=$1,TEST=$2,NAME=$3,LOAD_4BIT=1 scripts/slurm/label_scores.sbatch)
  S=$(wait_job "$J"); log "scores $3 job=$J state=$S"
}

COMMON="--base $E1/train.jsonl --val $E1/val.jsonl --scores $SCORES --threshold 1.01 --exclude-queries-from $TUNE $TEST"
prep e7-data $COMMON --add complement=1500,irrelevant=1500 --out-dir "$DATA_ROOT/e10_c1500_i1500" --assert-subset-of "$DATA_ROOT/e7a_ci_only"

for seed in 42 43 44; do
  RUN=e10-c1500-i1500-s$seed
  export TRAIN_OVERRIDES="seed=$seed"
  RUN=$RUN CONFIG=$CFG DATASET_DIR="$DATA_ROOT/e10_c1500_i1500/llamafactory" EVALS=none bash scripts/slurm/chain.sh || { log "train $RUN failed"; continue; }
  unset TRAIN_OVERRIDES
  score $RUN "$TUNE" $RUN-tune
  score $RUN "$TEST" $RUN-test
done
log "e10 all done"
