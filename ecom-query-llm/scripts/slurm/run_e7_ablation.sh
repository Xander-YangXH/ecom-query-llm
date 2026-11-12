#!/usr/bin/env bash
# E7 消融（DEVLOG S19）：等 run_e7.sh 结束后，分别只保留一种做法，各跑 3 个种子。
#   E7a：只追加 C/I（--threshold 1.01，等于关闭软标签）
#   E7b：只做软标签（--add complement=0,irrelevant=0，等于不追加）
# 复用 E7 中审计模型对 E2a 训练集的打分；其余设定与 E7 完全相同。
#   setsid nohup bash scripts/slurm/run_e7_ablation.sh >> logs/chain_e7_ablation.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
E1="$DATA_ROOT/e1_relevance/splits"
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
SCORES="$OUTPUT_ROOT/label_scores/e7-auditor-on-e2a-train.jsonl"
log() { echo "$(date +%T) [e7-abl] $*"; }
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

while pgrep -u "$USER" -f "scripts/slurm/run_e7.sh" > /dev/null; do sleep 120; done
log "e7 runner finished: $(tail -1 logs/chain_e7.log)"
[ -f "$SCORES" ] || { log "auditor scores missing, stop"; exit 1; }

COMMON="--base $E1/train.jsonl --val $E1/val.jsonl --scores $SCORES --exclude-queries-from $TUNE $TEST"
prep e7-data $COMMON --threshold 1.01 --add complement=3000,irrelevant=3000 --out-dir "$DATA_ROOT/e7a_ci_only"
prep e7-data $COMMON --threshold 0.6  --add complement=0,irrelevant=0       --out-dir "$DATA_ROOT/e7b_soft_only"

for variant in e7a_ci_only e7b_soft_only; do
  for seed in 42 43 44; do
    RUN=${variant%%_*}-s$seed
    [ "$variant" = e7a_ci_only ] && RUN=e7a-ci-only-s$seed || RUN=e7b-soft-only-s$seed
    export TRAIN_OVERRIDES="seed=$seed"
    RUN=$RUN CONFIG=$CFG DATASET_DIR="$DATA_ROOT/$variant/llamafactory" EVALS=none bash scripts/slurm/chain.sh || { log "train $RUN failed"; continue; }
    unset TRAIN_OVERRIDES
    score $RUN "$TUNE" $RUN-tune
    score $RUN "$TEST" $RUN-test
  done
done
log "e7 ablation all done"
