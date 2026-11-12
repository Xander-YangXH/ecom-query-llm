#!/usr/bin/env bash
# E7（DEVLOG S19）：追加 complement/irrelevant 真实样本 + 有争议 E/S 的软标签，种子 42/43/44，对照 E2a 三个种子。
#   1) 审计模型数据：50k 训练集的第 20000–40000 行（与 E2a 训练集无共同 query）
#   2) 训练审计模型（E2a 设定）→ 给 E2a 训练集的 2 万条打分
#   3) 构造 E7 训练集（软标签副本 + 追加 C/I）
#   4) 三个种子：训练 → 调参集与测试集打分
#   setsid nohup bash scripts/slurm/run_e7.sh >> logs/chain_e7.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
E1="$DATA_ROOT/e1_relevance/splits"
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
AUD="$DATA_ROOT/e7_auditor"
OUT="$DATA_ROOT/e7_ci_soft"
log() { echo "$(date +%T) [e7] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}
prep() {  # prep <args...>：用 build_e7 作为数据处理模块
  export PREP_MODULE=ecomq.data.build_e7 PREP_ARGS="$*"
  local J S; J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch)
  unset PREP_MODULE PREP_ARGS
  S=$(wait_job "$J"); log "prep $1 job=$J state=$S"
  [ "$S" = "COMPLETED" ] || { log "prep failed, stop"; exit 1; }
}
score() {  # score <run> <jsonl> <name>
  local J S; J=$(sbatch --parsable --export=ALL,MODEL=$MODEL,RUN=$1,TEST=$2,NAME=$3,LOAD_4BIT=1 scripts/slurm/label_scores.sbatch)
  S=$(wait_job "$J"); log "scores $3 job=$J state=$S"
  [ "$S" = "COMPLETED" ] || { log "scoring failed, stop"; exit 1; }
}

prep auditor-data --source "$DATA_ROOT/esci_train50k/splits/train.jsonl" --start 20000 --end 40000 \
  --val "$E1/val.jsonl" --disjoint-from "$E1/train.jsonl" --out-dir "$AUD"
RUN=e7-auditor CONFIG=$CFG DATASET_DIR="$AUD/llamafactory" EVALS=none bash scripts/slurm/chain.sh
score e7-auditor "$E1/train.jsonl" e7-auditor-on-e2a-train

prep e7-data --base "$E1/train.jsonl" --val "$E1/val.jsonl" \
  --scores "$OUTPUT_ROOT/label_scores/e7-auditor-on-e2a-train.jsonl" --threshold 0.6 \
  --add complement=3000,irrelevant=3000 --exclude-queries-from "$TUNE" "$TEST" --out-dir "$OUT"

for seed in 42 43 44; do
  RUN=e7-ci-soft-s$seed
  export TRAIN_OVERRIDES="seed=$seed"
  RUN=$RUN CONFIG=$CFG DATASET_DIR="$OUT/llamafactory" EVALS=none bash scripts/slurm/chain.sh || { log "train $RUN failed"; continue; }
  unset TRAIN_OVERRIDES
  score $RUN "$TUNE" $RUN-tune
  score $RUN "$TEST" $RUN-test
done
log "e7 all done"
