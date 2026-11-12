#!/usr/bin/env bash
# E8 只追加 complement 的数量曲线（DEVLOG S20）：E2a 20k + 1500 / 3000 / 6000 条 complement 真实样本，
# 不加 irrelevant、不用软标签；3 个种子，每个模型对调参集和测试集打分（先验调整与排序指标在本地算）。
# 三个规模嵌套：先构造 6000，再构造 3000 与 1500 并断言是 6000 的子集；3000 还断言是 E7a 追加样本的子集
# （同一种子、同一排除集合），所以 E8-3000 与 E7a 之间只差 3000 条 irrelevant。
# 顺序：先跑完种子 42 的三个规模（尽早看到曲线形状），再跑 43、44。
#   setsid nohup bash scripts/slurm/run_e8.sh >> logs/chain_e8.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
E1="$DATA_ROOT/e1_relevance/splits"
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
SCORES="$OUTPUT_ROOT/label_scores/e7-auditor-on-e2a-train.jsonl"   # --threshold 1.01 时不会用到，但参数必填
log() { echo "$(date +%T) [e8] $*"; }
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
prep e7-data $COMMON --add complement=6000 --out-dir "$DATA_ROOT/e8_c6000"
prep e7-data $COMMON --add complement=3000 --out-dir "$DATA_ROOT/e8_c3000" --assert-subset-of "$DATA_ROOT/e8_c6000" "$DATA_ROOT/e7a_ci_only"
prep e7-data $COMMON --add complement=1500 --out-dir "$DATA_ROOT/e8_c1500" --assert-subset-of "$DATA_ROOT/e8_c3000"

for seed in 42 43 44; do
  for n in 1500 3000 6000; do
    RUN=e8-c$n-s$seed
    export TRAIN_OVERRIDES="seed=$seed"
    RUN=$RUN CONFIG=$CFG DATASET_DIR="$DATA_ROOT/e8_c$n/llamafactory" EVALS=none bash scripts/slurm/chain.sh || { log "train $RUN failed"; continue; }
    unset TRAIN_OVERRIDES
    score $RUN "$TUNE" $RUN-tune
    score $RUN "$TEST" $RUN-test
  done
done
log "e8 all done"
