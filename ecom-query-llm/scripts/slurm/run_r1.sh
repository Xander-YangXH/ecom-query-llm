#!/usr/bin/env bash
# R1：在扩大的测试集（2,500 个 query）上重新评测基线、E1、E2a、E2b，不重新训练（见 DEVLOG S11）。
#   setsid nohup bash scripts/slurm/run_r1.sh >> logs/chain_r1.log 2>&1 < /dev/null &
# 只调用 squeue/sacct/sbatch；QoS 每人最多同时提交 2 个作业，所以逐个提交、逐个等待。
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
OUT="$DATA_ROOT/esci_test2500"
T="$OUT/splits/test.jsonl"
log() { echo "$(date +%T) [r1] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 30; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}

# 1) 只生成新测试集：与 E1 相同的种子和 per_query，test_queries 600 -> 2500
export PREP_ARGS="--out-dir $OUT --test-queries 2500 --only-test"
J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch)
unset PREP_ARGS
S=$(wait_job "$J"); log "test-set job=$J state=$S"
[ "$S" = "COMPLETED" ] || { log "test-set failed, stop"; exit 1; }

# 2) 四个系统依次评测；设置与原评测一致（基线 bf16，微调模型 4bit 基座 + 适配器）
COMMON="MODEL=$MODEL,TEST=$T,BATCH_SIZE=32,MAX_NEW_TOKENS=24"
for spec in "base::" "e1:e1-qlora-r16:1" "e2a:e2a-1ep-best:1" "e2b:e2b-rebal-1ep-best:1"; do
  IFS=: read -r name run lora <<< "$spec"
  if [ -n "$lora" ]; then
    EXP="ALL,$COMMON,RUN=$run,NAME=t2500-$name,LORA=1,LOAD_4BIT=1"
  else
    EXP="ALL,$COMMON,RUN=base,NAME=t2500-$name"
  fi
  J=$(sbatch --parsable --export="$EXP" scripts/slurm/eval.sbatch)
  S=$(wait_job "$J"); log "eval $name job=$J state=$S"
done
log "r1 all done"
