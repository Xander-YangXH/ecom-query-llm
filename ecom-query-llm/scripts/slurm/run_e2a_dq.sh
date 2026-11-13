#!/usr/bin/env bash
# S27：按 N1 的打分协议（4bit NF4 + bf16 计算 + double quantization）重新给 E2a 三个种子打分，
# 使 7B 与 14B 的比较只差基座大小。历史 E2a 打分没有开启 double quantization。
# 输出 $OUTPUT_ROOT/label_scores/e2a-dq-s{42,43,44}-{tune,test}.jsonl，旁边写 .protocol 标记。
#   setsid nohup bash scripts/slurm/run_e2a_dq.sh >> logs/chain_e2a_dq.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
source scripts/slurm/sbatch_retry.sh
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
LS="$OUTPUT_ROOT/label_scores"
log() { echo "$(date +%T) [e2a-dq] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}
for spec in 42:e2a-1ep-best 43:e5-20k-s43 44:e5-20k-s44; do
  seed=${spec%%:*}; run=${spec#*:}
  for part in tune test; do
    name=e2a-dq-s$seed-$part
    if [ -s "$LS/$name.jsonl" ]; then log "$name exists, skip"; continue; fi
    input=$TUNE; [ $part = test ] && input=$TEST
    J=$(sbatch_retry --parsable --export=ALL,MODEL=Qwen2.5-7B-Instruct,RUN=$run,TEST=$input,NAME=$name,LOAD_4BIT=1,DOUBLE_QUANT=1 \
        scripts/slurm/label_scores.sbatch)
    S=$(wait_job "$J"); log "$name job=$J state=$S"
    [ "$S" = COMPLETED ] && echo "bnb 4-bit NF4, bf16 compute, double quantization enabled" > "$LS/$name.jsonl.protocol"
  done
done
log "e2a-dq all done"
