#!/usr/bin/env bash
# E3：推理时先验调整（DEVLOG S12）。生成调参集 → 对 E2a、E2b 在调参集和 2500 query 测试集上计算四个标签的对数概率。
# 偏置搜索与对比在本机用 ecomq.eval.prior_adjust / compare 完成（只是 numpy 计算，不占 GPU）。
#   setsid nohup bash scripts/slurm/run_e3.sh >> logs/chain_e3.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
TUNE_DIR="$DATA_ROOT/esci_tune"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
log() { echo "$(date +%T) [e3] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 30; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}

# 1) 调参集：从官方 train 池抽 1500 个 query，排除 E1 的 train/val（同种子自动排除）以及 E2b 训练集用过的 query
export PREP_ARGS="--out-dir $TUNE_DIR --tune-queries 1500 --only-tune --tune-exclude $DATA_ROOT/e2_rebal/splits/train.jsonl"
J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch)
unset PREP_ARGS
S=$(wait_job "$J"); log "tune-set job=$J state=$S"
[ "$S" = "COMPLETED" ] || { log "tune-set failed, stop"; exit 1; }

# 2) 标签对数概率：与训练一致，用 4bit 基座 + 适配器
for spec in "e2a:e2a-1ep-best" "e2b:e2b-rebal-1ep-best"; do
  IFS=: read -r name run <<< "$spec"
  for part in tune test; do
    if [ "$part" = tune ]; then F="$TUNE_DIR/splits/tune.jsonl"; else F="$TEST"; fi
    J=$(sbatch --parsable --export=ALL,MODEL=$MODEL,RUN=$run,TEST=$F,NAME=$name-$part,LOAD_4BIT=1 scripts/slurm/label_scores.sbatch)
    S=$(wait_job "$J"); log "scores $name-$part job=$J state=$S"
    [ "$S" = "COMPLETED" ] || { log "scoring failed, stop"; exit 1; }
  done
done
log "e3 all done"
