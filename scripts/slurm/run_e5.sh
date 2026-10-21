#!/usr/bin/env bash
# 夜间队列（DEVLOG S14）：等 E4 调度结束后依次运行
#   E5：20k 数据、E2a 设定，换随机种子 43、44 各训练一次 → 估计“只换种子”带来的波动，用来判断数据量曲线上的差异是否可信
#   E4 延伸：在 100k 基础上嵌套扩充到 200k（50,000 个 query），看曲线是否饱和
# 每个模型都在调参集与 2500 query 测试集上计算四个标签的对数概率；偏置搜索与对比在本机完成。
#   setsid nohup bash scripts/slurm/run_e5.sh >> logs/chain_e5.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
log() { echo "$(date +%T) [e5] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}
scores() {  # scores <run>
  local part F J S
  for part in tune test; do
    if [ "$part" = tune ]; then F="$TUNE"; else F="$TEST"; fi
    J=$(sbatch --parsable --export=ALL,MODEL=$MODEL,RUN=$1,TEST=$F,NAME=$1-$part,LOAD_4BIT=1 scripts/slurm/label_scores.sbatch)
    S=$(wait_job "$J"); log "scores $1-$part job=$J state=$S"
  done
}

# 0) 等 E4 调度进程结束（QoS 每人最多同时提交 2 个作业，不能和 E4 并行提交）
while pgrep -u "$USER" -f "scripts/slurm/run_e4.sh" > /dev/null; do sleep 120; done
log "e4 runner finished: $(tail -1 logs/chain_e4.log)"

# 1) 种子方差：只改 seed（影响 LoRA 初始化与数据打乱顺序），数据与其余设定同 E2a
for seed in 43 44; do
  export TRAIN_OVERRIDES="seed=$seed"
  RUN=e5-20k-s$seed CONFIG=$CFG DATASET_DIR="$DATA_ROOT/e1_relevance/llamafactory" EVALS=none bash scripts/slurm/chain.sh || log "seed $seed chain failed"
  unset TRAIN_OVERRIDES
  scores e5-20k-s$seed
done

# 2) 200k：在 100k 基础上嵌套扩充；评测频率每 1000 步（约 6 次）；超过 6 小时由 chain.sh 自动续训
export PREP_ARGS="--out-dir $DATA_ROOT/esci_train200k --train-queries 50000 --grow-from $DATA_ROOT/esci_train100k/splits/train.jsonl --exclude-queries-from $TUNE --skip-test"
J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch)
unset PREP_ARGS
S=$(wait_job "$J"); log "prep esci_train200k job=$J state=$S"
[ "$S" = "COMPLETED" ] || { log "prep 200k failed, stop"; exit 1; }
cmp -s "$DATA_ROOT/e1_relevance/splits/val.jsonl" "$DATA_ROOT/esci_train200k/splits/val.jsonl" || { log "val differs, stop"; exit 1; }
export TRAIN_OVERRIDES="eval_steps=1000 save_steps=1000"
RUN=e4-200k CONFIG=$CFG DATASET_DIR="$DATA_ROOT/esci_train200k/llamafactory" EVALS=none bash scripts/slurm/chain.sh
unset TRAIN_OVERRIDES
scores e4-200k
log "e5 all done"
