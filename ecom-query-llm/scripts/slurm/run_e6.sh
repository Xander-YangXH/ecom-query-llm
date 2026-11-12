#!/usr/bin/env bash
# E6：丰富商品输入（品牌、颜色、商品要点），20k 数据、E2a 设定、种子 42/43/44（DEVLOG S16）。
# 样本、标签与 E2a 完全相同，只改输入；对照组是已有的 20k 标题版三个种子（e2a-1ep-best / e5-20k-s43 / e5-20k-s44）。
#   setsid nohup bash scripts/slurm/run_e6.sh >> logs/chain_e6.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
OUT="$DATA_ROOT/e6_rich"
log() { echo "$(date +%T) [e6] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}

# 1) 补齐商品字段（样本集合与原数据逐条一致）
export PREP_MODULE=ecomq.data.enrich_esci
export PREP_ARGS="--out-dir $OUT --inputs train=$DATA_ROOT/e1_relevance/splits/train.jsonl val=$DATA_ROOT/e1_relevance/splits/val.jsonl tune=$DATA_ROOT/esci_tune/splits/tune.jsonl test=$DATA_ROOT/esci_test2500/splits/test.jsonl"
J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch)
unset PREP_MODULE PREP_ARGS
S=$(wait_job "$J"); log "enrich job=$J state=$S"
[ "$S" = "COMPLETED" ] || { log "enrich failed, stop"; exit 1; }

# 2) 三个种子：训练 → 调参集与测试集打分。输入变长，cutoff_len 384 -> 640，避免回复被截断
for seed in 42 43 44; do
  RUN=e6-rich-s$seed
  export TRAIN_OVERRIDES="seed=$seed cutoff_len=640"
  RUN=$RUN CONFIG=$CFG DATASET_DIR="$OUT/llamafactory" EVALS=none bash scripts/slurm/chain.sh || { log "train $RUN failed"; continue; }
  unset TRAIN_OVERRIDES
  for part in tune test; do
    J=$(sbatch --parsable --export=ALL,MODEL=$MODEL,RUN=$RUN,TEST=$OUT/splits/$part.jsonl,NAME=$RUN-$part,LOAD_4BIT=1,BATCH_SIZE=24 scripts/slurm/label_scores.sbatch)
    S=$(wait_job "$J"); log "scores $RUN-$part job=$J state=$S"
  done
done
log "e6 all done"
