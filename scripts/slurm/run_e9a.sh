#!/usr/bin/env bash
# E9a 第二阶段（DEVLOG S23）：全量理由 → 构造数据 → 推理 SFT（单种子）→ 先写理由再打分的评测。
# 1. 理由：E2a 训练集（2 万条）与验证集，v3 提示词 + honest-first + 盲测（盲测给出“理由与标签矛盾”的标记）；
#    训练集的作业顺便把 score_items 重构后的打分与旧代码逐值比对（VERIFY_OLD）。
# 2. build_e9：丢弃仍含标签词等质量标记的样本，矛盾样本保留并标记，检查 token 长度不超过 448。
# 3. 训练 e9a-cot-s42（E2a 设定，cutoff_len 448），按设计先跑一个种子。
# 4. 调参集、测试集：先生成理由再以理由为条件打分（COT=1）；先验调整、三种口径和排序指标在本地算。
#   setsid nohup bash scripts/slurm/run_e9a.sh >> logs/chain_e9a.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
E1="$DATA_ROOT/e1_relevance/splits"
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
RAT="$OUTPUT_ROOT/rationales"
OUT="$DATA_ROOT/e9a_cot"
RUN=e9a-cot-s42
log() { echo "$(date +%T) [e9a] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}

J1=$(sbatch --parsable --time=04:00:00 --export=ALL,INPUT="$E1/train.jsonl",NAME=e2a-train,N=0,BLIND=1,HONEST=1,VERIFY_SCORES=1,VERIFY_OLD="$RAT/old_score_labels.py" scripts/slurm/rationales.sbatch)
J2=$(sbatch --parsable --export=ALL,INPUT="$E1/val.jsonl",NAME=e1-val,N=0,BLIND=1,HONEST=1 scripts/slurm/rationales.sbatch)
S1=$(wait_job "$J1"); log "rationales train job=$J1 state=$S1"
S2=$(wait_job "$J2"); log "rationales val job=$J2 state=$S2"
[ "$S1" = COMPLETED ] && [ "$S2" = COMPLETED ] || { log "rationales failed, stop"; exit 1; }

export PREP_MODULE=ecomq.data.build_e9
export PREP_ARGS="--train $RAT/e2a-train.jsonl --val $RAT/e1-val.jsonl --out-dir $OUT --tokenizer $MODEL_ROOT/$MODEL --cutoff-len 448"
J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch); unset PREP_MODULE PREP_ARGS
S=$(wait_job "$J"); log "build_e9 job=$J state=$S"
[ "$S" = COMPLETED ] || { log "build_e9 failed, stop"; exit 1; }

export TRAIN_OVERRIDES="seed=42"
RUN=$RUN CONFIG=configs/sft/e9a_cot_qlora_1ep_best.yaml DATASET_DIR="$OUT/llamafactory" EVALS=none bash scripts/slurm/chain.sh || { log "train failed, stop"; exit 1; }
unset TRAIN_OVERRIDES

for part in tune test; do
  T=$TUNE; [ "$part" = test ] && T=$TEST
  J=$(sbatch --parsable --time=03:00:00 --export=ALL,MODEL=$MODEL,RUN=$RUN,TEST="$T",NAME=$RUN-$part,LOAD_4BIT=1,COT=1 scripts/slurm/label_scores.sbatch)
  S=$(wait_job "$J"); log "scores $RUN-$part job=$J state=$S"
done
log "e9a all done"
