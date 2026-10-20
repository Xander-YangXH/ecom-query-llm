#!/bin/bash
# CCDS（SLURM）评测作业：用 transformers + peft 生成，再算指标。
#   sbatch --export=ALL,MODEL=Qwen2.5-7B-Instruct,RUN=smoke,LORA=1 scripts/slurm/eval.sbatch
# LORA=1 表示加载 $OUTPUT_ROOT/$RUN 作为适配器；不设则评测基座。资源数值需 <= QoS 上限。
# LOAD_4BIT=1 表示基座按 4bit 加载（与 QLoRA 训练时一致），结果文件名带 -4bit 后缀。
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# L40S 只在 TC2N07、TC2N08 上（GRES 未标显卡型号，无法用 --gres=gpu:l40s 指定），排除其余节点
#SBATCH --exclude=TC2N0[1-6]
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:30:00
#SBATCH --job-name=ecomq-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p reports

: "${MODEL:?}"; : "${RUN:?}"

set -a; source .env; set +a
module load anaconda
eval "$(conda shell.bash hook)"
conda activate ecomq
module load cuda/12.8.0
if [ "${OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

NAME="$RUN${LOAD_4BIT:+-4bit}"
PRED="$OUTPUT_ROOT/preds/$NAME.jsonl"
LORA_ARG=()
[ -n "${LORA:-}" ] && LORA_ARG=(--lora "$OUTPUT_ROOT/$RUN")
[ -n "${LOAD_4BIT:-}" ] && LORA_ARG+=(--load-4bit)

python -m ecomq.eval.generate_hf \
  --model "$MODEL_ROOT/$MODEL" "${LORA_ARG[@]}" \
  --test "$DATA_ROOT/splits/test.jsonl" --output "$PRED"

python -m ecomq.eval.score --test "$DATA_ROOT/splits/test.jsonl" --preds "$PRED" \
  --output "reports/$NAME.json"
