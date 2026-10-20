#!/bin/bash
# CCDS GPU Cluster TC2（SLURM）训练作业。提交前先 mkdir -p logs，否则日志文件无法创建。
#   sbatch --export=ALL,CONFIG=configs/sft/smoke_qlora.yaml,MODEL=Qwen2.5-7B-Instruct,RUN=smoke scripts/slurm/train.sbatch
# 6 小时到点后，用同样的命令（相同 RUN）重新提交即可从最新 checkpoint 续训。
# 资源数值需 <= 你的 QoS 上限，请用 mytcinfo 确认；超限的作业会一直 PENDING。
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# L40S 只在 TC2N07、TC2N08 上（GRES 未标显卡型号，无法用 --gres=gpu:l40s 指定），排除其余节点
#SBATCH --exclude=TC2N0[1-6]
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:55:00
#SBATCH --job-name=ecomq-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

: "${CONFIG:?用 --export=ALL,CONFIG=... 指定配置}"
: "${MODEL:?用 --export=ALL,MODEL=... 指定 MODEL_ROOT 下的模型目录名}"
: "${RUN:?用 --export=ALL,RUN=... 指定本次实验名}"

set -a; source .env; set +a
module load anaconda
eval "$(conda shell.bash hook)"
conda activate ecomq
module load cuda/12.8.0

# 计算节点是否有外网指南未说明；模型和数据已提前下载时可 OFFLINE=1 避免联网超时
if [ "${OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
export WANDB_MODE=offline

nvidia-smi --query-gpu=name,memory.total --format=csv

# 每 15 秒记录一次显存占用，用于估算 batch 大小与正式训练的资源需求
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 15 \
  > "logs/gpu_mem_${SLURM_JOB_ID}.csv" &
GPU_LOGGER=$!
trap 'kill $GPU_LOGGER 2>/dev/null || true' EXIT

# 冒烟测试时 MAKE_SMOKE=1：在计算节点上生成冒烟数据（头节点不允许运行计算代码）
if [ "${MAKE_SMOKE:-0}" = "1" ]; then
  PYTHONPATH=src python -m ecomq.data.make_smoke --n 400
fi

llamafactory-cli train "$CONFIG" \
  model_name_or_path="$MODEL_ROOT/$MODEL" \
  dataset_dir="$DATA_ROOT/llamafactory" \
  output_dir="$OUTPUT_ROOT/$RUN"
