#!/bin/bash
# 数据处理作业（CPU 工作）：ESCI → relevance 数据集；ECInstruct 统计与泄漏检查。
#   sbatch scripts/slurm/prepare_data.sbatch
# 集群要求作业必须申请 GPU（--gres），这里 GPU 实际不使用；不限定 L40S 节点，避免占用稀缺的 L40S。
# 头节点禁止运行计算代码，所以数据处理放在这里而不是头节点上。
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --job-name=ecomq-prepdata
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
set -a; source .env; set +a
module load anaconda
eval "$(conda shell.bash hook)"
conda activate ecomq

export PYTHONPATH=src
# 耗时用 SECONDS 记录；峰值内存事后用 sacct -j <作业号> -o MaxRSS 查看
START=$SECONDS
python -m ecomq.data.prepare_esci ${PREP_ARGS:-}
echo "prepare_esci: $((SECONDS - START))s"
if [ "${SKIP_INSPECT:-0}" != "1" ]; then
  START=$SECONDS
  python -m ecomq.data.inspect_ecinstruct
  echo "inspect_ecinstruct: $((SECONDS - START))s"
fi
