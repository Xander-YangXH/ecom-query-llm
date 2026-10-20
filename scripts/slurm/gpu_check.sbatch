#!/bin/bash
# 在计算节点上验证环境（几分钟）：GPU 型号与驱动、PyTorch 能否用 CUDA、bf16、bitsandbytes 4bit、计算节点外网。
#   mkdir -p logs && sbatch scripts/slurm/gpu_check.sbatch
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# L40S 只在 TC2N07、TC2N08 上（GRES 未标显卡型号，无法用 --gres=gpu:l40s 指定），排除其余节点
#SBATCH --exclude=TC2N0[1-6]
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=ecomq-gpucheck
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate ecomq
module load cuda/12.8.0

echo "== node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

python - <<'EOF'
import torch
print("torch", torch.__version__, "| built with CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).shape)

import bitsandbytes as bnb
lin = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
y = lin(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
print("bitsandbytes", bnb.__version__, "4bit forward ok:", tuple(y.shape))

import transformers, peft
print("transformers", transformers.__version__, "| peft", peft.__version__)
try:
    import llamafactory
    print("llamafactory", llamafactory.__version__)
except Exception as e:
    print("llamafactory import failed:", e)
EOF

echo "== compute node internet"
curl -sS -m 8 -o /dev/null -w "huggingface:%{http_code}\n" https://huggingface.co || echo "huggingface: unreachable"
