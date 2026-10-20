#!/usr/bin/env bash
# 在 CCDS 头节点运行一次，只做安装，不跑计算（头节点禁止运行计算代码）。
# 指南建议优先 conda install，pip 仅用于 conda 没有的包；PyTorch 与 LLaMA-Factory 习惯用 pip 安装。
set -eo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "缺少 .env，请先 cp .env.ccds.example .env 并确认路径"; exit 1; }
set -a; source .env; set +a

module load anaconda
eval "$(conda shell.bash hook)"

# 只用 conda-forge：Anaconda 默认频道在新版 conda 中需要先接受服务条款，conda-forge 不需要
if ! conda env list | grep -qE '^ecomq\s'; then
  conda create -y -n ecomq --override-channels -c conda-forge python=3.11
fi
conda activate ecomq
python -m pip install -U pip

# --no-cache-dir：家目录配额约 100GB，避免 pip 缓存再存一份几 GB 的 wheel
# PyTorch：wheel 的 CUDA 版本需与计算节点驱动兼容（指南里可用 module load cuda/12.8.0）。
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

pip install --no-cache-dir -e ".[dev]"
pip install --no-cache-dir -r requirements-single.txt

mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$OUTPUT_ROOT" "$HF_HOME" logs
pip freeze > requirements-lock-ccds.txt
echo "环境 ecomq 已创建，依赖版本已写入 requirements-lock-ccds.txt"
du -sh "$CONDA_PREFIX" 2>/dev/null || true
