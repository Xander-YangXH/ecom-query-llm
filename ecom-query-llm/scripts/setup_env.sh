#!/usr/bin/env bash
# 在远端登录节点运行一次：创建 venv、安装依赖。
# 先按 NSCC 文档 module load 合适的 python / cuda，再执行本脚本。
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] || { echo "缺少 .env，请先 cp .env.nscc.example .env 并填写路径"; exit 1; }
set -a; source .env; set +a

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 1) 先装与集群 CUDA 匹配的 PyTorch（版本和 index-url 见 pytorch.org，不要盲目照抄）
# pip install torch --index-url https://download.pytorch.org/whl/cuXXX

# 2) 项目本体 + 训练依赖
pip install -e ".[dev,synth]"
pip install -r requirements-train.txt

mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$OUTPUT_ROOT" "$HF_HOME" logs
echo "环境就绪。训练前请在登录节点把模型下载到 $MODEL_ROOT："
echo "  HF_HOME=$HF_HOME huggingface-cli download <model> --local-dir $MODEL_ROOT/<name>"
