#!/usr/bin/env bash
# 下载基座模型到 $MODEL_ROOT/<目录名>。CCDS 头节点可以联网，下载不属于计算，可在头节点运行。
#   bash scripts/download_model.sh Qwen/Qwen2.5-7B-Instruct
# 第二个参数可指定本地目录名，默认取仓库名最后一段。中断后重跑会跳过已下载的文件。
set -eo pipefail
cd "$(dirname "$0")/.."

REPO_ID="${1:?用法：download_model.sh <hf_repo_id> [本地目录名]}"
NAME="${2:-${REPO_ID##*/}}"

set -a; source .env; set +a
if command -v module >/dev/null 2>&1; then
  module load anaconda
  eval "$(conda shell.bash hook)"
  conda activate ecomq
fi

DEST="$MODEL_ROOT/$NAME"
mkdir -p "$DEST"
# 下载整个仓库。不要写成 --include "*.json" "*.safetensors" ...：
# 新版 hf CLI 只把第一个模式当 --include，其余会被当成文件名，结果漏下 config.json 和分词器文件。
hf download "$REPO_ID" --local-dir "$DEST"

du -sh "$DEST"
ls "$DEST"
