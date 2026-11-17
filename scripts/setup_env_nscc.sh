#!/usr/bin/env bash
# 在 NSCC ASPIRE2A 登录节点运行一次，只做安装，不跑计算（登录节点禁止运行计算作业）。
#   cp .env.nscc.example .env   # 填好 NSCC_PROJECT
#   bash scripts/setup_env_nscc.sh
# 依赖版本直接取 CCDS 的锁定文件（requirements-lock-ccds.txt），两边的 torch / transformers / LLaMA-Factory
# 版本一致，复现时只差显卡型号。
set -eo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "缺少 .env，请先 cp .env.nscc.example .env 并填好 NSCC_PROJECT"; exit 1; }
set -a; source .env; set +a
: "${ENV_PREFIX:?}"; : "${ECOMQ_ROOT:?}"

mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$OUTPUT_ROOT" "$HF_HOME" logs
# conda 包缓存也放 scratch，避免占用 50GB 的家目录
export CONDA_PKGS_DIRS="$ECOMQ_ROOT/conda_pkgs"

module load miniforge3
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge python=3.11
fi
export PATH="$ENV_PREFIX/bin:$PATH"
python -m pip install -U pip

# 锁定文件里有一行 conda 构建时留下的 "packaging @ file://..." 本地路径，别的机器上不存在，跳过。
LOCK=$(mktemp)
grep -v ' @ file://' requirements-lock-ccds.txt > "$LOCK"
# torch 系列带 +cu128 后缀，只在 PyTorch 的 wheel 源里有；其余包从 PyPI 取。
pip install --no-cache-dir -r "$LOCK" --extra-index-url https://download.pytorch.org/whl/cu128
rm -f "$LOCK"
pip install --no-cache-dir --no-deps -e .

# 多卡全参数微调（N3）用 DeepSpeed。安装时要能找到 nvcc（默认 PATH 里没有），用 .env 里的 CUDA_HOME；
# DS_BUILD_OPS=0 不预编译 CUDA 算子；-c 锁住 torch 等版本，避免被顺带升级。
if [ -n "${CUDA_HOME:-}" ]; then
  CONS=$(mktemp)
  pip freeze | grep -iE '^(torch|torchvision|torchaudio|transformers|accelerate|numpy|pydantic)==' > "$CONS"
  PATH="$CUDA_HOME/bin:$PATH" DS_BUILD_OPS=0 pip install --no-cache-dir --no-build-isolation -c "$CONS" deepspeed==0.20.0
  rm -f "$CONS"
fi

python - <<'PY'
import torch, transformers, peft, bitsandbytes
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| bitsandbytes", bitsandbytes.__version__)
PY
llamafactory-cli version 2>/dev/null | grep -i -m1 version || true
du -sh "$ENV_PREFIX"
