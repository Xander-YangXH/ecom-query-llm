# 被 scripts/pbs/*.pbs source：进入提交目录、读 .env、把 conda 环境放到 PATH 最前。
# 不用 conda activate：批处理 shell 里没有初始化 conda，直接用环境里的解释器即可（CUDA 库随 torch 的 wheel 安装）。
cd "${PBS_O_WORKDIR:?}"
set -a; source .env; set +a
: "${ENV_PREFIX:?.env 里缺少 ENV_PREFIX}"
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH=src
# 计算节点按没有外网处理：模型和数据都在登录节点提前下载好
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
mkdir -p logs
echo "== job ${PBS_JOBID:-?} on $(hostname) at $(date '+%F %T'), queue ${PBS_QUEUE:-?}"
