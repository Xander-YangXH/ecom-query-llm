#!/usr/bin/env bash
# 按顺序提交 E1 的训练与评测作业，前一个结束后再提交下一个。
# 原因：CCDS 的 QoS 限制每人最多同时“提交” 2 个作业（含排队中的，报错 QOSMaxSubmitJobPerUserLimit），
# 不能一次把整条依赖链都交上去。本脚本在头节点运行，只调用 squeue/sacct/sbatch，不做计算。
#   nohup env AFTER=<作业号> bash scripts/slurm/chain_e1.sh >> logs/chain_e1.log 2>&1 &
# 用 nohup 后台运行并直接写日志文件：本机 SSH 断开（例如电脑休眠）也不会中断调度（见 DEVLOG P11）。
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

T="$DATA_ROOT/e1_relevance/splits/test.jsonl"
MODEL=Qwen2.5-7B-Instruct
RUN=e1-qlora-r16
EVAL_ENV="MODEL=$MODEL,RUN=$RUN,LORA=1,TEST=$T,BATCH_SIZE=32,MAX_NEW_TOKENS=24"

wait_job() {  # 等作业离开队列，打印最终状态
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}

if [ -n "${AFTER:-}" ]; then
  echo "$(date +%T) wait $AFTER -> $(wait_job "$AFTER")"
fi

# 训练：单作业上限 6 小时，超时（TIMEOUT）则用同一 RUN 续训，最多再续 2 次
# TRAIN_JOB=<作业号>：接管一个已经提交的训练作业（调度脚本中途重启时用），不再重复提交
for attempt in 1 2 3; do
  if [ "$attempt" = 1 ] && [ -n "${TRAIN_JOB:-}" ]; then
    J=$TRAIN_JOB
  else
    J=$(sbatch --parsable --export=ALL,CONFIG=configs/sft/e1_relevance_qlora.yaml,MODEL=$MODEL,RUN=$RUN,DATASET_DIR=$DATA_ROOT/e1_relevance/llamafactory scripts/slurm/train.sbatch)
  fi
  S=$(wait_job "$J")
  echo "$(date +%T) train attempt=$attempt job=$J state=$S"
  [ "$S" = "TIMEOUT" ] || break
done
[ "$S" = "COMPLETED" ] || { echo "train failed, stop chain"; exit 1; }

J=$(sbatch --parsable --export=ALL,$EVAL_ENV,LOAD_4BIT=1 scripts/slurm/eval.sbatch)
echo "$(date +%T) eval-4bit job=$J state=$(wait_job "$J")"
J=$(sbatch --parsable --export=ALL,$EVAL_ENV scripts/slurm/eval.sbatch)
echo "$(date +%T) eval-bf16 job=$J state=$(wait_job "$J")"
echo "$(date +%T) chain done"
