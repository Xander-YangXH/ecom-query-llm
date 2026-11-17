#!/usr/bin/env bash
# 在登录节点提交一个 PBS 作业并打印作业号。项目号取自 .env 的 NSCC_PROJECT，不写进作业脚本。
#   scripts/pbs/submit.sh <作业名> [变量=值 ...]
#   AFTER=<作业号[:作业号...]> scripts/pbs/submit.sh ...   # 等这些作业成功结束后才开始（-W depend=afterok）
#   QSUB_ARGS="-l walltime=06:00:00" scripts/pbs/submit.sh train ...   # 追加 qsub 参数
# 值里不能有逗号；含空格或逗号的变量（如 TRAIN_OVERRIDES）先 export，再只写变量名：
#   TRAIN_OVERRIDES="seed=43" scripts/pbs/submit.sh train TRAIN_OVERRIDES CONFIG=... RUN=...
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${NSCC_PROJECT:?.env 里缺少 NSCC_PROJECT}"
JOB="${1:?用法：submit.sh <gpu_check|prepare_data|train|label_scores> [变量=值 ...]}"; shift
[ -f "scripts/pbs/$JOB.pbs" ] || { echo "没有 scripts/pbs/$JOB.pbs" >&2; exit 2; }
mkdir -p logs
ARGS=(-P "$NSCC_PROJECT")
[ -n "${AFTER:-}" ] && ARGS+=(-W "depend=afterok:$AFTER")
if [ "$#" -gt 0 ]; then VARS=$(IFS=,; echo "$*"); ARGS+=(-v "$VARS"); fi
# shellcheck disable=SC2086
qsub "${ARGS[@]}" ${QSUB_ARGS:-} "scripts/pbs/$JOB.pbs"
