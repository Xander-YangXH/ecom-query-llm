#!/usr/bin/env bash
# 本地便捷命令，需要 ~/.ssh/config 里配好主机别名，或直接写 user@host。
#   REMOTE_HOST=<user>@<ccds-host> scripts/remote.sh sync-code          上传代码（rsync，不含数据与 .git）
#   REMOTE_HOST=... scripts/remote.sh pull                                  远端 git pull（远端是 git 仓库时用，如 NSCC）
#   REMOTE_HOST=... REMOTE_DATA_ROOT=... scripts/remote.sh put <本地> <远端相对路径>   上传小体积数据
#   REMOTE_HOST=... scripts/remote.sh get-reports                           把远端 reports/ 拉回本地
# 变量：REMOTE_HOST（必填）、REMOTE_REPO（默认 ~/ecom-query-llm）、REMOTE_DATA_ROOT（put 时必填）。
# CCDS 需要在 NTU 校网或 VPN 下才能连通；尽量少建立连接，不要在头节点跑计算。
set -euo pipefail

HOST="${REMOTE_HOST:?请先 export REMOTE_HOST=ssh 别名或 user@host}"
REMOTE_REPO="${REMOTE_REPO:-~/ecom-query-llm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
  sync-code)
    # 不用 --delete：避免误删远端生成的 logs/、reports/；远端的 .env 也不会被覆盖
    rsync -az --itemize-changes \
      --exclude '.git/' --exclude '.env' --exclude '.venv/' --exclude '__pycache__/' \
      --exclude '.pytest_cache/' --exclude 'build/' --exclude 'data/' --exclude 'models/' \
      --exclude 'outputs/' --exclude 'saves/' --exclude 'logs/' --exclude 'reports/' \
      "$ROOT/" "$HOST:$REMOTE_REPO/" ;;
  pull)
    ssh "$HOST" "cd $REMOTE_REPO && git pull --ff-only" ;;
  put)
    DATA="${REMOTE_DATA_ROOT:?请先 export REMOTE_DATA_ROOT=远端 DATA_ROOT 的绝对路径}"
    rsync -avz --progress "$2" "$HOST:$DATA/$3" ;;
  get-reports)
    rsync -avz "$HOST:$REMOTE_REPO/reports/" "$ROOT/reports/" ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
