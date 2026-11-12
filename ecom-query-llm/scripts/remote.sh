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
    # 只同步 git 跟踪的文件（新文件需先 git add）：数据、.env、模型本来就不在 git 里，不需要写排除规则。
    # 早期版本用 --exclude 'data/'，会误伤任意层级的 data 目录（包括 src/ecomq/data），见 DEVLOG P7。
    # 不用 --delete：避免误删远端作业生成的 logs/、reports/。
    cd "$ROOT"
    git ls-files -z | rsync -az --itemize-changes --from0 --files-from=- ./ "$HOST:$REMOTE_REPO/"
    # 关键文件抽查：源码包的每个子包都要在远端存在
    ssh "$HOST" "cd $REMOTE_REPO && for p in src/ecomq src/ecomq/data src/ecomq/eval; do test -f \$p/__init__.py || { echo \"MISSING \$p\"; exit 1; }; done; echo sync-check-ok" 2>/dev/null | tail -1 ;;
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
