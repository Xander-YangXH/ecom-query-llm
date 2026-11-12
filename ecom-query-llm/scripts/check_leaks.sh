#!/usr/bin/env bash
# 提交前检查：仓库中不能出现服务器地址、用户名、机器相关的绝对路径（DEVLOG P9、P14）。
# 用法：bash scripts/check_leaks.sh   （有匹配时非零退出并列出位置）
set -uo pipefail
cd "$(dirname "$0")/.."
PATTERN='/home/[a-z]|/Users/[A-Za-z]|/private/tmp|Users-[a-z]+-|10\.[0-9]+\.[0-9]+\.[0-9]+'
if git ls-files -z --cached --others --exclude-standard | xargs -0 grep -nE "$PATTERN" -- 2>/dev/null \
    | grep -vE '^scripts/check_leaks\.sh:|^src/ecomq/paths\.py:|^docs/DEVLOG\.md:|^tests/test_schema\.py:.*redact' \
    | grep -vE 'feedstock_root|==10\.[0-9]|/home/project/<'; then
  # 白名单：本脚本与 redact 的实现和测试；conda-forge 打包路径；形如 ==10.x.y 的版本号；
  # NSCC 文档里的项目目录占位写法 /home/project/<项目号>（不含真实用户名）
  echo "check_leaks: found machine-specific paths or addresses above" >&2
  exit 1
fi
echo "check_leaks: ok"
