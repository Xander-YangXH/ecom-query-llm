#!/usr/bin/env bash
# 本地分析缓存（DEVLOG P16）：把调参集、测试集记录与指定模型的标签打分拉到 .cache/，
# 再用 prior_adjust 重新生成“不加偏置”和“先验调整”两种预测。全部可以从服务器上的打分重建，不进 git。
#   REMOTE_HOST=<user>@<ccds-host> bash scripts/local_cache.sh e2a e5-20k-s43 e6-rich-s42 ...
# 参数是打分文件名的前缀（服务器上的 $OUTPUT_ROOT/label_scores/<name>-{tune,test}.jsonl）。
# 输出：.cache/recs/{tune,test}.jsonl、.cache/scores/<name>-*.jsonl、.cache/preds/<name>-{raw,prior}.jsonl、
#       .cache/reports/<name>.json（与 reports/ 中已提交的先验调整报告应一致，可用来核对）。
set -eo pipefail
: "${REMOTE_HOST:?set REMOTE_HOST=<user>@<host>}"
cd "$(dirname "$0")/.."
C=.cache; mkdir -p $C/recs $C/scores $C/preds $C/reports
PY="${PYTHON:-python3}"
[ -s $C/recs/tune.jsonl ] || scp -q "$REMOTE_HOST:ecomq/data/esci_tune/splits/tune.jsonl" $C/recs/tune.jsonl
[ -s $C/recs/test.jsonl ] || scp -q "$REMOTE_HOST:ecomq/data/esci_test2500/splits/test.jsonl" $C/recs/test.jsonl
for n in "$@"; do
  for part in tune test; do
    [ -s $C/scores/$n-$part.jsonl ] || scp -q "$REMOTE_HOST:ecomq/outputs/label_scores/$n-$part.jsonl" $C/scores/$n-$part.jsonl
  done
  [ -s $C/preds/$n-prior.jsonl ] || PYTHONPATH=src $PY -m ecomq.eval.prior_adjust \
    --tune-recs $C/recs/tune.jsonl --tune-scores $C/scores/$n-tune.jsonl \
    --test-recs $C/recs/test.jsonl --test-scores $C/scores/$n-test.jsonl \
    --report $C/reports/$n.json --preds-out $C/preds/$n-prior.jsonl --raw-preds-out $C/preds/$n-raw.jsonl > /dev/null
  echo "cached $n"
done
