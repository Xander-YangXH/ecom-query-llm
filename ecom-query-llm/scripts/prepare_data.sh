#!/usr/bin/env bash
# 在远端（或本地）把 raw 合成数据处理成训练用数据。所有产物都落在 $DATA_ROOT，不进 git。
#   $DATA_ROOT/raw/synth.jsonl      合成原始数据（来自 ecomq.data.synth）
#   $DATA_ROOT/clean/valid.jsonl    校验 + 去重后的数据
#   $DATA_ROOT/splits/*.jsonl       train / val / test
#   $DATA_ROOT/llamafactory/        Alpaca 格式 + dataset_info.json
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

RAW="${1:-$DATA_ROOT/raw/synth.jsonl}"
python -m ecomq.data.validate --input "$RAW" --output "$DATA_ROOT/clean/valid.jsonl"
python -m ecomq.data.split --input "$DATA_ROOT/clean/valid.jsonl" --out-dir "$DATA_ROOT/splits"
python -m ecomq.data.to_alpaca --split-dir "$DATA_ROOT/splits" --out-dir "$DATA_ROOT/llamafactory"
