#!/usr/bin/env bash
# E2 的完整调度：先生成 E2b 的重采样数据并核对 val/test 与 E1 完全一致，再依次跑 E2a 和 E2b。
#   setsid nohup bash scripts/slurm/run_e2.sh >> logs/chain_e2.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml

# E2b 数据：与 E1 同样的种子与抽样参数，只多一个 --rebalance
J=$(sbatch --parsable --export=ALL,SKIP_INSPECT=1,PREP_ARGS="--out-dir $DATA_ROOT/e2_rebal --rebalance complement=0.10,irrelevant=0.20" scripts/slurm/prepare_data.sbatch)
while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 30; done
echo "$(date +%T) [e2b-data] job=$J state=$(sacct -j "$J" -X -n -o State | head -1 | tr -d ' ')"
for part in val test; do
  cmp -s "$DATA_ROOT/e1_relevance/splits/$part.jsonl" "$DATA_ROOT/e2_rebal/splits/$part.jsonl" \
    || { echo "$(date +%T) [e2b-data] $part differs from E1, stop"; exit 1; }
done
echo "$(date +%T) [e2b-data] val/test identical to E1"

RUN=e2a-1ep-best CONFIG=$CFG DATASET_DIR=$DATA_ROOT/e1_relevance/llamafactory bash scripts/slurm/chain.sh
RUN=e2b-rebal-1ep-best CONFIG=$CFG DATASET_DIR=$DATA_ROOT/e2_rebal/llamafactory bash scripts/slurm/chain.sh
echo "$(date +%T) e2 all done"
