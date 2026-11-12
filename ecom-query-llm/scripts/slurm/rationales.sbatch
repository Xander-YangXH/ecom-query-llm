#!/bin/bash
# E9 推理数据（DEVLOG S21/S22）：用基座模型为 relevance 训练样本生成理由。
#   试点：sbatch --export=ALL,INPUT=<train.jsonl>,NAME=pilot500,N=500,VERIFY_SCORES=1 scripts/slurm/rationales.sbatch
#   全量：sbatch --time=06:00:00 --export=ALL,INPUT=<train.jsonl>,NAME=e2a-train,N=0 scripts/slurm/rationales.sbatch
# 输出：$OUTPUT_ROOT/rationales/$NAME.{jsonl,report.json,review.md}
# VERIFY_SCORES=1：先用重构后的 score_labels 对 2500 query 测试集的前 200 条重新打分（E2a 适配器），
#   与已有的 label_scores/e2a-test.jsonl 比对，确认把打分循环抽成 label_logps 后结果不变。
#   再设 VERIFY_OLD=<重构前的 score_labels.py>：在同样的 200 条上运行旧代码，要求与新代码逐值相同
#   （与已有文件比对时，批次组成不同会带来很小的数值差异，只有同输入对比才能确认逻辑完全一致）。
# HONEST=1：--honest-first（先让模型不看标签自己判断，判对的采用其理由）。
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# L40S 只在 TC2N07、TC2N08 上（GRES 未标显卡型号），排除其余节点
#SBATCH --exclude=TC2N0[1-6]
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --job-name=ecomq-rationale
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${INPUT:?}"; : "${NAME:?}"
set -a; source .env; set +a
module load anaconda
eval "$(conda shell.bash hook)"
conda activate ecomq
module load cuda/12.8.0
MODEL_DIR="$MODEL_ROOT/${MODEL:-Qwen2.5-7B-Instruct}"
OUT="$OUTPUT_ROOT/rationales"; mkdir -p "$OUT"

if [ -n "${VERIFY_SCORES:-}" ]; then
  TMP="$OUT/verify-$SLURM_JOB_ID"; mkdir -p "$TMP"
  head -200 "$DATA_ROOT/esci_test2500/splits/test.jsonl" > "$TMP/test200.jsonl"
  PYTHONPATH=src python -m ecomq.eval.score_labels --model "$MODEL_DIR" --lora "$OUTPUT_ROOT/e2a-1ep-best" \
    --test "$TMP/test200.jsonl" --output "$TMP/new.jsonl" --batch-size 32 --load-4bit
  python - "$TMP/new.jsonl" "$OUTPUT_ROOT/label_scores/e2a-test.jsonl" <<'PY'
import json, sys
new = {r["id"]: r["logp"] for r in map(json.loads, open(sys.argv[1]))}
old = {r["id"]: r["logp"] for r in map(json.loads, open(sys.argv[2])) if r["id"] in new}
diff = max(abs(new[i][c] - old[i][c]) for i in new for c in new[i])
same = sum(max(new[i], key=new[i].get) == max(old[i], key=old[i].get) for i in new)
print(f"verify_scores: n={len(new)} max_abs_diff={diff:.4f} argmax_agree={same}/{len(new)}")
PY
  if [ -n "${VERIFY_OLD:-}" ]; then
    PYTHONPATH=src python "$VERIFY_OLD" --model "$MODEL_DIR" --lora "$OUTPUT_ROOT/e2a-1ep-best" \
      --test "$TMP/test200.jsonl" --output "$TMP/old.jsonl" --batch-size 32 --load-4bit
    python - "$TMP/new.jsonl" "$TMP/old.jsonl" <<'PY'
import json, sys
new = {r["id"]: r["logp"] for r in map(json.loads, open(sys.argv[1]))}
old = {r["id"]: r["logp"] for r in map(json.loads, open(sys.argv[2]))}
diff = max(abs(new[i][c] - old[i][c]) for i in new for c in new[i])
print(f"verify_old_vs_new: n={len(new)} same_ids={set(new) == set(old)} max_abs_diff={diff:.6f}")
PY
  fi
  rm -r "$TMP"
fi

START=$SECONDS
PYTHONPATH=src python -m ecomq.data.make_rationales --model "$MODEL_DIR" --input "$INPUT" \
  --output "$OUT/$NAME.jsonl" --report "$OUT/$NAME.report.json" --review "$OUT/$NAME.review.md" \
  --n "${N:-0}" --batch-size "${BATCH_SIZE:-32}" ${BLIND:+--blind-check} ${HONEST:+--honest-first}
echo "make_rationales: $((SECONDS - START))s"
