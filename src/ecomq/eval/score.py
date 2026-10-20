"""读取 test.jsonl 与 predictions.jsonl，输出指标 JSON（体积很小，可以提交进 git 的 reports/）。"""
import argparse
import json
from pathlib import Path

from ecomq.data.io import read_jsonl
from ecomq.eval.metrics import evaluate, evaluate_by_difficulty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--output", help="写出指标 JSON，例如 reports/qwen7b-lora-r16.json")
    args = ap.parse_args()

    recs = list(read_jsonl(args.test))
    pred_by_id = {p["id"]: p["output"] for p in read_jsonl(args.preds)}
    missing = [r["id"] for r in recs if r["id"] not in pred_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} 条测试样本没有预测，例如 {missing[:3]}")
    outputs = [pred_by_id[r["id"]] for r in recs]

    report = {
        "overall": evaluate([r["label"] for r in recs], outputs),
        "by_difficulty": evaluate_by_difficulty(recs, outputs),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
