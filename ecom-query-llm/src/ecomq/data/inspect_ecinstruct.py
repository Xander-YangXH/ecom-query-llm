"""统计 ECInstruct 的构成，并检查它与本项目 ESCI 测试集的 query 重叠（数据泄漏风险）。

ECInstruct 的 Multiclass_Product_Classification 等任务由 ESCI 派生。如果以后把 ECInstruct
混进训练数据，而它的训练部分包含我们 ESCI 测试集里的 query，评测分数就会虚高。
输出：<out> JSON（任务 × setting × split 的计数、各任务示例、重叠统计）。
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ecomq.data.io import read_jsonl
from ecomq.paths import DATA_ROOT


def _query_of(item: dict) -> str | None:
    inp = item.get("input")
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except json.JSONDecodeError:
            return None
    if isinstance(inp, dict):
        q = inp.get("query")
        return q.strip().lower() if isinstance(q, str) else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DATA_ROOT / "raw" / "ecinstruct" / "dataset.json"))
    ap.add_argument("--esci-test", default=str(DATA_ROOT / "e1_relevance" / "splits" / "test.jsonl"))
    ap.add_argument("--out", default="reports/ecinstruct_stats.json")
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # 兼容 {"train": [...]} 之类的外层结构
        data = next(v for v in data.values() if isinstance(v, list))

    counts = Counter((d.get("task"), d.get("setting"), d.get("split")) for d in data)
    by_task = defaultdict(dict)
    for (task, setting, split), n in sorted(counts.items(), key=lambda x: str(x[0])):
        by_task[task][f"{setting}/{split}"] = n
    examples = {}
    for d in data:
        t = d.get("task")
        if t not in examples:
            examples[t] = {k: (str(d.get(k))[:400] if d.get(k) is not None else None)
                           for k in ("instruction", "input", "options", "output", "setting", "split")}

    test_queries = {r["query"].strip().lower() for r in read_jsonl(args.esci_test)} if Path(args.esci_test).exists() else set()
    overlap = defaultdict(lambda: {"rows": 0, "overlap_rows": 0, "overlap_queries": set()})
    for d in data:
        q = _query_of(d)
        if q is None:
            continue
        key = f"{d.get('task')}/{d.get('split')}"
        overlap[key]["rows"] += 1
        if q in test_queries:
            overlap[key]["overlap_rows"] += 1
            overlap[key]["overlap_queries"].add(q)

    report = {
        "total_rows": len(data),
        "by_task": by_task,
        "examples": examples,
        "esci_test_queries": len(test_queries),
        "overlap_with_esci_test": {
            k: {"rows": v["rows"], "overlap_rows": v["overlap_rows"], "overlap_queries": len(v["overlap_queries"])}
            for k, v in sorted(overlap.items())
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total_rows": report["total_rows"], "overlap_with_esci_test": report["overlap_with_esci_test"]}, indent=2))


if __name__ == "__main__":
    main()
