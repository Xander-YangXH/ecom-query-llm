"""生成前端演示用的数据（demo/data/en.json、zh.json）。

全部来自模型在留出集上的真实打分，不是现场推理：
- en.json：ESCI 2,500 query 测试集（每个 query 4 个商品），E2a seed 42 对 4 个标签的对数概率，
  以及在调参集上选出的先验调整偏置（reports/e2a-prior 同一份）。
- zh.json：Multi-CPR 电商开发集 1,000 个 query 的 BM25 前 20 名，零样本 7B 与微调 7B（seed 42）的
  打分 log p(relevant) − log p(irrelevant)。
用法（输入由 ecom-query-llm/scripts/local_cache.sh 从服务器拉取到 .cache/）：
  python demo/build_data.py --cache ecom-query-llm/.cache --bias-report ecom-query-llm/.cache/reports/e2a.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

LABELS = ["exact", "substitute", "complement", "irrelevant"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_en(cache: Path, bias_report: Path) -> dict:
    recs = read_jsonl(cache / "recs/test.jsonl")
    lp = {r["id"]: r["logp"] for r in read_jsonl(cache / "scores/e2a-test.jsonl")}
    groups = defaultdict(list)
    order = []
    for r in recs:
        if r["group"] not in groups:
            order.append(r["group"])
        groups[r["group"]].append(r)
    queries = []
    for g in order:
        rows = groups[g]
        queries.append({
            "q": rows[0]["query"],
            "items": [[r["title"], LABELS.index(r["label"]["relevance"]),
                       [lp[r["id"]][c] for c in LABELS]] for r in rows],
        })
    bias = json.loads(bias_report.read_text(encoding="utf-8"))["bias"]
    return {"labels": LABELS, "bias": [bias[c] for c in LABELS], "model": "Qwen2.5-7B-Instruct + QLoRA（E2a seed 42）",
            "queries": queries}


def build_zh(cache: Path) -> dict:
    dev = read_jsonl(cache / "d5/dev_rerank.jsonl")

    def scores(name):
        return {r["id"]: r["logp"]["relevant"] - r["logp"]["irrelevant"] for r in read_jsonl(cache / f"d5/{name}.jsonl")}

    zs, ft = scores("d5-zeroshot-dev"), scores("d5-mcpr-s42-dev")
    groups = defaultdict(list)
    order = []
    for r in dev:
        if r["group"] not in groups:
            order.append(r["group"])
        groups[r["group"]].append(r)
    queries = []
    for g in order:
        rows = sorted(groups[g], key=lambda r: r["meta"]["bm25_rank"])
        queries.append({
            "q": rows[0]["query"],
            "items": [[r["title"], int(r["label"]["relevance"] == "relevant"),
                       zs[r["id"]], ft[r["id"]]] for r in rows],
        })
    return {"model": "Qwen2.5-7B-Instruct：零样本 / QLoRA 微调（seed 42）", "queries": queries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--bias-report", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("en", build_en(Path(a.cache), Path(a.bias_report))), ("zh", build_zh(Path(a.cache)))):
        (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(name, len(data["queries"]), "queries")


if __name__ == "__main__":
    main()
