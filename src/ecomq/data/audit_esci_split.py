"""审计 ESCI 官方 train/test 划分，以及本项目 train/val/test 之间的 query 重叠（数据泄漏）。

两个角度：
  query_id     官方给每个 query 的编号
  query 文本   小写、去首尾空格后的字符串（同一文本可能有不同编号）
用法：
  python -m ecomq.data.audit_esci_split --examples <examples.parquet> \
      --splits train=<train.jsonl> val=<val.jsonl> test=<test.jsonl> [--output reports/esci_split_audit.json]
"""
import argparse
import json
from pathlib import Path

from ecomq.data.io import read_jsonl


def norm(q: str) -> str:
    return " ".join(str(q).lower().split())


def official_overlap(ex, locale: str = "us") -> dict:
    out = {}
    for version in ("small_version", "large_version"):
        df = ex[(ex["product_locale"] == locale) & (ex[version] == 1)]
        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        shared_ids = set(tr["query_id"]) & set(te["query_id"])
        shared_txt = set(tr["query"].map(norm)) & set(te["query"].map(norm))
        out[version] = {
            "train_queries": int(tr["query_id"].nunique()),
            "test_queries": int(te["query_id"].nunique()),
            "shared_query_ids": len(shared_ids),
            "test_rows_with_shared_id": int(te["query_id"].isin(shared_ids).sum()),
            "shared_query_texts": len(shared_txt),
            "test_rows_with_shared_text": int(te["query"].map(norm).isin(shared_txt).sum()),
            "examples_shared_text": sorted(shared_txt)[:10],
        }
    # 同一个 query_id 是否在 train/test 中对应同一个商品（同一条判断出现在两边）
    df = ex[(ex["product_locale"] == locale) & (ex["large_version"] == 1)]
    pairs = df.groupby(["query_id", "product_id"])["split"].nunique()
    out["large_version"]["query_product_pairs_in_both_splits"] = int((pairs > 1).sum())
    return out


def project_overlap(splits: dict[str, str]) -> dict:
    ids, texts = {}, {}
    for name, path in splits.items():
        recs = list(read_jsonl(path))
        ids[name] = {r["meta"]["query_id"] for r in recs}
        texts[name] = {norm(r["query"]) for r in recs}
    out = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a} & {b}"] = {
                "shared_query_ids": len(ids[a] & ids[b]),
                "shared_query_texts": len(texts[a] & texts[b]),
                "sizes": [len(ids[a]), len(ids[b])],
            }
    return out


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--splits", nargs="*", default=[], help="name=path，规范格式 jsonl")
    ap.add_argument("--output")
    args = ap.parse_args()

    ex = pd.read_parquet(args.examples, columns=["query", "query_id", "product_id", "product_locale",
                                                  "small_version", "large_version", "split"])
    report = {"official": official_overlap(ex)}
    if args.splits:
        report["project"] = project_overlap(dict(s.split("=", 1) for s in args.splits))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
