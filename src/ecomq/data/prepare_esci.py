"""ESCI（Shopping Queries Dataset）→ relevance 任务的 train/val/test。

抽样策略（理由见 DEVLOG S7）：
- 只用美国英文（product_locale=us）、large_version=1（官方 Task 2 的数据范围）。
- 按 query 分组抽样：先抽 query，再在每个 query 下随机取至多 per_query 个商品。
  这样同样的样本量覆盖更多不同的 query，而且同一 query 只会出现在一个集合里（不泄漏）。
- val 从官方 train 中划出（与 train 的 query 不重叠）；test 从官方 test 中抽取。
- 保留自然的标签分布（不做类别平衡），作为第一组实验的基线设定。
输出：
  <out-dir>/splits/{train,val,test}.jsonl   规范格式（task=relevance）
  <out-dir>/llamafactory/                   Alpaca 格式 + dataset_info.json（prefix=esci_rel）
  <out-dir>/stats.json                      各集合的规模与标签分布
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ecomq.data.io import write_jsonl
from ecomq.data.to_alpaca import export
from ecomq.paths import DATA_ROOT
from ecomq.schema import ESCI_TO_RELEVANCE, validate_record

TITLE_MAX_CHARS = 200


def sample_queries(df, n_queries: int, per_query: int, seed: int, exclude: set | None = None):
    """从 df 中随机抽 n_queries 个 query（排除 exclude），每个 query 随机取至多 per_query 行。"""
    exclude = exclude or set()
    qids = sorted(set(df["query_id"]) - exclude)
    rng = __import__("random").Random(seed)
    chosen = set(rng.sample(qids, min(n_queries, len(qids))))
    sub = df[df["query_id"].isin(chosen)]
    sub = sub.sample(frac=1.0, random_state=seed)  # 打乱后每组取前 per_query 行 = 组内随机抽样
    return sub.groupby("query_id", sort=False).head(per_query), chosen


def to_record(row) -> dict:
    title = str(row["product_title"]).strip()
    if len(title) > TITLE_MAX_CHARS:
        title = title[:TITLE_MAX_CHARS].rsplit(" ", 1)[0]
    return {
        "id": f"esci-{row['example_id']}",
        "task": "relevance",
        "query": str(row["query"]).strip(),
        "title": title,
        "label": {"relevance": ESCI_TO_RELEVANCE[row["esci_label"]]},
        "group": f"q{row['query_id']}",
        "meta": {
            "source": "esci",
            "query_id": int(row["query_id"]),
            "product_id": row["product_id"],
            "esci_label": row["esci_label"],
            "esci_split": row["split"],
        },
    }


def label_stats(records: list[dict]) -> dict:
    counts = Counter(r["label"]["relevance"] for r in records)
    n = len(records)
    return {
        "n": n,
        "queries": len({r["group"] for r in records}),
        "labels": {k: {"count": counts[k], "share": round(counts[k] / n, 4) if n else 0} for k in ESCI_TO_RELEVANCE.values()},
    }


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(DATA_ROOT / "raw" / "esci"))
    ap.add_argument("--out-dir", default=str(DATA_ROOT / "e1_relevance"))
    ap.add_argument("--locale", default="us")
    ap.add_argument("--train-queries", type=int, default=5000)
    ap.add_argument("--val-queries", type=int, default=250)
    ap.add_argument("--test-queries", type=int, default=600)
    ap.add_argument("--per-query", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    ex = pd.read_parquet(raw / "shopping_queries_dataset_examples.parquet")
    ex = ex[(ex["product_locale"] == args.locale) & (ex["large_version"] == 1)]
    prod = pd.read_parquet(
        raw / "shopping_queries_dataset_products.parquet",
        columns=["product_id", "product_locale", "product_title"],
    )
    prod = prod[prod["product_locale"] == args.locale]
    df = ex.merge(prod, on=["product_locale", "product_id"], how="left")
    before = len(df)
    df = df[df["product_title"].notna() & (df["product_title"].str.strip() != "")]
    print(f"locale={args.locale} large_version rows={before}, with title={len(df)}")

    train_pool = df[df["split"] == "train"]
    test_pool = df[df["split"] == "test"]
    val_df, val_q = sample_queries(train_pool, args.val_queries, args.per_query, args.seed)
    train_df, train_q = sample_queries(train_pool, args.train_queries, args.per_query, args.seed + 1, exclude=val_q)
    test_df, _ = sample_queries(test_pool, args.test_queries, args.per_query, args.seed + 2)
    assert not (val_q & train_q), "val/train query overlap"

    stats = {"source": "ESCI us large_version", "per_query": args.per_query, "seed": args.seed,
             "pool": {"train_rows": len(train_pool), "test_rows": len(test_pool),
                      "train_label_share": train_pool["esci_label"].value_counts(normalize=True).round(4).to_dict()}}
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        records = [to_record(r) for r in part.to_dict("records")]
        bad = [r["id"] for r in records if validate_record(r)]
        assert not bad, f"{len(bad)} invalid records in {name}, e.g. {bad[:3]}"
        write_jsonl(out / "splits" / f"{name}.jsonl", records)
        stats[name] = label_stats(records)
        print(name, json.dumps(stats[name]))

    export(out / "splits", out / "llamafactory", prefix="esci_rel")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
