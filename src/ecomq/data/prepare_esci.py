"""ESCI（Shopping Queries Dataset）→ relevance 任务的 train/val/test。

抽样策略（理由见 DEVLOG S7）：
- 只用美国英文（product_locale=us）、large_version=1（官方 Task 2 的数据范围）。
- 按 query 分组抽样：先抽 query，再在每个 query 下随机取至多 per_query 个商品。
  这样同样的样本量覆盖更多不同的 query，而且同一 query 只会出现在一个集合里（不泄漏）。
- val 从官方 train 中划出（与 train 的 query 不重叠）；test 从官方 test 中抽取。
- 默认保留自然的标签分布（E1 的基线设定）。
- --rebalance complement=0.10,irrelevant=0.20（E2b）：只调整 train。总量不变，用训练池里新的真实样本
  把指定类别补到目标比例，其余类别按原比例随机减少；val/test 保持自然分布且与不加该参数时完全相同。
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


RELEVANCE_TO_ESCI = {v: k for k, v in ESCI_TO_RELEVANCE.items()}


def parse_targets(spec: str) -> dict[str, float]:
    """'complement=0.10,irrelevant=0.20' -> {'C': 0.10, 'I': 0.20}"""
    out = {}
    for part in filter(None, spec.split(",")):
        name, share = part.split("=")
        out[RELEVANCE_TO_ESCI[name.strip()]] = float(share)
    assert 0 < sum(out.values()) < 1, "target shares must sum to (0, 1)"
    return out


def rebalance(train_df, pool, targets: dict[str, float], seed: int):
    """把 train_df 中 targets 指定的类别补到目标比例（从 pool 中抽未使用过的真实样本），
    再从其余类别里均匀随机删除同样多的行，使总量不变、其余类别之间的相对比例不变。"""
    import pandas as pd

    n = len(train_df)
    counts = train_df["esci_label"].value_counts().to_dict()
    used = set(train_df["example_id"])
    extra = []
    for lab, share in targets.items():
        need = round(share * n) - counts.get(lab, 0)
        if need <= 0:
            continue
        cand = pool[(pool["esci_label"] == lab) & ~pool["example_id"].isin(used)]
        assert len(cand) >= need, f"pool has only {len(cand)} rows of {lab}, need {need}"
        extra.append(cand.sample(n=need, random_state=seed))
    add = pd.concat(extra) if extra else train_df.iloc[0:0]
    keep_target = train_df[train_df["esci_label"].isin(targets)]
    others = train_df[~train_df["esci_label"].isin(targets)]
    others_kept = others.sample(n=len(others) - len(add), random_state=seed)
    return pd.concat([keep_target, others_kept, add]).sample(frac=1.0, random_state=seed)


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
    ap.add_argument("--rebalance", default="", help="例如 complement=0.10,irrelevant=0.20；只作用于 train")
    ap.add_argument("--tune-queries", type=int, default=0,
                    help=">0 时额外从官方 train 池抽一个调参集 tune（用于推理时先验调整等，见 DEVLOG S12）")
    ap.add_argument("--tune-exclude", nargs="*", default=[],
                    help="这些 jsonl 中出现过的 query 不进入 tune（例如各实验的训练集）")
    ap.add_argument("--only-tune", action="store_true", help="只写出 tune")
    ap.add_argument("--only-test", action="store_true",
                    help="只写出 test（例如扩大测试集时），train/val 照常抽样用于不重叠校验但不写出")
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
    test_df, test_q = sample_queries(test_pool, args.test_queries, args.per_query, args.seed + 2)
    # 官方 train/test 之间有极少数共享的 query（美国英文 large_version 中只有 1 个，见 DEVLOG P13）。
    # 抽样之后再剔除，而不是先从池子里剔除：这样不含共享 query 的历史测试集（如 E1 的 600 个）重新生成时保持不变。
    shared = set(test_pool["query_id"]) & set(train_pool["query_id"])
    dropped = sorted(test_q & shared)
    test_df, test_q = test_df[~test_df["query_id"].isin(shared)], test_q - shared
    print(f"official train/test shared queries={len(shared)}, dropped from sampled test={len(dropped)}")
    assert not (val_q & train_q), "val/train query overlap"
    assert not (test_q & (train_q | val_q | set(train_pool["query_id"]))), "test query overlaps train/val"
    if args.rebalance:
        targets = parse_targets(args.rebalance)
        train_df = rebalance(train_df, train_pool[~train_pool["query_id"].isin(val_q)], targets, args.seed + 3)

    stats = {"source": "ESCI us large_version", "per_query": args.per_query, "seed": args.seed,
             "rebalance": args.rebalance or None,
             "official_shared_queries": len(shared), "test_dropped_shared_queries": len(dropped),
             "pool": {"train_rows": len(train_pool), "test_rows": len(test_pool),
                      "train_label_share": train_pool["esci_label"].value_counts(normalize=True).round(4).to_dict()}}
    parts = (("test", test_df),) if args.only_test else (("train", train_df), ("val", val_df), ("test", test_df))
    if args.tune_queries:
        from ecomq.data.io import read_jsonl
        excluded = set(train_q) | set(val_q)
        for path in args.tune_exclude:
            excluded |= {r["meta"]["query_id"] for r in read_jsonl(path)}
        tune_df, tune_q = sample_queries(train_pool, args.tune_queries, args.per_query, args.seed + 4, exclude=excluded)
        assert not (tune_q & excluded) and not (tune_q & test_q), "tune overlaps train/val/test"
        stats["tune_excluded_queries"] = len(excluded)
        parts = (("tune", tune_df),) if args.only_tune else parts + (("tune", tune_df),)
    for name, part in parts:
        records = [to_record(r) for r in part.to_dict("records")]
        bad = [r["id"] for r in records if validate_record(r)]
        assert not bad, f"{len(bad)} invalid records in {name}, e.g. {bad[:3]}"
        write_jsonl(out / "splits" / f"{name}.jsonl", records)
        stats[name] = label_stats(records)
        print(name, json.dumps(stats[name]))

    if not (args.only_test or args.only_tune):
        export(out / "splits", out / "llamafactory", prefix="esci_rel")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
