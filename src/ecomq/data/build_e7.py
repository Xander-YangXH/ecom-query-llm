"""E7 数据构造（DEVLOG S19）：追加 complement/irrelevant 真实样本 + 有争议 E/S 样本的软标签。

子命令：
  auditor-data  从一个训练集 jsonl 中切出一段（与 E2a 训练集没有共同 query），作为“审计模型”的训练数据。
                审计模型没见过 E2a 的训练样本，用它判断这些样本是否有争议，才不会因为“记住了标注”而失真。
  e7-data       以 E2a 训练集为基础：
                1) 软标签：审计模型以概率 ≥ threshold 判成相反 E/S 标签的样本，再复制一份并换成相反标签。
                   同一输入的目标变成 E、S 各一半，交叉熵上等价于 50/50 的软标签（代价：这些样本权重变为 2 倍）。
                2) 追加 C/I：从官方 train 池中抽取未使用过的 complement / irrelevant 真实样本追加进去，
                   排除验证集、调参集、测试集的全部 query。
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.data.prepare_esci import to_record
from ecomq.data.to_alpaca import export
from ecomq.paths import DATA_ROOT
from ecomq.schema import ESCI_TO_RELEVANCE

FLIP = {"exact": "substitute", "substitute": "exact"}
TO_ESCI = {v: k for k, v in ESCI_TO_RELEVANCE.items()}


def softmax_probs(logp: dict[str, float]) -> dict[str, float]:
    m = max(logp.values())
    z = {k: math.exp(v - m) for k, v in logp.items()}
    s = sum(z.values())
    return {k: v / s for k, v in z.items()}


def soft_duplicates(records: list[dict], scores: dict[str, dict], threshold: float) -> list[dict]:
    """对审计模型以概率 ≥ threshold 判成相反 E/S 标签的样本，生成一份换成相反标签的副本。"""
    out = []
    for r in records:
        gold = r["label"]["relevance"]
        if gold not in FLIP or r["id"] not in scores:
            continue
        p = softmax_probs(scores[r["id"]])
        if p[FLIP[gold]] >= threshold:
            dup = json.loads(json.dumps(r))
            dup["id"] = r["id"] + "-soft"
            dup["label"] = {"relevance": FLIP[gold]}
            dup.setdefault("meta", {})["soft_from"] = gold
            dup["meta"]["auditor_p_flip"] = round(p[FLIP[gold]], 4)
            out.append(dup)
    return out


def extra_class_rows(train_pool, counts: dict[str, int], exclude_queries: set, exclude_examples: set, seed: int) -> list[dict]:
    """从 train 池中按类别抽取未使用过的真实样本（ESCI 字母标签 -> 条数）。"""
    out = []
    for lab, n in counts.items():
        cand = train_pool[(train_pool["esci_label"] == lab)
                          & ~train_pool["query_id"].isin(exclude_queries)
                          & ~train_pool["example_id"].isin(exclude_examples)]
        assert len(cand) >= n, f"pool has only {len(cand)} rows of {lab}"
        out += [to_record(r) for r in cand.sample(n=n, random_state=seed).to_dict("records")]
    return out


def load_train_pool(raw_dir: Path, locale: str = "us"):
    import pandas as pd

    ex = pd.read_parquet(raw_dir / "shopping_queries_dataset_examples.parquet")
    ex = ex[(ex["product_locale"] == locale) & (ex["large_version"] == 1)]
    prod = pd.read_parquet(raw_dir / "shopping_queries_dataset_products.parquet",
                           columns=["product_id", "product_locale", "product_title"])
    prod = prod[prod["product_locale"] == locale]
    df = ex.merge(prod, on=["product_locale", "product_id"], how="left")
    df = df[df["product_title"].notna() & (df["product_title"].str.strip() != "")]
    return df[df["split"] == "train"], set(df[df["split"] == "test"]["query_id"])


def qids(path: str) -> set:
    return {r["meta"]["query_id"] for r in read_jsonl(path)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("auditor-data")
    a.add_argument("--source", required=True)
    a.add_argument("--start", type=int, required=True)
    a.add_argument("--end", type=int, required=True)
    a.add_argument("--val", required=True)
    a.add_argument("--disjoint-from", required=True, help="审计数据不能与这个训练集共享 query（E2a 训练集）")
    a.add_argument("--out-dir", required=True)
    e = sub.add_parser("e7-data")
    e.add_argument("--base", required=True, help="E2a 训练集 jsonl")
    e.add_argument("--val", required=True)
    e.add_argument("--scores", required=True, help="审计模型在 base 上的标签对数概率（score_labels 输出）")
    e.add_argument("--threshold", type=float, default=0.6)
    e.add_argument("--add", default="complement=3000,irrelevant=3000")
    e.add_argument("--exclude-queries-from", nargs="*", default=[])
    e.add_argument("--raw-dir", default=str(DATA_ROOT / "raw" / "esci"))
    e.add_argument("--seed", type=int, default=47)
    e.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)

    if args.cmd == "auditor-data":
        rows = list(read_jsonl(args.source))[args.start:args.end]
        overlap = {r["meta"]["query_id"] for r in rows} & qids(args.disjoint_from)
        assert not overlap, f"auditor data shares {len(overlap)} queries with {args.disjoint_from}"
        write_jsonl(out / "splits" / "train.jsonl", rows)
        write_jsonl(out / "splits" / "val.jsonl", read_jsonl(args.val))
        export(out / "splits", out / "llamafactory", prefix="esci_rel")
        stats = {"n": len(rows), "queries": len({r["meta"]["query_id"] for r in rows}),
                 "labels": dict(Counter(r["label"]["relevance"] for r in rows))}
        (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats))
        return

    base = list(read_jsonl(args.base))
    scores = {r["id"]: r["logp"] for r in read_jsonl(args.scores)}
    assert set(scores) >= {r["id"] for r in base}, "auditor scores missing some base rows"
    dups = soft_duplicates(base, scores, args.threshold)

    train_pool, official_test_q = load_train_pool(Path(args.raw_dir))
    exclude_q = qids(args.val) | official_test_q
    for path in args.exclude_queries_from:
        exclude_q |= qids(path)
    used_examples = {int(r["id"].split("-")[1]) for r in base}  # 规范记录的 id 为 "esci-<example_id>"
    counts = {TO_ESCI[name]: int(n) for name, n in (x.split("=") for x in args.add.split(","))}
    extra = extra_class_rows(train_pool, counts, exclude_q, used_examples, args.seed)
    assert not ({r["meta"]["query_id"] for r in extra} & exclude_q)

    train = base + dups + extra
    random.Random(args.seed).shuffle(train)
    write_jsonl(out / "splits" / "train.jsonl", train)
    write_jsonl(out / "splits" / "val.jsonl", read_jsonl(args.val))
    export(out / "splits", out / "llamafactory", prefix="esci_rel")
    gold_es = sum(r["label"]["relevance"] in FLIP for r in base)
    stats = {
        "base": len(base), "soft_duplicates": len(dups), "extra": len(extra), "total": len(train),
        "soft_share_of_base_es": round(len(dups) / gold_es, 4), "threshold": args.threshold,
        "soft_by_original_label": dict(Counter(d["meta"]["soft_from"] for d in dups)),
        "labels": {k: {"count": v, "share": round(v / len(train), 4)}
                   for k, v in Counter(r["label"]["relevance"] for r in train).items()},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
