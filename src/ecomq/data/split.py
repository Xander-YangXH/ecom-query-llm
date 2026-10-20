"""按分组哈希划分 train/val/test，同一分组永远落在同一个集合里，避免相似样本跨集泄漏。

分组键优先级：rec["group"] > rec["title"] > rec["query"]。
合成数据时请给同一商品、同一 query 模板派生出的样本写同一个 group。
"""
import argparse
import hashlib
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl


def assign(rec: dict, ratios: tuple[float, float, float], seed: int) -> str:
    key = rec.get("group") or rec.get("title") or rec["query"]
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    x = int(digest[:12], 16) / 16**12
    train, val, _ = ratios
    if x < train:
        return "train"
    if x < train + val:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.9, 0.05, 0.05])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratios = tuple(args.ratios)
    assert abs(sum(ratios) - 1) < 1e-6, "ratios 之和必须为 1"
    buckets = {"train": [], "val": [], "test": []}
    for rec in read_jsonl(args.input):
        buckets[assign(rec, ratios, args.seed)].append(rec)
    for name, rows in buckets.items():
        n = write_jsonl(Path(args.out_dir) / f"{name}.jsonl", rows)
        print(f"{name}: {n}")


if __name__ == "__main__":
    main()
