"""校验 + 精确去重。合格样本写入 --output，被拒样本连同原因写入 --rejects。"""
import argparse
import re
from collections import Counter

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.schema import validate_record


def normalize(text: str | None) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def dedup_key(rec: dict) -> tuple[str, str]:
    return normalize(rec["query"]), normalize(rec.get("title"))


def run(rows):
    kept, rejects, seen = [], [], set()
    for rec in rows:
        errors = validate_record(rec)
        if errors:
            rejects.append({**rec, "_errors": errors})
            continue
        key = dedup_key(rec)
        if key in seen:
            rejects.append({**rec, "_errors": ["重复样本"]})
            continue
        seen.add(key)
        kept.append(rec)
    return kept, rejects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rejects", help="默认写到 <output>.rejects.jsonl")
    args = ap.parse_args()

    kept, rejects = run(read_jsonl(args.input))
    write_jsonl(args.output, kept)
    write_jsonl(args.rejects or args.output + ".rejects.jsonl", rejects)

    reasons = Counter(e.split(":")[0] for r in rejects for e in r["_errors"])
    print(f"保留 {len(kept)} 条，拒绝 {len(rejects)} 条")
    for reason, n in reasons.most_common():
        print(f"  {n:>6}  {reason}")


if __name__ == "__main__":
    main()
