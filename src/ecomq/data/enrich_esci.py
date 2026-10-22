"""给已有的 relevance 数据补上商品附加字段（品牌、颜色、商品要点），其余内容逐条保持不变（E6，DEVLOG S16）。

为了让对比只差在“输入多了哪些信息”，样本集合、编号、标签都与原数据完全相同；脚本会逐条断言。
字段来自官方商品表 shopping_queries_dataset_products.parquet，按 (product_locale, product_id) 对齐。
缺失或空的字段直接不写（不写 unknown），渲染时也就不出现。
用法：
  python -m ecomq.data.enrich_esci --out-dir $DATA_ROOT/e6_rich \
      --inputs train=<train.jsonl> val=<val.jsonl> tune=<tune.jsonl> test=<test.jsonl>
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.data.to_alpaca import export
from ecomq.paths import DATA_ROOT

BULLETS_MAX_CHARS = 300
SHORT_MAX_CHARS = 60


def _clean(text) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(" |,;") + " ..."


def clean_bullets(raw, max_chars: int = BULLETS_MAX_CHARS) -> str:
    """官方字段里各要点以换行分隔：去空行、压缩空白，用 ' | ' 连接后截断。"""
    if raw is None:
        return ""
    parts = [_clean(p) for p in str(raw).split("\n")]
    joined = " | ".join(p for p in parts if p)
    return _truncate(joined, max_chars) if joined else ""


def product_fields(brand, color, bullets) -> dict:
    out = {}
    if (b := _truncate(_clean(brand), SHORT_MAX_CHARS)):
        out["brand"] = b
    if (c := _truncate(_clean(color), SHORT_MAX_CHARS)):
        out["color"] = c
    if (bl := clean_bullets(bullets)):
        out["bullets"] = bl
    return out


def enrich(records: list[dict], lookup: dict) -> list[dict]:
    out = []
    for r in records:
        fields = lookup.get(r["meta"]["product_id"], {})
        rec = {k: v for k, v in r.items() if k != "product"}
        if fields:
            rec["product"] = fields
        out.append(rec)
    assert [r["id"] for r in out] == [r["id"] for r in records]
    assert [r["label"] for r in out] == [r["label"] for r in records]
    return out


def coverage(records: list[dict]) -> dict:
    n = len(records)
    c = Counter(k for r in records for k in r.get("product", {}))
    return {"n": n, **{k: round(c[k] / n, 4) for k in ("brand", "color", "bullets")},
            "any": round(sum(1 for r in records if r.get("product")) / n, 4)}


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(DATA_ROOT / "raw" / "esci"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="name=path")
    ap.add_argument("--locale", default="us")
    args = ap.parse_args()

    inputs = dict(s.split("=", 1) for s in args.inputs)
    data = {name: list(read_jsonl(path)) for name, path in inputs.items()}
    needed = {r["meta"]["product_id"] for recs in data.values() for r in recs}

    prod = pd.read_parquet(Path(args.raw_dir) / "shopping_queries_dataset_products.parquet",
                           columns=["product_id", "product_locale", "product_brand", "product_color", "product_bullet_point"])
    prod = prod[(prod["product_locale"] == args.locale) & prod["product_id"].isin(needed)]
    lookup = {row["product_id"]: product_fields(row["product_brand"], row["product_color"], row["product_bullet_point"])
              for row in prod.to_dict("records")}

    out = Path(args.out_dir)
    stats = {"source": inputs, "bullets_max_chars": BULLETS_MAX_CHARS}
    for name, recs in data.items():
        rich = enrich(recs, lookup)
        write_jsonl(out / "splits" / f"{name}.jsonl", rich)
        stats[name] = coverage(rich)
        print(name, json.dumps(stats[name]))
    if {"train", "val"} <= set(data):
        export(out / "splits", out / "llamafactory", prefix="esci_rel")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
