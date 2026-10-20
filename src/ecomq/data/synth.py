"""用强模型从种子商品标题合成 query 标注数据（OpenAI 兼容接口）。

输入种子：每行 {"title": ..., "category": ...}（放在 DATA_ROOT/raw/ 下，不进 git）。
输出为规范格式的原始样本，之后必须经过 validate.py 才能进入训练集。
计算节点若无外网，请在有网的机器上跑本脚本，再把 jsonl 传到远端 DATA_ROOT。
"""
import argparse
import json
import os
import random

from ecomq.data.io import read_jsonl
from ecomq.schema import ATTR_KEYS, INTENTS, RELEVANCE

STYLES = ["short keywords", "conversational", "with typos or abbreviations", "multiple attributes",
          "with a negative constraint", "with a price range"]

PROMPT = """You are generating training data for an e-commerce query understanding model.
Product title: {title} (category: {category}).
Write {n} realistic English search queries a shopper might type when looking for this or similar products.
Vary the style using: {styles}.
For each query output a JSON object with:
- query: the user query
- intent: one of {intents}
- attributes: keys only from [{attr_keys}]; every value must be copied verbatim from the query
- rewrite: a normalized search string
- relevance: how this product relates to the query, one of [{relevance}]
Output a JSON array only, no explanation."""


def build_prompt(seed: dict, n: int = 4) -> str:
    return PROMPT.format(
        title=seed["title"],
        category=seed.get("category", ""),
        n=n,
        styles=", ".join(random.sample(STYLES, k=min(3, len(STYLES)))),
        intents=", ".join(INTENTS),
        attr_keys=", ".join(ATTR_KEYS),
        relevance=", ".join(RELEVANCE),
    )


def to_records(seed: dict, items: list[dict], idx: int) -> list[dict]:
    """把模型返回的条目转成规范格式；每条同时产出带标题与不带标题两种样本。"""
    out = []
    for j, it in enumerate(items):
        base = {
            "task": "full",
            "query": it.get("query", ""),
            "meta": {"source": "synth", "seed_idx": idx},
            "group": seed["title"],
        }
        label = {
            "intent": it.get("intent"),
            "attributes": it.get("attributes") or {},
            "rewrite": it.get("rewrite"),
        }
        out.append({**base, "id": f"s{idx}-{j}-q", "label": {**label, "relevance": None}})
        out.append({**base, "id": f"s{idx}-{j}-qt", "title": seed["title"],
                    "label": {**label, "relevance": it.get("relevance")}})
    return out


def main():
    from openai import OpenAI  # 可选依赖：pip install -e .[synth]

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-per-seed", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = OpenAI(base_url=os.environ["SYNTH_BASE_URL"], api_key=os.environ["SYNTH_API_KEY"])
    model = os.environ["SYNTH_MODEL"]

    with open(args.output, "a", encoding="utf-8") as f:
        for idx, seed in enumerate(read_jsonl(args.seeds)):
            if args.limit and idx >= args.limit:
                break
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_prompt(seed, args.n_per_seed)}],
                temperature=0.8,
            )
            try:
                items = json.loads(resp.choices[0].message.content)
            except json.JSONDecodeError:
                continue
            for rec in to_records(seed, items, idx):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
