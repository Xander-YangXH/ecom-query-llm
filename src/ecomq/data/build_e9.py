"""E9a 的训练数据（DEVLOG S23）：由 make_rationales 的全量输出构造 relevance_cot 的 train/val。

规则（S22 之后的决定）：
- 丢弃：重试后仍有质量标记的样本（含标签词、超长、为空）。
- 保留并标记：只看理由时判出的标签与标注不同（理由与标签矛盾），写入 meta.rationale.inconsistent，
  评测时可以按“训练中是否见过矛盾样本”分组分析；不丢弃，否则会损失约三分之一的 complement。
- 可选 --tokenizer：用对话模板计算每条样本（提示 + 回复）的 token 数，超过 --cutoff-len 的条数写进统计，
  超过时报错，避免训练时被悄悄截断。
输出：<out-dir>/splits/{train,val}.jsonl、<out-dir>/llamafactory/（prefix=esci_rel）、<out-dir>/stats.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.data.to_alpaca import export
from ecomq.schema import RELEVANCE, build_user_prompt, format_output, validate_record


def select(rows: list[dict]) -> tuple[list[dict], dict]:
    """返回保留的记录（加上 inconsistent 标记）与统计。"""
    kept, dropped = [], Counter()
    for r in rows:
        meta = r["meta"]["rationale"]
        if meta["flags"]:
            dropped[r["label"]["relevance"]] += 1
            continue
        blind = meta.get("blind", {}).get("with_reason")
        meta["inconsistent"] = blind is not None and blind != r["label"]["relevance"]
        assert not validate_record(r), r["id"]
        kept.append(r)
    count = lambda pred: {c: sum(pred(r) and r["label"]["relevance"] == c for r in kept) for c in RELEVANCE}  # noqa: E731
    stats = {
        "input": len(rows), "kept": len(kept), "dropped_flagged": dict(dropped),
        "kept_by_label": count(lambda r: True),
        "inconsistent_by_label": count(lambda r: r["meta"]["rationale"]["inconsistent"]),
        "self_by_label": count(lambda r: r["meta"]["rationale"].get("source") == "self"),
    }
    return kept, stats


def token_lengths(rows: list[dict], tokenizer_path: str) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    out = []
    for r in rows:
        msgs = [{"role": "user", "content": build_user_prompt(r["query"], r.get("title"), r["task"], r.get("product"))},
                {"role": "assistant", "content": format_output(r["label"], r["task"])}]
        out.append(len(tok.apply_chat_template(msgs, tokenize=True)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="make_rationales 对训练集的输出")
    ap.add_argument("--val", required=True, help="make_rationales 对验证集的输出")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", help="模型目录；给出时检查 token 长度")
    ap.add_argument("--cutoff-len", type=int, default=448)
    args = ap.parse_args()
    out = Path(args.out_dir)

    stats = {}
    for name, path in (("train", args.train), ("val", args.val)):
        kept, st = select(list(read_jsonl(path)))
        if args.tokenizer:
            lens = sorted(token_lengths(kept, args.tokenizer))
            st["tokens"] = {"max": lens[-1], "p99": lens[int(0.99 * (len(lens) - 1))],
                            "over_cutoff": sum(n > args.cutoff_len for n in lens), "cutoff_len": args.cutoff_len}
            assert st["tokens"]["over_cutoff"] == 0, f"{name}: {st['tokens']['over_cutoff']} rows exceed cutoff_len"
        write_jsonl(out / "splits" / f"{name}.jsonl", kept)
        stats[name] = st
        print(name, json.dumps(st))
    export(out / "splits", out / "llamafactory", prefix="esci_rel")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
