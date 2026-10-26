"""E9 的推理数据（DEVLOG S21）：为 relevance 训练样本生成一句理由，得到 relevance_cot 记录。

做法：给基座模型 query、商品标题和**标注标签**，让它用一句话（不超过 40 个词）说明商品与 query 的关系，
不许写出标签名。理由写在标签前面，训练时标签以理由为条件。
质量检查（写进 --report）：
  flags        理由中出现标签词（exact / substitute / complement / irrelevant 及其变形）、超长、为空。
               有问题的样本用采样（temperature 0.7）重试一次，仍有问题的保留标记，由下游决定是否丢弃。
  blind check  只给基座模型 query、标题和理由（不给标签），用 relevance 指令对 4 个标签打分，
               与标注比对；同时给出不带理由的零样本结果作对照。衡量理由里有没有足以判断标签的信息。
  speed        生成的 token 数、耗时、tokens/s，并按此估算全量所需时间。
另外写出 --review（按类别抽样的 Markdown 表），供人工逐条检查理由是否合理。
"""
import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.schema import LABEL_DEFINITIONS, MAX_REASON_WORDS, RELEVANCE, build_user_prompt, validate_record

LABEL_WORDS = re.compile(r"\b(exact\w*|substitut\w*|complement\w*|irrelevan\w*)\b", re.IGNORECASE)

RATIONALE_PROMPT = (
    "You are annotating e-commerce search data. A human judge has labeled how a product relates to a search query, "
    "using these labels: " + LABEL_DEFINITIONS + "\n"
    "query: {query}\nproduct title: {title}\nhuman label: {label}\n\n"
    "Write ONE sentence of at most 30 words explaining the relationship between the product and what the query is "
    "looking for, consistent with the human label. Be concrete: say what the product is, and which requirement of "
    "the query it meets or misses, or how it would be used together with the item the query asks for. "
    "Do not use the words exact, substitute, complement or irrelevant. Output only the sentence."
)


def stratified_sample(recs: list[dict], n: int, seed: int) -> list[dict]:
    """每个类别各取 n/4 条（不足则全取），用于试点；n=0 表示全部。"""
    if not n:
        return recs
    by_label = defaultdict(list)
    for r in recs:
        by_label[r["label"]["relevance"]].append(r)
    rng = random.Random(seed)
    out = []
    for lab in RELEVANCE:
        pool = by_label[lab]
        out += rng.sample(pool, min(n // len(RELEVANCE), len(pool)))
    rng.shuffle(out)
    return out


def clean_reason(text: str) -> str:
    text = text.strip().strip('"').strip()
    return " ".join(text.split())


def reason_flags(reason: str) -> list[str]:
    flags = []
    if not reason:
        flags.append("empty")
    if LABEL_WORDS.search(reason):
        flags.append("label_word")
    if len(reason.split()) > MAX_REASON_WORDS:
        flags.append("too_long")
    return flags


def to_cot_record(rec: dict, reason: str, flags: list[str], retried: bool) -> dict:
    out = json.loads(json.dumps(rec))
    out["task"] = "relevance_cot"
    out["label"] = {"reason": reason, "relevance": rec["label"]["relevance"]}
    out.setdefault("meta", {})["rationale"] = {"flags": flags, "retried": retried}
    return out


def macro_f1(gold: list[str], pred: list[str]) -> float:
    f1s = []
    for c in RELEVANCE:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        np_, ng = sum(p == c for p in pred), sum(g == c for g in gold)
        prec, rec = (tp / np_ if np_ else 0.0), (tp / ng if ng else 0.0)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def review_table(rows: list[dict], per_class: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = ["| # | label | query | product title | reason | flags |", "|---|---|---|---|---|---|"]
    k = 0
    for lab in RELEVANCE:
        pool = [r for r in rows if r["label"]["relevance"] == lab]
        for r in rng.sample(pool, min(per_class, len(pool))):
            k += 1
            cells = [str(k), lab, r["query"], r["title"], r["label"]["reason"], ",".join(r["meta"]["rationale"]["flags"])]
            lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="relevance 任务的规范记录 jsonl（例如 E2a 训练集）")
    ap.add_argument("--output", required=True, help="relevance_cot 记录 jsonl")
    ap.add_argument("--report", required=True)
    ap.add_argument("--review", help="按类别抽样的 Markdown 表，供人工检查")
    ap.add_argument("--review-per-class", type=int, default=13)
    ap.add_argument("--n", type=int, default=0, help=">0 时按类别分层抽 n 条（试点）；0 为全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=72)
    ap.add_argument("--blind-check", action="store_true", help="只看理由预测标签，并与不带理由的零样本对照")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ecomq.eval.score_labels import label_logps

    recs = stratified_sample(list(read_jsonl(args.input)), args.n, args.seed)
    assert all(r["task"] == "relevance" for r in recs)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 基座模型、不挂 LoRA，用 bf16（比 4bit 快，显存足够）
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    def chat(content: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    def generate(items: list[dict], sample: bool) -> tuple[list[str], int, float]:
        prompts = [chat(RATIONALE_PROMPT.format(query=r["query"], title=r["title"], label=r["label"]["relevance"]))
                   for r in items]
        texts, n_tok, start = [], 0, time.time()
        for i in range(0, len(prompts), args.batch_size):
            enc = tok(prompts[i : i + args.batch_size], return_tensors="pt", padding=True).to(model.device)
            gen = dict(max_new_tokens=args.max_new_tokens, do_sample=sample)
            if sample:
                gen.update(temperature=0.7, top_p=0.95)
            with torch.no_grad():
                out = model.generate(**enc, **gen)
            new = out[:, enc["input_ids"].shape[1] :]
            n_tok += int((new != tok.pad_token_id).sum())
            texts += tok.batch_decode(new, skip_special_tokens=True)
        return [clean_reason(t) for t in texts], n_tok, time.time() - start

    torch.manual_seed(args.seed)
    reasons, n_tok, secs = generate(recs, sample=False)
    flags = [reason_flags(x) for x in reasons]
    first_flags = Counter(f for fl in flags for f in fl)
    redo = [i for i, fl in enumerate(flags) if fl]
    retried = set()
    if redo:
        again, n2, s2 = generate([recs[i] for i in redo], sample=True)
        n_tok, secs = n_tok + n2, secs + s2
        for i, x in zip(redo, again):
            retried.add(i)
            if not reason_flags(x):
                reasons[i], flags[i] = x, []
    rows = [to_cot_record(r, x, fl, i in retried) for i, (r, x, fl) in enumerate(zip(recs, reasons, flags))]
    bad_schema = sum(bool(validate_record(r)) for r in rows if not r["meta"]["rationale"]["flags"])
    assert bad_schema == 0, f"{bad_schema} unflagged rows fail schema validation"
    write_jsonl(args.output, rows)

    words = [len(r["label"]["reason"].split()) for r in rows]
    report = {
        "input": Path(args.input).name, "n": len(rows), "stratified": bool(args.n),
        "labels": dict(Counter(r["label"]["relevance"] for r in rows)),
        "flags_first_pass": dict(first_flags), "retried": len(redo),
        "flagged_after_retry": dict(Counter(f for fl in flags for f in fl)),
        "flagged_rows_after_retry": sum(bool(fl) for fl in flags),
        "reason_words": {"mean": round(sum(words) / len(words), 1), "max": max(words),
                         "p90": sorted(words)[int(0.9 * (len(words) - 1))]},
        "speed": {"generated_tokens": n_tok, "seconds": round(secs, 1), "tokens_per_s": round(n_tok / secs, 1),
                  "rows_per_s": round((len(rows) + len(redo)) / secs, 2), "batch_size": args.batch_size},
    }

    if args.blind_check:
        gold = [r["label"]["relevance"] for r in rows]
        plain = [chat(build_user_prompt(r["query"], r["title"], "relevance")) for r in rows]
        with_reason = [chat(build_user_prompt(r["query"], r["title"], "relevance") + f"\nexplanation: {r['label']['reason']}")
                       for r in rows]
        out = {}
        for name, prompts in (("zero_shot", plain), ("with_reason", with_reason)):
            scores = label_logps(model, tok, prompts, args.batch_size)
            pred = [max(s, key=s.get) for s in scores]
            per_class = {c: round(sum(g == p == c for g, p in zip(gold, pred)) / max(1, gold.count(c)), 3)
                         for c in RELEVANCE}
            out[name] = {"accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
                         "macro_f1": round(macro_f1(gold, pred), 4), "recall_per_class": per_class}
        report["blind_check"] = out

    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.review:
        Path(args.review).write_text(review_table(rows, args.review_per_class, args.seed), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
