"""relevance_cot 模型的评测打分（E9a，DEVLOG S23）：先让模型写理由，再以理由为条件给 4 个标签打分。

步骤：
1. 贪心生成：relevance_cot 指令下，模型输出 {"reason": "…", "relevance": "…"}。解析出理由；
   解析失败时用正则取 reason 字段，仍失败则理由为空串（记为 parse_failed）。
2. 打分：对每个标签构造 format_output({"reason": 理由, "relevance": 标签}) + 结束符，四个候选只有标签部分不同。
   取四个 token 序列的最长公共前缀，只对前缀之后的 token 计 log p，也就是 log p(标签部分 | 提示, 理由)。
   不计入理由本身的 log p：它对四个候选相同，不影响 argmax 与先验调整，但会随样本变化，
   计入的话会给与偏置无关的排序指标（AUC / AP）加上与标签无关的噪声。
输出每行 {"id", "logp": {label: float}, "reason", "generated_label", "parse"}，
logp 的格式与 score_labels 相同，可以直接交给 prior_adjust、ranking。
"""
import argparse
import re
from collections import Counter

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.schema import RELEVANCE, build_user_prompt, format_output, parse_output

REASON_RE = re.compile(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"')


def extract_reason(text: str) -> tuple[str, str | None, str]:
    """返回 (理由, 生成的标签或 None, 解析状态 ok / reason_only / parse_failed)。"""
    obj = parse_output(text, "relevance_cot")
    if obj:
        lab = obj["relevance"] if obj["relevance"] in RELEVANCE else None
        return obj["reason"], lab, "ok"
    m = REASON_RE.search(text)
    if m:
        import json

        return json.loads(f'"{m.group(1)}"'), None, "reason_only"
    return "", None, "parse_failed"


def common_prefix_len(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs)
    for k in range(n):
        if any(s[k] != seqs[0][k] for s in seqs[1:]):
            return k
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from ecomq.eval.score_labels import score_items

    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model, **kwargs), args.lora)
    model.eval()
    gen_tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")  # 生成用左侧填充
    tok = AutoTokenizer.from_pretrained(args.model)                           # 打分用右侧填充（score_items 自己填充）
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    recs = list(read_jsonl(args.test))
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": build_user_prompt(r["query"], r.get("title"), "relevance_cot", r.get("product"))}],
        tokenize=False, add_generation_prompt=True) for r in recs]

    texts = []
    for i in range(0, len(prompts), args.batch_size):
        enc = gen_tok(prompts[i : i + args.batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        texts += gen_tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    parsed = [extract_reason(t) for t in texts]

    items = []
    for i, (prompt, (reason, _, _)) in enumerate(zip(prompts, parsed)):
        p_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        fulls = [tok(prompt + format_output({"reason": reason, "relevance": lab}, "relevance_cot") + tok.eos_token,
                     add_special_tokens=False)["input_ids"] for lab in RELEVANCE]
        start = common_prefix_len(fulls)
        assert start > p_len, "candidates diverge inside the prompt"
        items += [(i, lab, full, start) for lab, full in zip(RELEVANCE, fulls)]
    scores = score_items(model, tok, items, len(recs), args.batch_size)

    write_jsonl(args.output, [{"id": r["id"], "logp": scores[i], "reason": parsed[i][0],
                               "generated_label": parsed[i][1], "parse": parsed[i][2]} for i, r in enumerate(recs)])
    status = Counter(p[2] for p in parsed)
    agree = sum(p[1] == max(s, key=s.get) for p, s in zip(parsed, scores) if p[1])
    print(f"写出 {len(recs)} 条 -> {args.output}；解析 {dict(status)}；生成标签与打分最大标签一致 {agree}/{status['ok']}")


if __name__ == "__main__":
    main()
