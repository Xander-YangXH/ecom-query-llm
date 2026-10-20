"""用 transformers + peft 对测试集批量生成，写出 predictions.jsonl。

不依赖 vLLM，适合 CCDS 单卡的冒烟测试和小规模评测；速度比 vLLM 慢。
示例：
  python -m ecomq.eval.generate_hf --model $MODEL_ROOT/Qwen2.5-7B-Instruct \
      --lora $OUTPUT_ROOT/smoke --test $DATA_ROOT/splits/test.jsonl \
      --output $OUTPUT_ROOT/preds/smoke.jsonl
"""
import argparse

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.schema import build_user_prompt, task_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", help="LoRA 适配器目录；不填则评测基座")
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--load-4bit", action="store_true", help="与 QLoRA 训练一致时建议打开")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # transformers 5.x 用 dtype（旧名 torch_dtype 已弃用）；版本见 requirements-lock-ccds.txt
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    recs = list(read_jsonl(args.test))
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": build_user_prompt(r["query"], r.get("title"), task_of(r))}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in recs
    ]
    rows = []
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i : i + args.batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for r, text in zip(recs[i : i + args.batch_size], texts):
            rows.append({"id": r["id"], "output": text})
    write_jsonl(args.output, rows)
    print(f"写出 {len(rows)} 条预测 -> {args.output}")


if __name__ == "__main__":
    main()
