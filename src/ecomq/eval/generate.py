"""用 vLLM 对测试集批量生成，写出 predictions.jsonl（与 score.py 解耦）。

示例：
  python -m ecomq.eval.generate --model $MODEL_ROOT/Qwen2.5-7B-Instruct \
      --lora $OUTPUT_ROOT/qwen7b-lora-r16 --test $DATA_ROOT/splits/test.jsonl \
      --output $OUTPUT_ROOT/preds/qwen7b-lora-r16.jsonl

注意：vLLM 的接口随版本变化，升级后请对照官方文档确认 chat / LoRARequest 用法。
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
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel 卡数")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    recs = list(read_jsonl(args.test))
    convs = [[{"role": "user", "content": build_user_prompt(r["query"], r.get("title"), task_of(r))}] for r in recs]
    llm = LLM(
        model=args.model,
        enable_lora=bool(args.lora),
        max_lora_rank=64,
        tensor_parallel_size=args.tp,
    )
    lora = LoRARequest("adapter", 1, args.lora) if args.lora else None
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=args.max_tokens), lora_request=lora)
    rows = [{"id": r["id"], "output": o.outputs[0].text} for r, o in zip(recs, outs)]
    write_jsonl(args.output, rows)
    print(f"写出 {len(rows)} 条预测 -> {args.output}")


if __name__ == "__main__":
    main()
