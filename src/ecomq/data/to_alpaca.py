"""把规范格式转成 LLaMA-Factory 的 Alpaca 格式，并写出 dataset_info.json。

训练配置里用 dataset_dir=<out-dir>，dataset=<prefix>_train / <prefix>_val。
每条样本的 instruction 由其 task 决定（见 schema.INSTRUCTIONS）。
"""
import argparse
import json
from pathlib import Path

from ecomq.data.io import read_jsonl
from ecomq.schema import format_input, format_output, instruction_for, task_of


def convert(rec: dict) -> dict:
    task = task_of(rec)
    return {
        "instruction": instruction_for(task),
        "input": format_input(rec["query"], rec.get("title")),
        "output": format_output(rec["label"], task),
    }


def export(split_dir: str | Path, out_dir: str | Path, prefix: str = "ecom_query") -> dict:
    """把 split_dir 下的 train/val.jsonl 写成 Alpaca json，返回 dataset_info 字典。"""
    out_dir = Path(out_dir)
    info = {}
    for name in ("train", "val"):
        src = Path(split_dir) / f"{name}.jsonl"
        if not src.exists():
            continue
        file_name = f"{prefix}_{name}.json"
        rows = [convert(r) for r in read_jsonl(src)]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / file_name, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        info[f"{prefix}_{name}"] = {
            "file_name": file_name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
        print(f"{name}: {len(rows)} -> {out_dir / file_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, help="包含 train/val/test.jsonl 的目录")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="ecom_query")
    args = ap.parse_args()
    export(args.split_dir, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
