"""生成冒烟测试数据：把 data_sample 里的十几条样本复制成 N 条训练数据。

只用来验证流程（数据加载、4bit 加载、前后向、保存、推理、打分）能不能跑通。
验证集和测试集就是原始样本本身，与训练集完全重叠，所以得到的指标没有任何参考意义。
输出位置：
  <data-root>/splits/{train,val,test}.jsonl
  <data-root>/llamafactory/ecom_query_{train,val}.json + dataset_info.json
"""
import argparse
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.data.to_alpaca import export
from ecomq.paths import DATA_ROOT, REPO_ROOT


def expand(rows: list[dict], n: int) -> list[dict]:
    return [{**rows[k % len(rows)], "id": f"{rows[k % len(rows)]['id']}-x{k}"} for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-file", default=str(REPO_ROOT / "data_sample" / "seed_sample.jsonl"))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--n", type=int, default=400, help="训练样本条数")
    args = ap.parse_args()

    root = Path(args.data_root)
    rows = list(read_jsonl(args.seed_file))
    write_jsonl(root / "splits" / "train.jsonl", expand(rows, args.n))
    write_jsonl(root / "splits" / "val.jsonl", rows)
    write_jsonl(root / "splits" / "test.jsonl", rows)
    export(root / "splits", root / "llamafactory")
    print(f"冒烟数据已写入 {root}（val/test 与 train 重叠，指标不具参考意义）")


if __name__ == "__main__":
    main()
