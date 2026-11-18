"""打分集成（DEVLOG S33）：把多个模型在同一批样本上的标签打分按类平均，得到一份新的打分。

- mode="logp"（默认）：对每一类的 log p 取算术平均，相当于概率的几何平均。先验调整本来就在 log 空间加偏置，
  所以平均之后照常在调参集上选偏置即可。
- mode="prob"：先把 log p 换成概率再取算术平均，最后取 log（用 logsumexp 计算，避免下溢）。

输出与 score_labels 同格式（每行 {"id", "logp": {label: float}}），可以直接交给 prior_adjust、ranking、noise_aware。
  python -m ecomq.eval.ensemble --inputs a-test.jsonl b-test.jsonl c-test.jsonl --output ens-test.jsonl
"""
import argparse

import numpy as np

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.schema import RELEVANCE

MODES = ("logp", "prob")


def average(Ls: list[np.ndarray], mode: str = "logp") -> np.ndarray:
    """Ls：每个成员一个 (样本数, 类别数) 的 log p 矩阵，行顺序必须一致。"""
    if not Ls:
        raise ValueError("至少需要一个成员")
    if mode not in MODES:
        raise ValueError(f"mode 只能是 {MODES}")
    stack = np.stack(Ls)
    if mode == "logp":
        return stack.mean(axis=0)
    return np.logaddexp.reduce(stack, axis=0) - np.log(len(Ls))


def load_scores(path: str) -> tuple[list[str], np.ndarray]:
    rows = list(read_jsonl(path))
    return [r["id"] for r in rows], np.array([[r["logp"][c] for c in RELEVANCE] for r in rows], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="成员的打分文件（score_labels 的输出）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=MODES, default="logp")
    args = ap.parse_args()

    ids, L0 = load_scores(args.inputs[0])
    Ls = [L0]
    for path in args.inputs[1:]:
        other_ids, L = load_scores(path)
        if set(other_ids) != set(ids) or len(other_ids) != len(ids):
            raise SystemExit(f"{path} 的样本 id 与 {args.inputs[0]} 不一致")
        pos = {i: k for k, i in enumerate(other_ids)}
        Ls.append(L[[pos[i] for i in ids]])
    A = average(Ls, args.mode)
    write_jsonl(args.output, [{"id": i, "logp": dict(zip(RELEVANCE, map(float, row)))} for i, row in zip(ids, A)])
    print(f"{len(Ls)} 个成员，{len(ids)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
