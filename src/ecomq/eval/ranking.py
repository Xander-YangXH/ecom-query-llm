"""与偏置无关的排序指标（DEVLOG S19）：按“一类对其余”计算 AUC / AP。

改变训练集类别比例（E2b、E7）或推理时加偏置，都会改变 F1，但其中有多少只是先验倾向变了，单看 F1 分不清。
argmax(logp + b) 中的 b 是对每一类的常数平移。排序分数的选法保证它不受这种平移影响：
- 某一类 c：直接用 log p(c)。不能用 softmax 后的概率，因为分母里含其他类的偏置，会改变排序（单测覆盖）。
- E vs S：log p(E) − log p(S)。
- 相关（E/S）vs C/I：log(p(E) + p(S))，只对 E、S 加相同偏置时不变。
输入是 score_labels 的输出（每行 {"id", "logp": {label: float}}）。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ecomq.data.io import read_jsonl
from ecomq.schema import RELEVANCE


def load_logp(scores_path: str, ids: list[str]) -> np.ndarray:
    by_id = {r["id"]: [r["logp"][c] for c in RELEVANCE] for r in read_jsonl(scores_path)}
    return np.array([by_id[i] for i in ids], dtype=np.float64)


def ranking_metrics(gold: np.ndarray, L: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {}
    for c in ("complement", "irrelevant"):
        y, s = gold == RELEVANCE.index(c), L[:, RELEVANCE.index(c)]
        out[c] = {"auc": roc_auc_score(y, s), "ap": average_precision_score(y, s)}
    y, s = gold <= 1, np.logaddexp(L[:, 0], L[:, 1])          # E/S（相关）对 C/I
    out["relevant_vs_ci"] = {"auc": roc_auc_score(y, s), "ap": average_precision_score(y, s)}
    m = gold <= 1                                             # 只在标注为 E/S 的样本上比较 E 与 S
    out["exact_vs_sub"] = {"auc": roc_auc_score(gold[m] == 0, (L[:, 0] - L[:, 1])[m])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="测试集记录（提供 id 与标注）")
    ap.add_argument("--groups", nargs="+", required=True, help="name=scores1,scores2,...（同一设定的多个种子）")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    recs = list(read_jsonl(args.test))
    ids = [r["id"] for r in recs]
    gold = np.array([RELEVANCE.index(r["label"]["relevance"]) for r in recs])
    results = {}
    for spec in args.groups:
        name, paths = spec.split("=", 1)
        rows = [ranking_metrics(gold, load_logp(p, ids)) for p in paths.split(",")]
        results[name] = {k: {m: {"mean": round(float(np.mean([r[k][m] for r in rows])), 4),
                                 "range": [round(float(min(r[k][m] for r in rows)), 4),
                                           round(float(max(r[k][m] for r in rows)), 4)]}
                             for m in rows[0][k]} for k in rows[0]}
        print(name, json.dumps({k: {m: v["mean"] for m, v in d.items()} for k, d in results[name].items()}))
    report = {"note": "one-vs-rest ranking metrics on raw log p (class c: log p(c); E vs S: log p(E) - log p(S); relevant: log(p(E)+p(S))); invariant to per-class bias; seed mean and range",
              "test": Path(args.test).name, "results": results}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
