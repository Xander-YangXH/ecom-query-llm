"""对比多组 relevance 预测：按 query 分组 bootstrap，给出准确率、宏 F1 及两两差值的 95% 置信区间。

按 query 重采样（而不是按样本）：同一 query 下的几个商品判断彼此相关，按样本重采样会低估方差。
示例：
  python -m ecomq.eval.compare --test test.jsonl \
      --preds base=preds/e1-base-zeroshot.jsonl qlora4bit=preds/e1-qlora-r16-4bit.jsonl \
      --output reports/e1-compare.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ecomq.data.io import read_jsonl
from ecomq.eval.metrics import INVALID, per_class_prf
from ecomq.schema import RELEVANCE, parse_output


def _label(text: str) -> str:
    obj = parse_output(text, "relevance")
    lab = obj.get("relevance") if obj else None
    return lab if lab in RELEVANCE else INVALID


def _scores(golds: list[str], preds: list[str]) -> tuple[float, float]:
    acc = sum(g == p for g, p in zip(golds, preds)) / len(golds)
    pc = per_class_prf(golds, preds, RELEVANCE)
    return acc, sum(v["f1"] for v in pc.values()) / len(RELEVANCE)


def _draws(groups: dict, names: list[str], n_boot: int, seed: int):
    """返回点估计与 n_boot 次按 query 重采样的结果：{name: (accuracy, macro_f1)}。"""
    keys = sorted(groups)
    rng = random.Random(seed)

    def evaluate(sample_keys):
        golds, preds = [], defaultdict(list)
        for k in sample_keys:
            for gold, by_name in groups[k]:
                golds.append(gold)
                for n in names:
                    preds[n].append(by_name[n])
        return {n: _scores(golds, preds[n]) for n in names}

    return evaluate(keys), [evaluate([rng.choice(keys) for _ in keys]) for _ in range(n_boot)]


def _ci(values):
    v = sorted(values)
    return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)]


def bootstrap(groups: dict, names: list[str], n_boot: int, seed: int, draws=None) -> dict:
    """groups: query -> list of (gold, {name: pred})。返回每个系统和每对差值的点估计与 95% CI。"""
    point, draws = draws or _draws(groups, names, n_boot, seed)
    out = {"systems": {}, "diffs": {}}
    for n in names:
        out["systems"][n] = {
            "accuracy": round(point[n][0], 4), "accuracy_ci95": _ci([d[n][0] for d in draws]),
            "macro_f1": round(point[n][1], 4), "macro_f1_ci95": _ci([d[n][1] for d in draws]),
        }
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["diffs"][f"{b} - {a}"] = {
                "accuracy": round(point[b][0] - point[a][0], 4),
                "accuracy_ci95": _ci([d[b][0] - d[a][0] for d in draws]),
                "macro_f1": round(point[b][1] - point[a][1], 4),
                "macro_f1_ci95": _ci([d[b][1] - d[a][1] for d in draws]),
            }
    return out


def seed_sets(point: dict, draws: list, sets: dict[str, list[str]]) -> dict:
    """把多个种子归为一组：每次重采样先在组内取平均，再比较组间差值（DEVLOG S16）。
    这里的置信区间只包含测试集抽样的不确定性；种子带来的波动用组内范围（min/max）单独报告。"""
    def mean(d, members, i):
        return sum(d[m][i] for m in members) / len(members)

    out = {"sets": {}, "set_diffs": {}}
    for name, members in sets.items():
        out["sets"][name] = {
            "members": members,
            "accuracy_mean": round(mean(point, members, 0), 4),
            "accuracy_ci95": _ci([mean(d, members, 0) for d in draws]),
            "accuracy_range": [round(min(point[m][0] for m in members), 4), round(max(point[m][0] for m in members), 4)],
            "macro_f1_mean": round(mean(point, members, 1), 4),
            "macro_f1_ci95": _ci([mean(d, members, 1) for d in draws]),
            "macro_f1_range": [round(min(point[m][1] for m in members), 4), round(max(point[m][1] for m in members), 4)],
        }
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["set_diffs"][f"{b} - {a}"] = {
                "accuracy": round(mean(point, sets[b], 0) - mean(point, sets[a], 0), 4),
                "accuracy_ci95": _ci([mean(d, sets[b], 0) - mean(d, sets[a], 0) for d in draws]),
                "macro_f1": round(mean(point, sets[b], 1) - mean(point, sets[a], 1), 4),
                "macro_f1_ci95": _ci([mean(d, sets[b], 1) - mean(d, sets[a], 1) for d in draws]),
                "worst_b_minus_best_a_macro_f1": round(min(point[m][1] for m in sets[b]) - max(point[m][1] for m in sets[a]), 4),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="name=path，可多个")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", nargs="*", default=[], help="按种子分组：name=sys1,sys2,sys3")
    ap.add_argument("--output")
    args = ap.parse_args()

    named = dict(p.split("=", 1) for p in args.preds)
    preds = {n: {r["id"]: _label(r["output"]) for r in read_jsonl(path)} for n, path in named.items()}
    groups = defaultdict(list)
    for rec in read_jsonl(args.test):
        groups[rec.get("group", rec["id"])].append((rec["label"]["relevance"], {n: preds[n][rec["id"]] for n in named}))
    point, draws = _draws(groups, list(named), args.n_boot, args.seed)
    report = {"n_queries": len(groups), "n_examples": sum(len(v) for v in groups.values()), "n_boot": args.n_boot,
              **bootstrap(groups, list(named), args.n_boot, args.seed, draws=(point, draws))}
    if args.sets:
        report.update(seed_sets(point, draws, {k: v.split(",") for k, v in (x.split("=", 1) for x in args.sets)}))
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
