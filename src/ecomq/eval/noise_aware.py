"""噪声感知的评测（DEVLOG S18）：在原指标之外，补充两个辅助指标。

clean   去掉“有争议”的样本后的准确率与宏 F1（四分类）。
        争议集由**不参与比较的模型**（委员会）定义：标注为 exact/substitute，而委员会全体
        （或至少 k 个）都判成另一个（substitute/exact）。不能用被比较的模型自己来定义，
        否则等于把它们共同犯的错从考卷上删掉，分数会被抬高。
merged  把 exact 与 substitute 合并为 relevant，按三分类（relevant/complement/irrelevant）
        计算准确率与宏 F1。不依赖任何模型。
所有指标都用按 query 分组的 bootstrap（numpy 向量化）给出 95% CI，支持两两差值与按种子分组的均值。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ecomq.data.io import read_jsonl
from ecomq.schema import RELEVANCE, parse_output

IDX = {lab: i for i, lab in enumerate(RELEVANCE)}  # exact=0, substitute=1, complement=2, irrelevant=3
INVALID = len(RELEVANCE)                             # 4：解析失败
MERGE = np.array([0, 0, 1, 2, 3])                    # E,S -> 0；C -> 1；I -> 2；invalid -> 3


def load_preds(path: str, ids: list[str]) -> np.ndarray:
    by_id = {}
    for p in read_jsonl(path):
        obj = parse_output(p["output"], "relevance") or {}
        by_id[p["id"]] = IDX.get(obj.get("relevance"), INVALID)
    return np.array([by_id[i] for i in ids])


def disputed_mask(gold: np.ndarray, committee: list[np.ndarray], k: int | None = None) -> np.ndarray:
    """标注为 E（或 S），而委员会中至少 k 个模型（默认全体）判为 S（或 E）。"""
    k = k or len(committee)
    flipped = np.where(gold == 0, 1, np.where(gold == 1, 0, -1))
    votes = sum((c == flipped) for c in committee)
    return (flipped >= 0) & (votes >= k)


def confusion_scores(gold: np.ndarray, pred: np.ndarray, n_cls: int) -> tuple[float, float]:
    """准确率与宏 F1（宏平均只在 n_cls 个真实类别上取；解析失败算作错误的预测）。"""
    width = n_cls + 1
    cm = np.bincount(gold * width + pred, minlength=width * width).reshape(width, width)[:n_cls]
    tp = np.diag(cm[:, :n_cls]).astype(float)
    pred_cnt = cm[:, :n_cls].sum(0) + 0.0  # 只统计真实类别中的预测（不含 invalid 列）
    gold_cnt = cm.sum(1) + 0.0
    prec = np.divide(tp, pred_cnt, out=np.zeros(n_cls), where=pred_cnt > 0)
    rec = np.divide(tp, gold_cnt, out=np.zeros(n_cls), where=gold_cnt > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_cls), where=(prec + rec) > 0)
    return float(tp.sum() / cm.sum()), float(f1.mean())


def metric_fns(disputed: np.ndarray):
    keep = ~disputed

    def full(g, p, rows):
        return confusion_scores(g[rows], p[rows], 4)

    def clean(g, p, rows):
        r = rows[keep[rows]]
        return confusion_scores(g[r], p[r], 4)

    def merged(g, p, rows):
        return confusion_scores(MERGE[g[rows]], MERGE[p[rows]], 3)

    return {"full": full, "clean": clean, "merged": merged}


def bootstrap(gold, preds: dict[str, np.ndarray], groups: np.ndarray, disputed, n_boot: int, seed: int):
    """返回 point[metric][name] = (acc, macro_f1) 与 draws[b][metric][name]。按 query 重采样。"""
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_q = [np.flatnonzero(inv == q) for q in range(len(uniq))]
    fns = metric_fns(disputed)

    def evaluate(rows):
        return {m: {n: f(gold, p, rows) for n, p in preds.items()} for m, f in fns.items()}

    rng = np.random.default_rng(seed)
    point = evaluate(np.arange(len(gold)))
    draws = []
    for _ in range(n_boot):
        qs = rng.integers(0, len(uniq), len(uniq))
        draws.append(evaluate(np.concatenate([rows_by_q[q] for q in qs])))
    return point, draws


def _ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def summarize(point, draws, pairs: list[tuple[str, str]], sets: dict[str, list[str]]):
    out = {}
    for m in point:
        sys_ = {n: {"accuracy": round(v[0], 4), "accuracy_ci95": _ci([d[m][n][0] for d in draws]),
                    "macro_f1": round(v[1], 4), "macro_f1_ci95": _ci([d[m][n][1] for d in draws])}
                for n, v in point[m].items()}
        mean = lambda src, members, i: sum(src[x][i] for x in members) / len(members)  # noqa: E731
        resolve = lambda name: sets.get(name, [name])  # noqa: E731
        diffs = {}
        for a, b in pairs:
            A, B = resolve(a), resolve(b)
            diffs[f"{b} - {a}"] = {
                "accuracy": round(mean(point[m], B, 0) - mean(point[m], A, 0), 4),
                "accuracy_ci95": _ci([mean(d[m], B, 0) - mean(d[m], A, 0) for d in draws]),
                "macro_f1": round(mean(point[m], B, 1) - mean(point[m], A, 1), 4),
                "macro_f1_ci95": _ci([mean(d[m], B, 1) - mean(d[m], A, 1) for d in draws]),
            }
        set_stats = {s: {"accuracy_mean": round(mean(point[m], ms, 0), 4),
                         "accuracy_range": [round(min(point[m][x][0] for x in ms), 4), round(max(point[m][x][0] for x in ms), 4)],
                         "macro_f1_mean": round(mean(point[m], ms, 1), 4),
                         "macro_f1_range": [round(min(point[m][x][1] for x in ms), 4), round(max(point[m][x][1] for x in ms), 4)]}
                     for s, ms in sets.items()}
        out[m] = {"systems": sys_, "sets": set_stats, "diffs": diffs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--committee", nargs="+", required=True, help="定义争议集的模型预测文件（不参与比较）")
    ap.add_argument("--k", type=int, default=0, help="至少 k 个委员判反才算争议；0 表示全体")
    ap.add_argument("--systems", nargs="+", required=True, help="name=path")
    ap.add_argument("--sets", nargs="*", default=[], help="name=sys1,sys2,...（按种子分组取均值）")
    ap.add_argument("--pairs", nargs="*", default=[], help="a:b，计算 b - a；a/b 可以是系统名或组名")
    ap.add_argument("--audit-key", help="可选：用 S17 审计样本检验争议集（analysis/es_audit/key.json）")
    ap.add_argument("--audit-labels", help="与 --audit-key 配合的盲评结果 CSV")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    recs = list(read_jsonl(args.test))
    ids = [r["id"] for r in recs]
    gold = np.array([IDX[r["label"]["relevance"]] for r in recs])
    groups = np.array([r.get("group", r["id"]) for r in recs])
    committee = [load_preds(p, ids) for p in args.committee]
    disputed = disputed_mask(gold, committee, args.k or None)
    systems = {n: load_preds(p, ids) for n, p in (s.split("=", 1) for s in args.systems)}
    sets = {k: v.split(",") for k, v in (s.split("=", 1) for s in args.sets)}
    pairs = [tuple(p.split(":", 1)) for p in args.pairs]

    point, draws = bootstrap(gold, systems, groups, disputed, args.n_boot, args.seed)
    report = {
        "n": int(len(gold)),
        "disputed": {"n": int(disputed.sum()), "share": round(float(disputed.mean()), 4),
                     "gold_E": int((disputed & (gold == 0)).sum()), "gold_S": int((disputed & (gold == 1)).sum()),
                     "committee": args.committee, "k": args.k or len(committee)},
        "metrics": summarize(point, draws, pairs, sets),
    }
    if args.audit_key and args.audit_labels:
        import csv
        key = json.loads(Path(args.audit_key).read_text(encoding="utf-8"))["items"]
        mine = {r["item"]: r["label"] for r in csv.DictReader(open(args.audit_labels, encoding="utf-8"))}
        pos = {i: j for j, i in enumerate(ids)}
        tally = defaultdict(lambda: defaultdict(int))
        for item, info in key.items():
            if not info["set"] in ("E2S", "S2E"):
                continue
            in_d = "disputed" if disputed[pos[info["id"]]] else "not_disputed"
            verdict = "model" if mine[item] == info["model_majority"] else "gold" if mine[item] == info["gold"] else "other"
            tally[in_d][verdict] += 1
        report["audit_check"] = {k: dict(v) for k, v in tally.items()}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"disputed": report["disputed"], "audit_check": report.get("audit_check")}, indent=2))


if __name__ == "__main__":
    main()
