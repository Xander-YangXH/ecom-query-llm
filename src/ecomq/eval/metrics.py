"""评测指标。输入为规范格式记录与模型原始输出文本，按任务分别计算。

full       格式合法率、意图准确率与宏 F1、属性实体级 P/R/F1、相关性准确率、改写完全匹配
relevance  格式合法率、准确率（单标签多分类下 = micro-F1）、宏 F1、各类 P/R/F1、混淆矩阵
解析失败或标签不合法的输出按全错计分。
"""
from collections import Counter, defaultdict

from ecomq.schema import RELEVANCE, parse_output, task_of

INVALID = "<invalid>"


def per_class_prf(golds: list[str], preds: list[str], labels: list[str]) -> dict[str, dict]:
    out = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(golds: list[str], preds: list[str], labels: list[str] | None = None) -> float:
    labels = labels or sorted(set(golds) | set(preds))
    scores = per_class_prf(golds, preds, labels)
    return sum(s["f1"] for s in scores.values()) / len(scores) if scores else 0.0


def _attr_pairs(attrs: dict) -> set[tuple[str, str]]:
    return {(str(k).lower(), str(v).lower()) for k, v in attrs.items()}


def evaluate_full(golds: list[dict], raw_outputs: list[str]) -> dict:
    assert len(golds) == len(raw_outputs)
    parsed = [parse_output(t, "full") for t in raw_outputs]
    n = len(golds)
    valid = sum(p is not None for p in parsed)

    intent_g, intent_p = [], []
    tp = fp = fn = 0
    rel_correct = rel_total = 0
    rewrite_exact = 0
    for gold, pred in zip(golds, parsed):
        intent_g.append(gold["intent"])
        intent_p.append(pred["intent"] if pred else INVALID)

        g_pairs = _attr_pairs(gold["attributes"])
        p_pairs = _attr_pairs(pred["attributes"]) if pred else set()
        tp += len(g_pairs & p_pairs)
        fp += len(p_pairs - g_pairs)
        fn += len(g_pairs - p_pairs)

        if gold.get("relevance") is not None:
            rel_total += 1
            rel_correct += bool(pred and pred["relevance"] == gold["relevance"])
        rewrite_exact += bool(pred and pred["rewrite"] == gold["rewrite"])

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "format_valid_rate": valid / n if n else 0.0,
        "intent_acc": sum(g == p for g, p in zip(intent_g, intent_p)) / n if n else 0.0,
        "intent_macro_f1": macro_f1(intent_g, intent_p),
        "attr_precision": prec,
        "attr_recall": rec,
        "attr_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "relevance_acc": rel_correct / rel_total if rel_total else None,
        "rewrite_exact_match": rewrite_exact / n if n else 0.0,
    }


def evaluate_relevance(golds: list[dict], raw_outputs: list[str]) -> dict:
    assert len(golds) == len(raw_outputs)
    g = [x["relevance"] for x in golds]
    p = []
    valid = 0
    for text in raw_outputs:
        obj = parse_output(text, "relevance")
        label = obj.get("relevance") if obj else None
        if label in RELEVANCE:
            valid += 1
            p.append(label)
        else:
            p.append(INVALID)
    n = len(g)
    confusion = defaultdict(Counter)
    for gold, pred in zip(g, p):
        confusion[gold][pred] += 1
    per_class = per_class_prf(g, p, RELEVANCE)
    return {
        "n": n,
        "format_valid_rate": valid / n if n else 0.0,
        "accuracy": sum(a == b for a, b in zip(g, p)) / n if n else 0.0,
        "macro_f1": sum(s["f1"] for s in per_class.values()) / len(RELEVANCE),
        "per_class": per_class,
        "gold_distribution": dict(Counter(g)),
        "pred_distribution": dict(Counter(p)),
        "confusion": {gold: dict(row) for gold, row in confusion.items()},
    }


def evaluate_by_difficulty(records: list[dict], raw_outputs: list[str]) -> dict[str, dict]:
    """full 任务按 rec["meta"]["difficulty"] 分层汇总，用于难例集分析。"""
    groups = defaultdict(lambda: ([], []))
    for rec, out in zip(records, raw_outputs):
        tag = (rec.get("meta") or {}).get("difficulty", "unlabeled")
        groups[tag][0].append(rec["label"])
        groups[tag][1].append(out)
    return {tag: evaluate_full(g, o) for tag, (g, o) in groups.items()}


def evaluate_records(records: list[dict], raw_outputs: list[str]) -> dict:
    """按 task 分组评测，返回 {task: 指标}；full 任务附带 by_difficulty。"""
    by_task = defaultdict(lambda: ([], []))
    for rec, out in zip(records, raw_outputs):
        by_task[task_of(rec)][0].append(rec)
        by_task[task_of(rec)][1].append(out)
    report = {}
    for task, (recs, outs) in by_task.items():
        golds = [r["label"] for r in recs]
        if task in ("relevance", "relevance_cot"):  # relevance_cot 的输出多一个 reason 字段，标签解析方式相同
            report[task] = evaluate_relevance(golds, outs)
        else:
            report[task] = evaluate_full(golds, outs)
            report[task]["by_difficulty"] = evaluate_by_difficulty(recs, outs)
    return report
