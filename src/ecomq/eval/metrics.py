"""评测指标。输入为 (参考标签, 模型原始输出文本) 对，输出各任务指标与格式合法率。"""
from collections import defaultdict

from ecomq.schema import parse_output


def macro_f1(golds: list[str], preds: list[str]) -> float:
    labels = sorted(set(golds) | set(preds))
    scores = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _attr_pairs(attrs: dict) -> set[tuple[str, str]]:
    return {(str(k), str(v)) for k, v in attrs.items()}


def evaluate(golds: list[dict], raw_outputs: list[str]) -> dict:
    """golds 为 label dict 列表；raw_outputs 为模型输出文本。解析失败的样本按全错计分。"""
    assert len(golds) == len(raw_outputs)
    parsed = [parse_output(t) for t in raw_outputs]
    n = len(golds)
    valid = sum(p is not None for p in parsed)

    intent_g, intent_p = [], []
    tp = fp = fn = 0
    rel_correct = rel_total = 0
    rewrite_exact = 0
    for gold, pred in zip(golds, parsed):
        intent_g.append(gold["intent"])
        intent_p.append(pred["intent"] if pred else "<invalid>")

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


def evaluate_by_difficulty(records: list[dict], raw_outputs: list[str]) -> dict[str, dict]:
    """按 rec["meta"]["difficulty"] 分层汇总，用于难例集分析。"""
    groups = defaultdict(lambda: ([], []))
    for rec, out in zip(records, raw_outputs):
        tag = (rec.get("meta") or {}).get("difficulty", "unlabeled")
        groups[tag][0].append(rec["label"])
        groups[tag][1].append(out)
    return {tag: evaluate(g, o) for tag, (g, o) in groups.items()}
