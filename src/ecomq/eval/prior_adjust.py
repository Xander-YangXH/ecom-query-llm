"""推理时先验调整（DEVLOG S12）：pred = argmax_c [log p(c|x) + b_c]，b_exact 固定为 0。

偏置 b 只在调参集（tune）上搜索，目标是宏 F1 最大；测试集只用于最终报告。
搜索方法：坐标上升，每轮依次对 substitute/complement/irrelevant 的偏置做一维网格搜索，直到不再变化。
输出：
  --report     JSON：偏置、调参集与测试集上调整前后的指标、与贪心生成结果的一致率、单类召回–精确率曲线
  --preds-out  与 generate 输出同格式的预测文件，可直接交给 ecomq.eval.compare 做 bootstrap 对比
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.eval.metrics import evaluate_relevance
from ecomq.schema import RELEVANCE, parse_output

GRID = np.round(np.arange(-6.0, 8.0001, 0.25), 2)


def load(recs_path: str, scores_path: str):
    recs = list(read_jsonl(recs_path))
    by_id = {r["id"]: r["logp"] for r in read_jsonl(scores_path)}
    L = np.array([[by_id[r["id"]][c] for c in RELEVANCE] for r in recs], dtype=np.float64)
    gold = np.array([RELEVANCE.index(r["label"]["relevance"]) for r in recs])
    return recs, L, gold


def macro_f1(gold: np.ndarray, pred: np.ndarray, k: int = len(RELEVANCE)) -> float:
    cm = np.bincount(gold * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return float(f1.mean())


def predict(L: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(L + bias, axis=1)


def tune_bias(L: np.ndarray, gold: np.ndarray, grid=GRID, max_rounds: int = 10) -> tuple[np.ndarray, float]:
    bias = np.zeros(L.shape[1])
    best = macro_f1(gold, predict(L, bias))
    for _ in range(max_rounds):
        changed = False
        for c in range(1, L.shape[1]):  # c=0（exact）固定为 0，作为参照
            for v in grid:
                trial = bias.copy()
                trial[c] = v
                score = macro_f1(gold, predict(L, trial))
                if score > best + 1e-12:
                    best, bias, changed = score, trial, True
        if not changed:
            break
    return bias, best


def class_curve(L, gold, bias, cls: int, grid=GRID) -> list[dict]:
    """固定其余偏置，只改变 cls 的偏置，返回该类的召回与精确率。"""
    out = []
    for v in grid:
        b = bias.copy()
        b[cls] = v
        pred = predict(L, b)
        tp = int(((pred == cls) & (gold == cls)).sum())
        npred, ngold = int((pred == cls).sum()), int((gold == cls).sum())
        out.append({"bias": float(v), "recall": tp / ngold if ngold else 0.0,
                    "precision": tp / npred if npred else 0.0, "macro_f1": macro_f1(gold, pred)})
    return out


def as_outputs(pred: np.ndarray) -> list[str]:
    return [json.dumps({"relevance": RELEVANCE[p]}) for p in pred]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-recs", required=True)
    ap.add_argument("--tune-scores", required=True)
    ap.add_argument("--test-recs", required=True)
    ap.add_argument("--test-scores", required=True)
    ap.add_argument("--gen-preds", help="同一模型贪心生成的预测，用于核对打分实现（无偏置时应基本一致）")
    ap.add_argument("--curve-class", default="irrelevant")
    ap.add_argument("--report", required=True)
    ap.add_argument("--preds-out", required=True)
    ap.add_argument("--raw-preds-out", help="同时写出不加偏置（直接取概率最大）的预测，便于 bootstrap 对比调整前后")
    args = ap.parse_args()

    _, Lt, gt = load(args.tune_recs, args.tune_scores)
    test_recs, Ls, gs = load(args.test_recs, args.test_scores)
    zero = np.zeros(len(RELEVANCE))
    bias, tune_best = tune_bias(Lt, gt)

    golds = [{"relevance": RELEVANCE[g]} for g in gs]
    raw_pred, adj_pred = predict(Ls, zero), predict(Ls, bias)
    report = {
        "bias": dict(zip(RELEVANCE, map(float, bias))),
        "tune": {"n": int(len(gt)), "macro_f1_before": macro_f1(gt, predict(Lt, zero)), "macro_f1_after": tune_best},
        "test_argmax_no_bias": evaluate_relevance(golds, as_outputs(raw_pred)),
        "test_adjusted": evaluate_relevance(golds, as_outputs(adj_pred)),
        "curve": {args.curve_class: class_curve(Ls, gs, bias, RELEVANCE.index(args.curve_class))},
    }
    if args.gen_preds:
        gen = {p["id"]: (parse_output(p["output"], "relevance") or {}).get("relevance") for p in read_jsonl(args.gen_preds)}
        agree = [gen[r["id"]] == RELEVANCE[p] for r, p in zip(test_recs, raw_pred)]
        report["agreement_with_greedy_generation"] = sum(agree) / len(agree)

    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(args.preds_out, [{"id": r["id"], "output": o} for r, o in zip(test_recs, as_outputs(adj_pred))])
    if args.raw_preds_out:
        write_jsonl(args.raw_preds_out, [{"id": r["id"], "output": o} for r, o in zip(test_recs, as_outputs(raw_pred))])
    t = report["test_adjusted"]
    print(json.dumps({"bias": report["bias"], "tune": report["tune"],
                      "test_before": {k: report["test_argmax_no_bias"][k] for k in ("accuracy", "macro_f1")},
                      "test_after": {k: t[k] for k in ("accuracy", "macro_f1")},
                      "agreement": report.get("agreement_with_greedy_generation")}, indent=2))


if __name__ == "__main__":
    main()
