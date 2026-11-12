import math

import pandas as pd

from ecomq.data.build_e7 import extra_class_rows, soft_duplicates, softmax_probs


def _rec(i, gold):
    return {"id": f"esci-{i}", "task": "relevance", "query": "q", "title": "t", "label": {"relevance": gold},
            "group": f"q{i}", "meta": {"query_id": i, "product_id": f"P{i}"}}


def test_softmax_probs_normalizes():
    p = softmax_probs({"exact": 0.0, "substitute": math.log(3), "complement": -50, "irrelevant": -50})
    assert abs(sum(p.values()) - 1) < 1e-9 and abs(p["substitute"] - 0.75) < 1e-6


def test_soft_duplicates_only_confident_es_flips():
    recs = [_rec(1, "exact"), _rec(2, "exact"), _rec(3, "substitute"), _rec(4, "irrelevant")]
    lp = lambda e, s: {"exact": math.log(e), "substitute": math.log(s), "complement": math.log(1e-9), "irrelevant": math.log(1e-9)}  # noqa: E731
    scores = {"esci-1": lp(0.3, 0.7), "esci-2": lp(0.5, 0.5), "esci-3": lp(0.9, 0.1),
              "esci-4": {"exact": 0.0, "substitute": 0.0, "complement": 0.0, "irrelevant": 0.0}}
    dups = soft_duplicates(recs, scores, threshold=0.6)
    assert [d["id"] for d in dups] == ["esci-1-soft", "esci-3-soft"]
    assert dups[0]["label"] == {"relevance": "substitute"} and dups[1]["label"] == {"relevance": "exact"}
    assert recs[0]["label"] == {"relevance": "exact"}          # 原样本不被修改


def test_extra_class_rows_respects_exclusions():
    rows = [{"example_id": i, "query": f"q{i}", "query_id": i, "product_id": f"P{i}", "esci_label": "CI"[i % 2],
             "split": "train", "product_title": f"title {i}"} for i in range(40)]
    pool = pd.DataFrame(rows)
    out = extra_class_rows(pool, {"C": 5, "I": 5}, exclude_queries={0, 1, 2, 3}, exclude_examples={4, 5}, seed=0)
    assert len(out) == 10
    assert not ({r["meta"]["query_id"] for r in out} & {0, 1, 2, 3, 4, 5})
    assert sorted(r["label"]["relevance"] for r in out) == ["complement"] * 5 + ["irrelevant"] * 5


def test_extra_class_rows_are_nested_across_sizes():
    """E8 的嵌套曲线依赖这一点：同一个种子下，追加 n 条是追加 m（> n）条的子集（DEVLOG S20）。"""
    rows = [{"example_id": i, "query": f"q{i}", "query_id": i, "product_id": f"P{i}", "esci_label": "C",
             "split": "train", "product_title": f"title {i}"} for i in range(500)]
    pool = pd.DataFrame(rows)
    ids = {n: {r["id"] for r in extra_class_rows(pool, {"C": n}, set(), set(), seed=47)} for n in (25, 50, 100)}
    assert ids[25] <= ids[50] <= ids[100]
