import numpy as np

from ecomq.eval.metrics import evaluate_relevance
from ecomq.eval.noise_aware import MERGE, confusion_scores, disputed_mask, metric_fns
from ecomq.schema import RELEVANCE


def test_confusion_scores_match_reference_metrics():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 4, 500)
    pred = np.where(rng.random(500) < 0.7, gold, rng.integers(0, 5, 500))  # 含 invalid(4)
    acc, mf1 = confusion_scores(gold, pred, 4)
    names = RELEVANCE + ["<invalid>"]
    ref = evaluate_relevance([{"relevance": names[g]} for g in gold],
                             ['{"relevance": "%s"}' % names[p] if p < 4 else "oops" for p in pred])
    assert abs(acc - ref["accuracy"]) < 1e-12 and abs(mf1 - ref["macro_f1"]) < 1e-12


def test_disputed_requires_committee_flip_between_e_and_s():
    gold = np.array([0, 0, 1, 1, 2, 3])
    c1 = np.array([1, 1, 0, 1, 0, 1])
    c2 = np.array([1, 0, 0, 0, 0, 1])
    d = disputed_mask(gold, [c1, c2])
    assert d.tolist() == [True, False, True, False, False, False]     # C/I 样本永远不算争议
    assert disputed_mask(gold, [c1, c2], k=1).tolist() == [True, True, True, True, False, False]


def test_merged_and_clean_metrics():
    gold = np.array([0, 1, 2, 3])
    pred = np.array([1, 0, 2, 3])                     # 只在 E/S 之间互换
    fns = metric_fns(np.array([True, True, False, False]))
    rows = np.arange(4)
    assert fns["full"](gold, pred, rows)[0] == 0.5
    assert fns["merged"](gold, pred, rows) == (1.0, 1.0)   # 合并后全对
    assert fns["clean"](gold, pred, rows)[0] == 1.0        # 去掉两个争议样本后全对
    assert MERGE.tolist() == [0, 0, 1, 2, 3]
