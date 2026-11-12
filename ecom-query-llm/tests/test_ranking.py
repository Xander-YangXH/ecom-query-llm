import numpy as np

from ecomq.eval.ranking import ranking_metrics


def _data():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 4, 400)
    return gold, rng.normal(size=(400, 4)) + np.eye(4)[gold]  # 有信号但不完美


def test_ranking_metrics_are_bias_invariant():
    gold, L = _data()
    base = ranking_metrics(gold, L)
    shifted = ranking_metrics(gold, L + np.array([0.0, 1.0, -1.5, 0.75]))
    for k in ("complement", "irrelevant", "exact_vs_sub"):
        for m in base[k]:
            assert abs(base[k][m] - shifted[k][m]) < 1e-12, (k, m)
    same_es = ranking_metrics(gold, L + np.array([0.5, 0.5, -1.5, 0.75]))  # E、S 偏置相同
    assert abs(base["relevant_vs_ci"]["auc"] - same_es["relevant_vs_ci"]["auc"]) < 1e-12
    assert 0.5 < base["complement"]["auc"] <= 1.0


def test_softmax_probability_would_not_be_bias_invariant():
    """记录为什么不用 softmax 概率：其他类的偏置会改变某一类的排序。"""
    from sklearn.metrics import roc_auc_score

    gold, L = _data()

    def auc_c(x):
        e = np.exp(x - x.max(1, keepdims=True))
        return roc_auc_score(gold == 2, (e / e.sum(1, keepdims=True))[:, 2])

    assert abs(auc_c(L) - auc_c(L + np.array([0.0, 3.0, 0.0, -3.0]))) > 1e-4
