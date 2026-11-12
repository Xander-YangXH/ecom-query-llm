import numpy as np

from ecomq.eval.prior_adjust import class_curve, macro_f1, predict, tune_bias


def test_macro_f1_matches_hand_computation():
    gold = np.array([0, 0, 1, 3])
    pred = np.array([0, 1, 1, 0])
    # exact: P=1/2 R=1/2 F1=1/2; substitute: P=1/2 R=1 F1=2/3; complement: 0; irrelevant: 0
    assert abs(macro_f1(gold, pred) - (0.5 + 2 / 3) / 4) < 1e-9


def test_tune_bias_recovers_minority_class():
    rng = np.random.default_rng(0)
    n = 2000
    gold = rng.choice(4, size=n, p=[0.7, 0.2, 0.03, 0.07])
    L = rng.normal(0, 1, size=(n, 4))
    L[np.arange(n), gold] += 1.5      # 有信息量的打分
    L[:, 0] += 2.0                    # 但整体偏向 exact（类似未校准的模型）
    before = macro_f1(gold, predict(L, np.zeros(4)))
    bias, after = tune_bias(L, gold)
    assert bias[0] == 0.0
    assert after > before + 0.05
    assert (bias[1:] > 0).all()       # 少数类的偏置应被调高


def test_class_curve_is_monotone_in_recall():
    rng = np.random.default_rng(1)
    gold = rng.choice(4, size=500)
    L = rng.normal(size=(500, 4))
    curve = class_curve(L, gold, np.zeros(4), cls=3)
    recalls = [c["recall"] for c in curve]
    assert all(b >= a for a, b in zip(recalls, recalls[1:]))
