import numpy as np
import pytest

from ecomq.eval.ensemble import average


def test_single_member_is_identity():
    L = np.log(np.array([[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7]]))
    assert np.allclose(average([L]), L)
    assert np.allclose(average([L], "prob"), L)


def test_logp_mode_is_mean_of_logs():
    a = np.array([[-0.1, -2.0, -5.0, -3.0]])
    b = np.array([[-0.3, -1.0, -4.0, -6.0]])
    assert np.allclose(average([a, b]), (a + b) / 2)


def test_prob_mode_is_log_of_mean_probability():
    a = np.array([[-0.1, -2.0, -5.0, -3.0]])
    b = np.array([[-0.3, -1.0, -4.0, -6.0]])
    expect = np.log((np.exp(a) + np.exp(b)) / 2)
    assert np.allclose(average([a, b], "prob"), expect)


def test_prob_mode_does_not_underflow():
    a = np.full((1, 4), -2000.0)
    b = np.full((1, 4), -2001.0)
    out = average([a, b], "prob")
    assert np.isfinite(out).all() and np.all(out > -2001.0) and np.all(out < -2000.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        average([])
    with pytest.raises(ValueError):
        average([np.zeros((1, 4))], "median")
