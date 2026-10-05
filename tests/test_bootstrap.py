import numpy as np
import pandas as pd
import pytest

from src.validation.bootstrap import bootstrap_difference, circular_block_indices, explain_bootstrap


def series(values, start="2000-01-31"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="ME"))


def test_block_indices_are_contiguous_and_wrap():
    idx = circular_block_indices(10, 4, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10)
    for row in idx:
        for k in range(0, 8, 4):
            block = row[k:k + 4]
            assert all((block[j + 1] - block[j]) % 10 == 1 for j in range(len(block) - 1))


def test_identical_series_give_zero_difference():
    r = series(np.random.default_rng(1).normal(0.01, 0.04, 120))
    res = bootstrap_difference(r, r, n_samples=500)
    assert res.return_diff == 0 and res.return_diff_ci == (0.0, 0.0)
    assert res.sharpe_diff_ci == (0.0, 0.0)


def test_observed_difference_hand_computed():
    s = series([0.01] * 24)
    b = series([0.005] * 24)
    res = bootstrap_difference(s, b, n_samples=200, block=12)
    assert res.return_diff == pytest.approx(1.01 ** 12 - 1.005 ** 12)
    assert res.return_diff_ci[0] == pytest.approx(res.return_diff)       # constant returns: no spread


def test_clear_edge_excludes_zero():
    rng = np.random.default_rng(2)
    b = series(rng.normal(0.006, 0.045, 240))
    s = b + 0.01 + series(rng.normal(0, 0.005, 240))
    res = bootstrap_difference(s, b, n_samples=2000)
    assert res.return_diff_ci[0] > 0 and res.sharpe_diff_ci[0] > 0
    assert res.return_diff_prob_positive > 0.99
    assert "distinguishable from zero" in explain_bootstrap("x", res)


def test_no_edge_ci_usually_contains_zero():
    covered = 0
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        common = rng.normal(0.006, 0.04, 240)
        s = series(common + rng.normal(0, 0.01, 240))
        b = series(common + rng.normal(0, 0.01, 240))
        res = bootstrap_difference(s, b, n_samples=1000, seed=seed)
        covered += res.return_diff_ci[0] <= 0 <= res.return_diff_ci[1]
    assert covered >= 34                      # nominal 95% coverage, allowing for sampling noise


def test_deterministic_with_seed_and_short_series_rejected():
    rng = np.random.default_rng(3)
    s, b = series(rng.normal(0.01, 0.04, 60)), series(rng.normal(0.01, 0.04, 60))
    assert bootstrap_difference(s, b, seed=7, n_samples=300) == bootstrap_difference(s, b, seed=7, n_samples=300)
    with pytest.raises(ValueError):
        bootstrap_difference(s.iloc[:20], b.iloc[:20], block=12)
