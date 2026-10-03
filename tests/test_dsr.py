import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.validation.dsr import (
    deflated_sharpe_ratio, expected_max_sharpe, explain_dsr, probabilistic_sharpe_ratio, sharpe_per_period,
)


def test_psr_hand_computed():
    assert probabilistic_sharpe_ratio(0.2, 0.2, 100, 0.0, 3.0) == pytest.approx(0.5)
    # sr=0.2, n=61, normal returns: z = 0.2*sqrt(60)/sqrt(1 + 0.5*0.04) = 1.5340
    z = 0.2 * math.sqrt(60) / math.sqrt(1.02)
    assert probabilistic_sharpe_ratio(0.2, 0.0, 61, 0.0, 3.0) == pytest.approx(stats.norm.cdf(z))
    assert stats.norm.cdf(z) == pytest.approx(0.9375, abs=1e-3)
    # negative skew and fat tails widen the uncertainty and lower the PSR
    assert probabilistic_sharpe_ratio(0.2, 0.0, 61, -1.0, 6.0) < probabilistic_sharpe_ratio(0.2, 0.0, 61, 0.0, 3.0)


def test_expected_max_sharpe_matches_monte_carlo():
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(3.25, abs=0.01)
    rng = np.random.default_rng(0)
    mc = rng.standard_normal((20000, 100)).max(axis=1).mean()          # about 2.51
    assert expected_max_sharpe(100, 1.0) == pytest.approx(mc, abs=0.05)
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))


def test_dsr_deflates_best_of_many_noise_strategies():
    """Pick the best of 50 pure-noise strategies: the uncorrected PSR often looks
    significant, the deflated one rarely does."""
    false_psr = false_dsr = 0
    seeds = range(100)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        R = rng.normal(0.0, 0.04, (240, 50))
        srs = R.mean(0) / R.std(0, ddof=1)
        best = int(np.argmax(srs))
        res = deflated_sharpe_ratio(pd.Series(R[:, best]), n_trials=50, var_trial_sharpe=float(srs.var(ddof=1)))
        false_psr += res.psr_vs_zero > 0.95
        false_dsr += res.dsr > 0.95
    assert false_psr > 50            # selection bias makes the raw PSR look significant
    assert false_dsr <= 15           # deflation brings false positives near the nominal 5%


def test_dsr_detects_a_real_edge():
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.04, (240, 9))
    skilled = rng.normal(0.012, 0.03, 240)                               # annualized Sharpe ~1.4
    R = np.column_stack([noise, skilled])
    srs = R.mean(0) / R.std(0, ddof=1)
    res = deflated_sharpe_ratio(pd.Series(skilled), n_trials=10, var_trial_sharpe=float(srs.var(ddof=1)))
    assert res.dsr > 0.95            # conventional significance threshold
    assert res.sharpe_annualized == pytest.approx(sharpe_per_period(skilled) * math.sqrt(12))
    assert "strong evidence" in explain_dsr("x", res)


def test_dsr_single_trial_equals_psr_vs_zero():
    r = pd.Series(np.random.default_rng(2).normal(0.005, 0.04, 120))
    res = deflated_sharpe_ratio(r, n_trials=1, var_trial_sharpe=0.0)
    assert res.dsr == pytest.approx(res.psr_vs_zero)
