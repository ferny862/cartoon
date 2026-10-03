"""Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014).

The Probabilistic Sharpe Ratio (PSR) is the probability that the true Sharpe
ratio exceeds a threshold, given the observed Sharpe ratio, the number of
observations, and the skewness and kurtosis of returns. The Deflated Sharpe
Ratio is the PSR measured against the Sharpe ratio one would expect from the
*best of N unskilled trials*, which corrects for having tried several
configurations and kept the best one.

All Sharpe ratios here are per period (monthly, not annualized), as the
formulas require.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe_per_period(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """P(true SR > sr_benchmark). ``kurtosis`` is Pearson kurtosis (3 for a normal)."""
    var = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    if n_obs < 2 or var <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(var)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_trial_sharpe: float) -> float:
    """Expected maximum Sharpe ratio among ``n_trials`` unskilled trials whose
    Sharpe ratios have cross-sectional variance ``var_trial_sharpe``."""
    if n_trials <= 1 or not var_trial_sharpe > 0:
        return 0.0
    n = float(n_trials)
    return math.sqrt(var_trial_sharpe) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * math.e))
    )


@dataclass
class DSRResult:
    sharpe: float              # per period
    sharpe_annualized: float
    n_obs: int
    skew: float
    kurtosis: float
    n_trials: int
    expected_max_sharpe: float  # per period, the deflation threshold
    psr_vs_zero: float
    dsr: float

    def as_dict(self) -> dict:
        return asdict(self)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, var_trial_sharpe: float,
                          periods_per_year: int = 12) -> DSRResult:
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    sr = sharpe_per_period(r)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    sr0 = expected_max_sharpe(n_trials, var_trial_sharpe)
    return DSRResult(
        sharpe=sr,
        sharpe_annualized=sr * math.sqrt(periods_per_year),
        n_obs=len(r),
        skew=skew,
        kurtosis=kurt,
        n_trials=n_trials,
        expected_max_sharpe=sr0,
        psr_vs_zero=probabilistic_sharpe_ratio(sr, 0.0, len(r), skew, kurt),
        dsr=probabilistic_sharpe_ratio(sr, sr0, len(r), skew, kurt),
    )


def explain_dsr(name: str, res: DSRResult) -> str:
    """Plain-English reading of one result."""
    verdict = ("strong evidence of a real edge" if res.dsr >= 0.95 else
               "some evidence, but not conclusive" if res.dsr >= 0.80 else
               "no reliable evidence that the result is more than luck")
    return (f"{name}: annualized Sharpe {res.sharpe_annualized:.2f} over {res.n_obs} months. "
            f"After allowing for {res.n_trials} configurations tried, the probability that its true "
            f"Sharpe ratio beats the best result luck alone would produce is {res.dsr:.0%} "
            f"({verdict}). Without that correction it would be {res.psr_vs_zero:.0%}.")
