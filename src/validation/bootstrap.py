"""Circular block bootstrap of paired monthly returns.

Resamples whole blocks of consecutive months (wrapping around the end), using
the same block draws for the strategy, the benchmark and the risk-free rate,
so their co-movement and short-run autocorrelation are preserved. Produces
confidence intervals for the difference in annualized return and in Sharpe
ratio between a strategy and the benchmark.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class BootstrapResult:
    n_obs: int
    block: int
    n_samples: int
    return_diff: float                # observed annualized return difference (strategy - benchmark)
    return_diff_ci: tuple[float, float]
    return_diff_prob_positive: float
    sharpe_diff: float
    sharpe_diff_ci: tuple[float, float]
    sharpe_diff_prob_positive: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["return_diff_ci_low"], d["return_diff_ci_high"] = d.pop("return_diff_ci")
        d["sharpe_diff_ci_low"], d["sharpe_diff_ci_high"] = d.pop("sharpe_diff_ci")
        return d


def circular_block_indices(n_obs: int, block: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Index matrix (n_samples x n_obs) of circular block resamples."""
    n_blocks = math.ceil(n_obs / block)
    starts = rng.integers(0, n_obs, size=(n_samples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_obs
    return idx.reshape(n_samples, -1)[:, :n_obs]


def _ann_return(r: np.ndarray, ppy: int) -> np.ndarray:
    growth = np.prod(1.0 + r, axis=-1)
    return growth ** (ppy / r.shape[-1]) - 1.0


def _sharpe(r: np.ndarray, rf: np.ndarray, ppy: int) -> np.ndarray:
    ex = r - rf
    sd = ex.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, ex.mean(axis=-1) / sd * math.sqrt(ppy), np.nan)


def _ci(x: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def _prob_positive(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    return float((finite > 0).mean()) if finite.size else float("nan")


def bootstrap_difference(strategy: pd.Series, benchmark: pd.Series, risk_free: pd.Series | None = None,
                         block: int = 12, n_samples: int = 5000, seed: int = 12345,
                         periods_per_year: int = 12, alpha: float = 0.05) -> BootstrapResult:
    df = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    df.columns = ["s", "b"]
    rf = (risk_free.reindex(df.index).fillna(0.0) if risk_free is not None
          else pd.Series(0.0, index=df.index)).to_numpy(dtype=float)
    s, b = df["s"].to_numpy(dtype=float), df["b"].to_numpy(dtype=float)
    T = len(df)
    if T < 2 * block:
        raise ValueError(f"Need at least {2 * block} observations for blocks of {block}; have {T}")
    rng = np.random.default_rng(seed)
    idx = circular_block_indices(T, block, n_samples, rng)
    ret_diff = _ann_return(s[idx], periods_per_year) - _ann_return(b[idx], periods_per_year)
    sr_diff = _sharpe(s[idx], rf[idx], periods_per_year) - _sharpe(b[idx], rf[idx], periods_per_year)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    obs_ret = float(_ann_return(s, periods_per_year) - _ann_return(b, periods_per_year))
    obs_sr = float(_sharpe(s, rf, periods_per_year) - _sharpe(b, rf, periods_per_year))
    return BootstrapResult(
        n_obs=T, block=block, n_samples=n_samples,
        return_diff=obs_ret,
        return_diff_ci=_ci(ret_diff, lo, hi),
        return_diff_prob_positive=_prob_positive(ret_diff),
        sharpe_diff=obs_sr,
        sharpe_diff_ci=_ci(sr_diff, lo, hi),
        sharpe_diff_prob_positive=_prob_positive(sr_diff),
    )


def explain_bootstrap(name: str, res: BootstrapResult) -> str:
    def span(ci, pct):
        return f"{ci[0]:+.2%} to {ci[1]:+.2%}" if pct else f"{ci[0]:+.2f} to {ci[1]:+.2f}"
    ret_clear = res.return_diff_ci[0] > 0 or res.return_diff_ci[1] < 0
    sr_clear = res.sharpe_diff_ci[0] > 0 or res.sharpe_diff_ci[1] < 0
    return (f"{name} vs SPY: annual return difference {res.return_diff:+.2%} "
            f"(95% range {span(res.return_diff_ci, True)}; "
            f"{'distinguishable from zero' if ret_clear else 'not distinguishable from zero'}); "
            f"Sharpe difference {res.sharpe_diff:+.2f} (95% range {span(res.sharpe_diff_ci, False)}; "
            f"{'distinguishable from zero' if sr_clear else 'not distinguishable from zero'}).")
