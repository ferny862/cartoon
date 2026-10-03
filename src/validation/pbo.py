"""Probability of Backtest Overfitting via combinatorially symmetric
cross-validation (CSCV; Bailey, Borwein, Lopez de Prado and Zhu, 2017).

Given a T x N matrix of returns for N candidate configurations:

1. Split the T rows into S equal, contiguous blocks (S even).
2. For every way of choosing S/2 blocks as "in-sample" (the rest are
   "out-of-sample"), pick the candidate with the best in-sample Sharpe ratio
   and find its relative rank w among all candidates out of sample.
3. The logit l = ln(w / (1 - w)) is < 0 when the in-sample winner lands in
   the bottom half out of sample. PBO is the share of splits where that
   happens; a winner exactly at the median (l = 0, possible when N is odd)
   counts as half.

PBO near 0: choosing the in-sample best tends to pick a genuinely better
strategy. Near 0.5 or above: selection is no better than chance, so the
in-sample winner's backtest is not informative.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PBOResult:
    pbo: float
    n_strategies: int
    n_obs: int
    n_partitions: int
    n_combinations: int
    logits: np.ndarray = field(repr=False)
    is_sharpe_selected: np.ndarray = field(repr=False)
    oos_sharpe_selected: np.ndarray = field(repr=False)
    prob_oos_loss: float = float("nan")          # P(selected strategy's OOS excess-return Sharpe < 0)
    degradation_slope: float = float("nan")      # OOS Sharpe regressed on IS Sharpe of the selection
    selection_frequency: dict = field(default_factory=dict)   # how often each candidate wins in-sample
    start: object = None
    end: object = None

    def summary(self) -> dict:
        return {"pbo": self.pbo, "n_strategies": self.n_strategies, "n_obs": self.n_obs,
                "n_partitions": self.n_partitions, "n_combinations": self.n_combinations,
                "prob_oos_loss": self.prob_oos_loss, "degradation_slope": self.degradation_slope,
                "median_logit": float(np.median(self.logits)), "start": self.start, "end": self.end,
                **{f"selected_{k}": v for k, v in self.selection_frequency.items()}}


def _sharpe_from_moments(s: np.ndarray, q: np.ndarray, c: np.ndarray) -> np.ndarray:
    mean = s / c
    var = (q - s * s / c) / (c - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 0, mean / np.sqrt(np.maximum(var, 0)), np.nan)


def pbo_cscv(returns: pd.DataFrame, n_partitions: int = 16) -> PBOResult:
    """CSCV on a returns matrix (rows = periods, columns = candidates).

    Rows with any missing value are dropped so every candidate is compared
    over identical periods.
    """
    M = returns.dropna()
    T, N = M.shape
    S = int(n_partitions)
    if S % 2 or S < 2:
        raise ValueError("n_partitions must be an even number >= 2")
    if N < 2:
        raise ValueError("PBO needs at least 2 candidates")
    if T < 2 * S:
        raise ValueError(f"Need at least {2 * S} observations for {S} partitions; have {T}")
    X = M.to_numpy(dtype=float)
    blocks = np.array_split(np.arange(T), S)
    sums = np.array([X[b].sum(0) for b in blocks])            # S x N
    sqs = np.array([(X[b] ** 2).sum(0) for b in blocks])       # S x N
    cnt = np.array([len(b) for b in blocks], dtype=float)      # S

    combos = np.array(list(itertools.combinations(range(S), S // 2)))
    C = np.zeros((len(combos), S))
    C[np.arange(len(combos))[:, None], combos] = 1.0
    is_s, is_q, is_c = C @ sums, C @ sqs, (C @ cnt)[:, None]
    oos_s, oos_q, oos_c = sums.sum(0) - is_s, sqs.sum(0) - is_q, cnt.sum() - is_c
    is_sr = _sharpe_from_moments(is_s, is_q, is_c)
    oos_sr = _sharpe_from_moments(oos_s, oos_q, oos_c)

    best = np.nanargmax(is_sr, axis=1)
    rows = np.arange(len(combos))
    sel_is, sel_oos = is_sr[rows, best], oos_sr[rows, best]
    # Average rank (1 = worst) of the selected candidate among all N out of sample.
    less = (oos_sr < sel_oos[:, None]).sum(1)
    ties = (oos_sr == sel_oos[:, None]).sum(1)
    w = (less + (ties + 1) / 2) / (N + 1)
    logits = np.log(w / (1 - w))
    slope = float(np.polyfit(sel_is, sel_oos, 1)[0]) if np.ptp(sel_is) > 0 else float("nan")
    freq = pd.Series(best).map(dict(enumerate(M.columns))).value_counts(normalize=True)
    return PBOResult(
        # A logit of exactly 0 (selected candidate at the median rank, possible
        # when N is odd) counts as half, so pure noise gives 0.5 for any N.
        pbo=float((logits < 0).mean() + 0.5 * np.isclose(logits, 0.0).mean()),
        n_strategies=N, n_obs=T, n_partitions=S, n_combinations=len(combos),
        logits=logits, is_sharpe_selected=sel_is, oos_sharpe_selected=sel_oos,
        prob_oos_loss=float((sel_oos < 0).mean()),
        degradation_slope=slope,
        selection_frequency={str(k): float(v) for k, v in freq.items()},
        start=M.index[0], end=M.index[-1],
    )


def explain_pbo(label: str, res: PBOResult) -> str:
    if res.pbo < 0.2:
        reading = "picking the in-sample winner has usually picked a strategy that also did well out of sample"
    elif res.pbo < 0.5:
        reading = "picking the in-sample winner helps somewhat, but often fails"
    else:
        reading = "picking the in-sample winner is no better than a coin flip, so the best backtest is not informative"
    top = max(res.selection_frequency, key=res.selection_frequency.get) if res.selection_frequency else "n/a"
    return (f"{label}: across {res.n_combinations:,} ways of splitting {res.n_obs} months into halves, the "
            f"in-sample best of {res.n_strategies} candidates ranked in the bottom half out of sample "
            f"{res.pbo:.0%} of the time (PBO). In plain terms, {reading}. The most frequent in-sample winner "
            f"was {top}; the selected strategy did worse than Treasury bills out of sample in "
            f"{res.prob_oos_loss:.0%} of splits.")
