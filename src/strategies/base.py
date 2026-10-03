"""Common strategy interface.

Every strategy turns a daily total-return price frame into target weights on
monthly signal dates (the last trading day of each month).

No look-ahead by construction: ``Strategy.generate`` computes month-end
closes, then at each signal date ``d`` calls ``target(history, d)`` with
``history`` truncated to month-end closes dated on or before ``d``. A
strategy never receives a later price. Signals use the close on ``d``; the
engine executes them ``execution_lag`` trading days later.

``target`` returns a dict of weights (summing to 1), or ``None`` when the
strategy has no opinion (not enough history yet, or, for strategies that
rebalance only sometimes, no trade this month).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month in ``index``.

    The final month is included only if it is complete: the next business
    day after its last date falls in a new month.
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return index
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).last()
    out = pd.DatetimeIndex(last.values)
    final = out[-1]
    if (final + pd.offsets.BDay(1)).month == final.month:
        out = out[:-1]
    return out


def sma(history: pd.Series, months: int) -> float:
    """Simple moving average of the last ``months`` month-end closes (NaN if not enough)."""
    window = history.iloc[-months:]
    if len(window) < months or window.isna().any():
        return float("nan")
    return float(window.mean())


def trailing_return(history: pd.Series, months: int, skip: int = 0) -> float:
    """Total return over ``months`` months ending ``skip`` months before the latest close.

    With month-end closes P[-1] (latest), P[-2], ...:
    ``trailing_return(h, 12, 0) = P[-1] / P[-13] - 1``
    ``trailing_return(h, 12, 1) = P[-2] / P[-14] - 1`` (skips the most recent month)
    """
    need = months + skip + 1
    if len(history) < need:
        return float("nan")
    end = history.iloc[-1 - skip]
    start = history.iloc[-1 - skip - months]
    if not (np.isfinite(end) and np.isfinite(start)) or start <= 0:
        return float("nan")
    return float(end / start - 1)


def above_sma(history: pd.Series, months: int) -> bool | None:
    """True if the latest close is strictly above its SMA; None if not computable."""
    avg = sma(history, months)
    last = history.iloc[-1] if len(history) else float("nan")
    if not np.isfinite(avg) or not np.isfinite(last):
        return None
    return bool(last > avg)


class Strategy(ABC):
    """Base class. Subclasses implement ``symbols`` and ``target``."""

    name: str = "strategy"

    def __init__(self, name: str | None = None, **params):
        if name:
            self.name = name
        self.params = params

    @abstractmethod
    def symbols(self) -> list[str]:
        """Every symbol whose prices the strategy reads or may hold."""

    @abstractmethod
    def target(self, history: pd.DataFrame, date: pd.Timestamp) -> dict[str, float] | None:
        """Target weights given month-end closes up to and including ``date``."""

    def reset(self) -> None:
        """Clear any state carried between signal dates (override if stateful)."""

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights indexed by signal date, one column per symbol."""
        syms = self.symbols()
        missing = [s for s in syms if s not in prices.columns]
        if missing:
            raise ValueError(f"{self.name}: prices missing for {missing}")
        self.reset()
        dates = month_end_dates(prices.index)
        monthly = prices.loc[dates, syms]
        rows: dict[pd.Timestamp, dict[str, float]] = {}
        for i, d in enumerate(dates):
            history = monthly.iloc[: i + 1]          # never includes anything after d
            w = self.target(history, d)
            if w is None:
                continue
            total = sum(w.values())
            if abs(total - 1.0) > 1e-9 or any(v < 0 for v in w.values()):
                raise ValueError(f"{self.name}: invalid weights on {d.date()}: {w}")
            rows[d] = w
        # Build the matrix explicitly: DataFrame.from_dict can reorder rows whose
        # dicts have different keys.
        index = pd.DatetimeIndex(list(rows), name="signal_date")
        matrix = np.array([[rows[d].get(s, 0.0) for s in syms] for d in index], dtype=float).reshape(len(index), len(syms))
        unknown = {k for w in rows.values() for k in w} - set(syms)
        if unknown:
            raise ValueError(f"{self.name}: weights for symbols outside symbols(): {unknown}")
        return pd.DataFrame(matrix, index=index, columns=syms).sort_index()

    def describe(self) -> dict:
        return {"name": self.name, "class": type(self).__name__, **self.params}
