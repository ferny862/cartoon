"""Shared helpers: monthly returns and window slicing."""

from __future__ import annotations

import pandas as pd


def monthly_returns(equity: pd.Series) -> pd.Series:
    """Calendar-month simple returns from a daily equity curve.

    The first month is measured from the first value, so a partial first
    month is included.
    """
    m = equity.resample("ME").last()
    first = pd.Series([equity.iloc[0]], index=[m.index[0] - pd.offsets.MonthEnd(1)])
    return pd.concat([first, m]).pct_change().iloc[1:].rename(equity.name)


def monthly_risk_free(risk_free_daily: pd.Series) -> pd.Series:
    """Compound daily risk-free returns into calendar-month returns."""
    return (1.0 + risk_free_daily.fillna(0.0)).resample("ME").prod() - 1.0


def monthly_excess(equity: pd.Series, risk_free_daily: pd.Series) -> pd.Series:
    r = monthly_returns(equity)
    rf = monthly_risk_free(risk_free_daily).reindex(r.index).fillna(0.0)
    return (r - rf).rename(equity.name)


def slice_equity(equity: pd.Series, start, end, min_days: int = 300) -> pd.Series | None:
    """Equity within [start, end], rebased on the last value before ``start``.

    If the curve starts inside the window, the slice begins at the curve's
    first value (callers compare against a benchmark that starts the same
    day). Returns None if fewer than ``min_days`` calendar days of the
    window are covered.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if equity.index[0] > end or equity.index[-1] < start:
        return None
    before = equity[equity.index < start]
    inside = equity[(equity.index >= start) & (equity.index <= end)]
    if inside.empty:
        return None
    if len(before):
        inside = pd.concat([before.iloc[-1:], inside])
    if (inside.index[-1] - inside.index[0]).days < min_days:
        return None
    return inside
