"""Sub-period stability: fixed windows, expanding windows and named regimes.

None of the candidate strategies fits parameters to past data, so a
walk-forward analysis reduces to asking whether each strategy's results
against SPY hold up across separate periods rather than coming from one
lucky stretch.
"""

from __future__ import annotations

import pandas as pd

from src.backtest import metrics as m
from src.validation.common import slice_equity


def _window_row(name: str, label: str, start, end, strat: pd.Series, bench: pd.Series, rf: pd.Series,
                min_days: int = 300) -> dict:
    s = slice_equity(strat, start, end, min_days)
    b = slice_equity(bench, start, end, min_days)
    row = {"strategy": name, "window": label, "start": pd.Timestamp(start).date(), "end": pd.Timestamp(end).date()}
    if s is None or b is None or len(s) < 3:
        row["status"] = "not live"
        return row
    if s.index[0] != b.index[0]:
        raise ValueError(f"{name}: strategy and benchmark slices start on different dates")
    sr, br = s.pct_change().iloc[1:], b.pct_change().iloc[1:]
    full = (s.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7)
            and s.index[-1] >= pd.Timestamp(end) - pd.Timedelta(days=7))
    row.update({
        "status": "ok" if full else "partial",
        "live_from": s.index[0].date(), "live_to": s.index[-1].date(),
        "cagr": m.cagr(s), "benchmark_cagr": m.cagr(b),
        "sharpe": m.sharpe_ratio(sr, rf), "benchmark_sharpe": m.sharpe_ratio(br, rf),
        "max_drawdown": m.max_drawdown(s), "benchmark_max_drawdown": m.max_drawdown(b),
    })
    row["excess_cagr"] = row["cagr"] - row["benchmark_cagr"]
    # A tolerance keeps exact ties (floating-point noise) from counting as wins.
    row["beat_benchmark_return"] = bool(row["excess_cagr"] > 1e-9)
    row["smaller_drawdown"] = bool(row["max_drawdown"] > row["benchmark_max_drawdown"] + 1e-9)
    return row


def fixed_windows(runs: dict, windows: list[tuple[str, str]], risk_free: pd.Series) -> pd.DataFrame:
    """Each strategy vs. its same-period SPY benchmark in every fixed window."""
    rows = []
    for name, run in runs.items():
        for start, end in windows:
            rows.append(_window_row(name, f"{start[:4]}-{end[:4]}", start, end,
                                    run.pretax.equity, run.benchmark.equity, risk_free))
    return pd.DataFrame(rows)


def expanding_windows(runs: dict, step_years: int, risk_free: pd.Series) -> pd.DataFrame:
    """From each strategy's start, windows that grow by ``step_years`` until the data ends."""
    rows = []
    for name, run in runs.items():
        eq = run.pretax.equity
        start = eq.index[0]
        k = 1
        while True:
            end = start + pd.DateOffset(years=step_years * k)
            last = end >= eq.index[-1]
            end = min(end, eq.index[-1])
            rows.append(_window_row(name, f"first {((end - start).days / 365.25):.1f}y", start, end,
                                    eq, run.benchmark.equity, risk_free))
            if last:
                break
            k += 1
    return pd.DataFrame(rows)


def regime_table(runs: dict, regimes: dict[str, tuple[str, str]], risk_free: pd.Series,
                 heldout_start: pd.Timestamp, heldout_unlocked: bool) -> pd.DataFrame:
    """Results in each named regime. Regimes inside a locked held-out period are marked, not computed."""
    rows = []
    for name, run in runs.items():
        for label, (start, end) in regimes.items():
            if not heldout_unlocked and pd.Timestamp(end) >= heldout_start:
                rows.append({"strategy": name, "window": label, "start": pd.Timestamp(start).date(),
                             "end": pd.Timestamp(end).date(), "status": "held out (locked)"})
                continue
            # Regimes can be shorter than a year (e.g. 2020), so any coverage over 3 months counts.
            rows.append(_window_row(name, label, start, end, run.pretax.equity, run.benchmark.equity, risk_free,
                                    min_days=90))
    return pd.DataFrame(rows)


def consistency(table: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: how many evaluated windows beat SPY on return and on drawdown."""
    ok = table[table["status"].isin(["ok", "partial"])]
    if ok.empty:
        return pd.DataFrame(columns=["strategy", "windows", "beat_return", "smaller_drawdown"])
    g = ok.groupby("strategy")
    return pd.DataFrame({
        "windows": g.size(),
        "beat_return": g["beat_benchmark_return"].sum().astype(int),
        "smaller_drawdown": g["smaller_drawdown"].sum().astype(int),
        "median_excess_cagr": g["excess_cagr"].median(),
    }).reset_index()
