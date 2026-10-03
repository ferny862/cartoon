"""Performance metrics.

All functions take plain pandas objects. ``returns`` are simple per-period
returns, ``equity`` is a value series. Annualization uses the data frequency
(252 for daily, 12 for monthly). CAGR uses calendar time (365.25-day years).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer the sampling frequency from the median spacing between dates."""
    if len(index) < 3:
        raise ValueError("Need at least 3 dates to infer the frequency")
    spacing = pd.Series(index).diff().dt.days.median()
    if spacing <= 4:
        return 252
    if spacing <= 8:
        return 52
    if spacing <= 40:
        return 12
    if spacing <= 100:
        return 4
    return 1


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25


def cagr(equity: pd.Series, base: float | None = None) -> float:
    """Compound annual growth rate from ``base`` (default: first value) to the last value."""
    years = years_between(equity.index[0], equity.index[-1])
    start = equity.iloc[0] if base is None else base
    if years <= 0 or start <= 0:
        return float("nan")
    return (equity.iloc[-1] / start) ** (1 / years) - 1


def growth_to_cagr(start_value: float, end_value: float, years: float) -> float:
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return float("nan")
    return (end_value / start_value) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, ppy: int | None = None) -> float:
    ppy = ppy or periods_per_year(returns.index)
    return float(returns.std(ddof=1) * math.sqrt(ppy))


def _excess(returns: pd.Series, risk_free: pd.Series | float | None) -> pd.Series:
    if risk_free is None:
        return returns
    if isinstance(risk_free, pd.Series):
        return returns - risk_free.reindex(returns.index).fillna(0.0)
    return returns - risk_free


def sharpe_ratio(returns: pd.Series, risk_free: pd.Series | float | None = None, ppy: int | None = None) -> float:
    """Annualized mean excess return over its standard deviation."""
    ppy = ppy or periods_per_year(returns.index)
    ex = _excess(returns, risk_free)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(ppy)) if sd > 0 else float("nan")


def sortino_ratio(returns: pd.Series, risk_free: pd.Series | float | None = None, ppy: int | None = None) -> float:
    """Annualized mean excess return over downside deviation (target = risk-free)."""
    ppy = ppy or periods_per_year(returns.index)
    ex = _excess(returns, risk_free)
    downside = math.sqrt(float((np.minimum(ex, 0.0) ** 2).mean()))
    return float(ex.mean() / downside * math.sqrt(ppy)) if downside > 0 else float("nan")


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def drawdown_episodes(equity: pd.Series) -> pd.DataFrame:
    """Every peak-to-recovery episode.

    Columns: peak, trough, recovery (NaT if not recovered), depth (negative),
    days_to_trough, days_to_recover (peak to recovery, or to the last date if
    unrecovered), recovered.
    """
    rows = []
    peak_val = equity.iloc[0]
    peak_date = equity.index[0]
    trough_val, trough_date, in_dd = peak_val, peak_date, False
    for date, v in equity.items():
        if v >= peak_val:
            if in_dd:
                rows.append((peak_date, trough_date, date, trough_val / peak_val - 1, True))
                in_dd = False
            peak_val, peak_date = v, date
            trough_val, trough_date = v, date
        else:
            in_dd = True
            if v < trough_val:
                trough_val, trough_date = v, date
    if in_dd:
        rows.append((peak_date, trough_date, pd.NaT, trough_val / peak_val - 1, False))
    df = pd.DataFrame(rows, columns=["peak", "trough", "recovery", "depth", "recovered"])
    end = equity.index[-1]
    df["days_to_trough"] = (df["trough"] - df["peak"]).dt.days
    df["days_to_recover"] = (df["recovery"].fillna(end) - df["peak"]).dt.days
    return df


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest peak-to-recovery time in calendar days (to the end if never recovered)."""
    ep = drawdown_episodes(equity)
    return int(ep["days_to_recover"].max()) if len(ep) else 0


def calmar_ratio(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    return cagr(equity) / abs(mdd) if mdd < 0 else float("nan")


def _aligned(r: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([r, b], axis=1, join="inner").dropna()
    return df.iloc[:, 0], df.iloc[:, 1]


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    r, b = _aligned(returns, benchmark)
    var = b.var(ddof=1)
    return float(r.cov(b) / var) if var > 0 else float("nan")


def tracking_error(returns: pd.Series, benchmark: pd.Series, ppy: int | None = None) -> float:
    r, b = _aligned(returns, benchmark)
    ppy = ppy or periods_per_year(r.index)
    return float((r - b).std(ddof=1) * math.sqrt(ppy))


def information_ratio(returns: pd.Series, benchmark: pd.Series, ppy: int | None = None) -> float:
    r, b = _aligned(returns, benchmark)
    ppy = ppy or periods_per_year(r.index)
    te = (r - b).std(ddof=1) * math.sqrt(ppy)
    return float((r - b).mean() * ppy / te) if te > 0 else float("nan")


def calendar_year_returns(equity: pd.Series, base: float | None = None) -> pd.DataFrame:
    """Return per calendar year. The first and last years are flagged partial
    unless they span the whole year."""
    year_end = equity.groupby(equity.index.year).last()
    start_val = equity.iloc[0] if base is None else base
    prev = pd.Series([start_val] + year_end.iloc[:-1].tolist(), index=year_end.index)
    rets = year_end / prev - 1
    first_date, last_date = equity.index[0], equity.index[-1]
    partial = pd.Series(False, index=year_end.index)
    if first_date > pd.Timestamp(year=first_date.year, month=1, day=7):
        partial.iloc[0] = True
    if last_date < pd.Timestamp(year=last_date.year, month=12, day=24):
        partial.iloc[-1] = True
    return pd.DataFrame({"return": rets, "partial": partial})


def rolling_window_win_rate(equity: pd.Series, benchmark_equity: pd.Series, years: int = 5) -> tuple[float, int]:
    """Share of rolling ``years``-year windows (month-end steps) in which the
    strategy's total return beat the benchmark's. Returns (rate, n_windows)."""
    m = pd.concat([equity, benchmark_equity], axis=1, join="inner").dropna().resample("ME").last()
    n = years * 12
    if len(m) <= n:
        return float("nan"), 0
    roll = m / m.shift(n) - 1
    roll = roll.dropna()
    wins = (roll.iloc[:, 0] > roll.iloc[:, 1]).mean()
    return float(wins), int(len(roll))


def rolling_excess_return(equity: pd.Series, benchmark_equity: pd.Series, years: int = 3) -> pd.Series:
    """Annualized rolling excess return over the benchmark, at month ends."""
    m = pd.concat([equity, benchmark_equity], axis=1, join="inner").dropna().resample("ME").last()
    n = years * 12
    ann = (m / m.shift(n)) ** (1 / years) - 1
    return (ann.iloc[:, 0] - ann.iloc[:, 1]).dropna().rename("rolling_excess")


def annual_turnover(trades: pd.DataFrame, equity: pd.Series) -> float:
    """One-way turnover per year: half of all traded notional (excluding the
    initial purchase) divided by average portfolio value, per year."""
    years = years_between(equity.index[0], equity.index[-1])
    if years <= 0 or trades.empty:
        return 0.0
    later = trades[trades["date"] > equity.index[0]]
    return float(later["notional"].abs().sum() / 2 / equity.mean() / years)


def cost_drag(costs: pd.Series, equity: pd.Series) -> float:
    """Trading costs per year as a fraction of average portfolio value."""
    years = years_between(equity.index[0], equity.index[-1])
    return float(costs.sum() / equity.mean() / years) if years > 0 else float("nan")


def summarize(result, benchmark=None, risk_free: pd.Series | None = None, taxable=None,
              traditional_ira_rate: float | None = None) -> dict:
    """Headline metrics for one strategy.

    ``result`` is a pre-tax BacktestResult (also the Roth IRA result);
    ``benchmark`` a pre-tax BacktestResult for SPY; ``taxable`` the same
    strategy run in taxable mode. Relative metrics use common dates only.
    """
    eq = result.equity
    rets = result.returns
    ppy = periods_per_year(eq.index)
    years = years_between(eq.index[0], eq.index[-1])
    annual = calendar_year_returns(eq, base=result.initial_capital)
    full = annual.loc[~annual["partial"], "return"]
    out = {
        "start": eq.index[0].date(),
        "end": eq.index[-1].date(),
        "years": years,
        "cagr_pretax": growth_to_cagr(result.initial_capital, eq.iloc[-1], years),
        "volatility": annualized_volatility(rets, ppy),
        "sharpe": sharpe_ratio(rets, risk_free, ppy),
        "sortino": sortino_ratio(rets, risk_free, ppy),
        "max_drawdown": max_drawdown(eq),
        "max_drawdown_days": max_drawdown_duration(eq),
        "calmar": calmar_ratio(eq),
        "annual_turnover": annual_turnover(result.trades, eq),
        "cost_drag": cost_drag(result.costs, eq),
        "best_year": float(full.max()) if len(full) else float("nan"),
        "worst_year": float(full.min()) if len(full) else float("nan"),
    }
    if benchmark is not None:
        b_rets = benchmark.returns
        out["beta"] = beta(rets, b_rets)
        out["tracking_error"] = tracking_error(rets, b_rets, ppy)
        out["information_ratio"] = information_ratio(rets, b_rets, ppy)
        out["pct_5y_windows_beating_benchmark"], out["n_5y_windows"] = rolling_window_win_rate(eq, benchmark.equity, 5)
    if traditional_ira_rate is not None:
        out["cagr_traditional_ira"] = growth_to_cagr(result.initial_capital, eq.iloc[-1] * (1 - traditional_ira_rate), years)
    if taxable is not None:
        t_years = years_between(taxable.equity.index[0], taxable.equity.index[-1])
        out["cagr_aftertax_holding"] = growth_to_cagr(taxable.initial_capital, taxable.after_tax_value_holding, t_years)
        out["cagr_aftertax_liquidated"] = growth_to_cagr(taxable.initial_capital, taxable.after_tax_value_liquidated, t_years)
        out["wash_sale_flags"] = int(len(taxable.wash_sales)) if taxable.wash_sales is not None else 0
    return out
