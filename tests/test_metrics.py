"""Metrics against hand-computed examples."""

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics as m
from src.backtest.engine import MarketData, run_backtest


def test_periods_per_year():
    assert m.periods_per_year(pd.bdate_range("2020-01-01", periods=30)) == 252
    assert m.periods_per_year(pd.date_range("2020-01-31", periods=30, freq="ME")) == 12


def test_cagr_doubling_in_two_years():
    eq = pd.Series([100.0, 200.0], index=pd.to_datetime(["2020-01-01", "2022-01-01"]))
    # 731 calendar days = 2.0014 years of 365.25 days -> about sqrt(2) - 1
    assert m.cagr(eq) == pytest.approx(2 ** (365.25 / 731) - 1)
    assert m.cagr(eq) == pytest.approx(math.sqrt(2) - 1, abs=5e-4)
    assert m.growth_to_cagr(100, 121, 2.0) == pytest.approx(0.10)


def test_volatility_and_sharpe_hand_computed():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.01, -0.01, 0.02, 0.0], index=idx)
    mean, sd = 0.005, np.std([0.01, -0.01, 0.02, 0.0], ddof=1)   # sd = 0.0129099
    assert m.annualized_volatility(r, 252) == pytest.approx(sd * math.sqrt(252))
    assert m.sharpe_ratio(r, 0.001, 252) == pytest.approx((mean - 0.001) / sd * math.sqrt(252))
    rf = pd.Series(0.001, index=idx)
    assert m.sharpe_ratio(r, rf, 252) == pytest.approx((mean - 0.001) / sd * math.sqrt(252))


def test_sortino_hand_computed():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.02, -0.01, 0.03, -0.02], index=idx)
    downside = math.sqrt((0.01 ** 2 + 0.02 ** 2) / 4)
    assert m.sortino_ratio(r, 0.0, 252) == pytest.approx(0.005 / downside * math.sqrt(252))


def test_drawdowns_hand_computed():
    idx = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-01-31", "2020-02-10", "2020-02-20"])
    eq = pd.Series([100, 120, 90, 110, 125, 100], index=idx, dtype=float)
    assert m.max_drawdown(eq) == pytest.approx(-0.25)              # 120 -> 90
    ep = m.drawdown_episodes(eq)
    assert len(ep) == 2
    first = ep.iloc[0]
    assert first["peak"] == idx[1] and first["trough"] == idx[2] and first["recovery"] == idx[4]
    assert first["days_to_recover"] == 30 and first["recovered"]
    second = ep.iloc[1]
    assert not second["recovered"] and second["depth"] == pytest.approx(-0.2)
    assert m.max_drawdown_duration(eq) == 30
    assert m.calmar_ratio(eq) == pytest.approx(m.cagr(eq) / 0.25)


def test_beta_te_ir():
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 50), index=idx)
    r = 2 * b
    assert m.beta(r, b) == pytest.approx(2.0)
    assert m.tracking_error(r, b, 252) == pytest.approx(b.std(ddof=1) * math.sqrt(252))
    r2 = b + 0.001
    assert m.tracking_error(r2, b, 252) == pytest.approx(0.0, abs=1e-12)
    r3 = b + pd.Series(np.where(np.arange(50) % 2, 0.002, 0.0), index=idx)
    d = r3 - b
    assert m.information_ratio(r3, b, 252) == pytest.approx(d.mean() * 252 / (d.std(ddof=1) * math.sqrt(252)))


def test_calendar_year_returns():
    idx = pd.to_datetime(["2020-03-02", "2020-12-31", "2021-12-31", "2022-06-30"])
    eq = pd.Series([100, 110, 99, 120], index=idx, dtype=float)
    y = m.calendar_year_returns(eq)
    assert y.loc[2020, "return"] == pytest.approx(0.10) and y.loc[2020, "partial"]
    assert y.loc[2021, "return"] == pytest.approx(-0.10) and not y.loc[2021, "partial"]
    assert y.loc[2022, "partial"]


def test_rolling_window_win_rate():
    idx = pd.date_range("2000-01-31", periods=84, freq="ME")
    strat = pd.Series(np.cumprod(np.full(84, 1.01)), index=idx)
    bench = pd.Series(np.cumprod(np.full(84, 1.005)), index=idx)
    rate, n = m.rolling_window_win_rate(strat, bench, 5)
    assert n == 24 and rate == 1.0
    assert m.rolling_window_win_rate(bench, strat, 5)[0] == 0.0
    assert m.rolling_window_win_rate(strat.iloc[:30], bench.iloc[:30], 5)[1] == 0
    ex = m.rolling_excess_return(strat, bench, 3)
    assert ex.iloc[0] == pytest.approx(1.01 ** 12 - 1.005 ** 12)


def test_turnover_and_cost_drag():
    idx = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"])
    eq = pd.Series(100_000.0, index=idx)
    trades = pd.DataFrame({"date": [idx[0], idx[1], idx[1]], "notional": [100_000, -50_000, 50_000]})
    years = (idx[-1] - idx[0]).days / 365.25
    assert m.annual_turnover(trades, eq) == pytest.approx(0.5 / years)   # initial buy excluded
    costs = pd.Series([50.0, 50.0, 0.0], index=idx)
    assert m.cost_drag(costs, eq) == pytest.approx(0.001 / years)


def test_summarize_end_to_end():
    dates = pd.bdate_range("2010-01-01", "2016-12-31")
    rng = np.random.default_rng(2)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(3e-4, 0.01, len(dates))),
                       "B": 100 * np.cumprod(1 + rng.normal(2e-4, 0.005, len(dates)))}, index=dates)
    md = MarketData(px, pd.Series(0.0, index=dates))
    bench = run_backtest(pd.DataFrame({"A": [1.0]}, index=[dates[0]]), md)
    months = px.resample("BME").last().index
    months = months[months.isin(dates)]
    strat = run_backtest(pd.DataFrame({"A": 0.5, "B": 0.5}, index=months), md)
    s = m.summarize(strat, bench, md.risk_free, traditional_ira_rate=0.29)
    for key in ("cagr_pretax", "sharpe", "sortino", "max_drawdown", "beta", "tracking_error",
                "information_ratio", "annual_turnover", "best_year", "pct_5y_windows_beating_benchmark",
                "cagr_traditional_ira"):
        assert key in s and not (isinstance(s[key], float) and math.isnan(s[key])), key
    assert 0.3 < s["beta"] < 0.7
    assert s["cagr_traditional_ira"] < s["cagr_pretax"]
    selfcmp = m.summarize(bench, bench, md.risk_free)
    assert selfcmp["beta"] == pytest.approx(1.0) and selfcmp["tracking_error"] == pytest.approx(0.0, abs=1e-12)
