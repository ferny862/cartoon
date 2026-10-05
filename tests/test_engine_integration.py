"""Engine cross-checks on longer synthetic histories."""

import time

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.engine import MarketData, TaxSettings, run_backtest
from src.backtest.taxes import TaxRates

RATES = TaxRates(0.24, 0.15, 0.15, 0.24, 0.05)


def synthetic(n_assets=10, start="2000-01-03", end="2024-12-31", seed=7):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    syms = [f"S{i}" for i in range(n_assets)]
    rets = rng.normal(3e-4, 0.012, (len(dates), n_assets))
    px = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=syms)
    y = pd.DataFrame(0.0, index=dates, columns=syms)
    quarter_ends = px.resample("BQE").last().index
    y.loc[y.index.isin(quarter_ends)] = 0.005           # 0.5% quarterly dividends
    return px, y


def month_ends(index):
    me = pd.Series(index, index=index).groupby([index.year, index.month]).last()
    return pd.DatetimeIndex(me.values)


def test_matches_independent_vectorized_calculation():
    """Monthly-rebalanced equal weight, zero cost, lag 0: compare with a direct formula."""
    px, _ = synthetic(n_assets=3, end="2005-12-30")
    sig = month_ends(px.index)
    w = pd.DataFrame(1 / 3, index=sig, columns=px.columns)
    # min_trade_value=0: otherwise drifts under $1 are (correctly) left untraded.
    res = run_backtest(w, MarketData(px, pd.Series(0.0, index=px.index)), execution_lag=0, min_trade_value=0.0)

    # Independent: between rebalances each sleeve grows with its own price.
    value = 100_000.0
    eq = {}
    for a, b in zip(sig[:-1], sig[1:]):
        growth = (px.loc[b] / px.loc[a]).mean()
        value *= growth
        eq[b] = value
    expected = pd.Series(eq)
    pd.testing.assert_series_equal(res.equity.reindex(expected.index), expected, check_names=False, rtol=1e-10)


def test_taxable_run_is_consistent_and_fast():
    px, y = synthetic()
    sig = month_ends(px.index)
    rng = np.random.default_rng(3)
    # Rotate into 3 random assets each month: high turnover, many lots.
    picks = np.zeros((len(sig), px.shape[1]))
    for k in range(len(sig)):
        picks[k, rng.choice(px.shape[1], 3, replace=False)] = 1 / 3
    w = pd.DataFrame(picks, index=sig, columns=px.columns)
    chars = {s: "qualified" for s in px.columns}
    md = MarketData(px, pd.Series(0.0, index=px.index), div_yield=y, tax_character=chars)

    t0 = time.perf_counter()
    taxable = run_backtest(w, md, CostModel(5), tax=TaxSettings(RATES))
    elapsed = time.perf_counter() - t0
    pre = run_backtest(w, md, CostModel(5))

    assert elapsed < 30, f"taxable 25-year run took {elapsed:.1f}s"
    assert taxable.equity.iloc[-1] < pre.equity.iloc[-1]
    # Every closed year's bill is paid in the following April (except bills still pending at the end).
    closed = taxable.tax_years[~taxable.tax_years["open_at_end"]]
    assert taxable.taxes_paid.sum() == pytest.approx(closed["total"].sum(), rel=1e-9)
    # Monthly rotation makes almost every gain short-term, and many dividends fail the holding test.
    r = taxable.realized
    assert (r["term"] == "short").mean() > 0.8
    assert closed["income_nonqualified"].sum() > 0
    assert len(taxable.wash_sales) > 0
    # Pre-tax total return with dividends is in the TR prices: no dividend cash flows appear.
    assert pre.taxes_paid is None
    assert taxable.after_tax_value_liquidated <= taxable.after_tax_value_holding + 1e-6 or \
        taxable.realized["gain"].sum() < 0
