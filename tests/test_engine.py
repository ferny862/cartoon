"""Backtest engine on small hand-computable examples."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.engine import CASH, MarketData, TaxSettings, execution_dates, run_backtest
from src.backtest.taxes import TaxRates

RATES = TaxRates(short_term=0.24, long_term=0.15, qualified_dividend=0.15, ordinary=0.24, state=0.05)


def market(prices: dict, dates, **kw) -> MarketData:
    idx = pd.DatetimeIndex(dates)
    df = pd.DataFrame(prices, index=idx, dtype=float)
    return MarketData(prices=df, risk_free=pd.Series(0.0, index=idx), **kw)


def W(rows: dict, columns) -> pd.DataFrame:
    return pd.DataFrame(rows, index=columns).T.astype(float)


D5 = pd.bdate_range("2020-01-06", periods=5)


def test_execution_dates_lag_and_drop():
    m = execution_dates(pd.DatetimeIndex([D5[0], D5[4]]), D5, lag=1)
    assert list(m.index) == [D5[0]] and m.iloc[0] == D5[1]
    with pytest.raises(ValueError):
        execution_dates(pd.DatetimeIndex([pd.Timestamp("2020-01-04")]), D5, 1)


def test_buy_and_hold_tracks_total_return():
    mk = market({"A": [100, 101, 103, 102, 110]}, D5)
    res = run_backtest(W({D5[0]: [1.0]}, ["A"]), mk, execution_lag=1)
    assert res.start == D5[1]                                    # fill at the next close
    assert res.equity.iloc[0] == pytest.approx(100_000)
    assert res.equity.iloc[-1] == pytest.approx(100_000 * 110 / 101)


def test_lag_zero_fills_on_signal_close():
    mk = market({"A": [100, 101, 103, 102, 110]}, D5)
    res = run_backtest(W({D5[0]: [1.0]}, ["A"]), mk, execution_lag=0)
    assert res.equity.iloc[-1] == pytest.approx(100_000 * 110 / 100)


def test_initial_cost_fixed_point():
    mk = market({"A": [100.0] * 5}, D5)
    res = run_backtest(W({D5[0]: [1.0]}, ["A"]), mk, costs=CostModel(spread_slippage_bps=5))
    # cost c = 0.0005 * (100,000 - c)  ->  c = 50 / 1.0005
    c = 0.0005 * 100_000 / 1.0005
    assert res.costs.iloc[0] == pytest.approx(c)
    assert res.equity.iloc[0] == pytest.approx(100_000 - c)
    assert res.weights[CASH].abs().max() < 1e-9


def test_monthly_rebalance_hand_computed():
    mk = market({"A": [100, 100, 110, 121, 121], "B": [100, 100, 90, 99, 99]}, D5)
    w = W({D5[0]: [0.5, 0.5], D5[1]: [0.5, 0.5]}, ["A", "B"])
    res = run_backtest(w, mk, execution_lag=1)
    # Day 1 buys 500 A + 500 B. Day 2: A 55,000, B 45,000 -> rebalance to 50,000 each.
    assert res.equity.loc[D5[2]] == pytest.approx(100_000)
    t = res.trades[res.trades.date == D5[2]].set_index("symbol")
    assert t.loc["A", "notional"] == pytest.approx(-5000) and t.loc["B", "notional"] == pytest.approx(5000)
    # Day 3: both rise 10% from the rebalanced 50/50 -> 110,000.
    assert res.equity.loc[D5[3]] == pytest.approx(110_000)
    assert res.weights.loc[D5[2], "A"] == pytest.approx(0.5)


def test_switch_with_costs_hand_computed():
    mk = market({"A": [100.0] * 5, "B": [50.0] * 5}, D5)
    w = W({D5[0]: [1.0, 0.0], D5[2]: [0.0, 1.0]}, ["A", "B"])
    res = run_backtest(w, mk, costs=CostModel(spread_slippage_bps=10))
    v1 = 100_000 / 1.001                       # after the initial buy
    # Switch: sell v1 of A and buy (v1 - c) of B with c = 0.001 * (v1 + v1 - c)
    c = 0.002 * v1 / 1.001
    assert res.equity.iloc[-1] == pytest.approx(v1 - c)
    assert res.costs.sum() == pytest.approx(100_000 - (v1 - c))


def test_partial_cash_weight_and_validation():
    mk = market({"A": [100.0] * 5}, D5)
    res = run_backtest(W({D5[0]: [0.6]}, ["A"]), mk)
    assert res.weights[CASH].iloc[-1] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="sum above 1"):
        run_backtest(W({D5[0]: [1.2]}, ["A"]), mk)
    with pytest.raises(ValueError, match="Negative"):
        run_backtest(W({D5[0]: [-0.1]}, ["A"]), mk)
    with pytest.raises(ValueError, match="No prices"):
        run_backtest(W({D5[0]: [1.0]}, ["Z"]), mk)


def test_target_before_inception_raises():
    mk = market({"A": [100.0] * 5, "B": [np.nan, np.nan, 50, 50, 50]}, D5)
    with pytest.raises(ValueError, match="no price"):
        run_backtest(W({D5[0]: [0.5, 0.5]}, ["A", "B"]), mk)
    ok = run_backtest(W({D5[0]: [1.0, 0.0], D5[1]: [0.5, 0.5]}, ["A", "B"]), mk)
    assert ok.equity.iloc[-1] == pytest.approx(100_000)


def test_held_position_losing_price_raises():
    mk = market({"A": [100, 100, np.nan, 100, 100]}, D5)
    with pytest.raises(ValueError, match="held position"):
        run_backtest(W({D5[0]: [1.0]}, ["A"]), mk)


def test_whole_shares_leave_cash():
    mk = market({"A": [300.0] * 5}, D5)
    res = run_backtest(W({D5[0]: [1.0]}, ["A"]), mk, initial_capital=1000, allow_fractional=False)
    assert res.trades.iloc[0]["units"] == pytest.approx(3)
    assert res.weights[CASH].iloc[-1] == pytest.approx(0.1)


def test_engine_is_causal():
    dates = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, .01, 60)),
                       "B": 100 * np.cumprod(1 + rng.normal(0, .01, 60))}, index=dates)
    w = W({dates[0]: [1, 0], dates[20]: [0.3, 0.7], dates[40]: [0, 1]}, ["A", "B"])
    base = run_backtest(w, MarketData(px, pd.Series(0.0, index=dates)), CostModel(5))
    shocked = px.copy()
    shocked.iloc[30:] *= 3.0
    alt = run_backtest(w, MarketData(shocked, pd.Series(0.0, index=dates)), CostModel(5))
    cut = dates[29]
    pd.testing.assert_series_equal(base.equity[:cut], alt.equity[:cut])


# ---------------------------------------------------------------- taxable mode

def test_taxable_liquidation_long_term_gain():
    dates = pd.bdate_range("2020-01-01", "2021-06-30")
    px = np.linspace(100, 150, len(dates))
    mk = market({"A": px}, dates, tax_character={"A": "qualified"})
    res = run_backtest(W({dates[0]: [1.0]}, ["A"]), mk, tax=TaxSettings(RATES))
    start_px, end_px = px[1], px[-1]
    gain = 100_000 * end_px / start_px - 100_000
    assert res.taxes_paid.sum() == 0                              # nothing realized before the end
    assert res.after_tax_value_holding == pytest.approx(res.equity.iloc[-1])
    assert res.after_tax_value_liquidated == pytest.approx(100_000 + gain - gain * (0.15 + 0.05))


def test_taxable_short_term_gain_paid_next_april():
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    px = pd.Series(100.0, index=dates)
    px.loc["2020-03-02":] = 120.0
    mk = market({"A": px.values, "B": [50.0] * len(dates)}, dates, tax_character={"A": "qualified", "B": "qualified"})
    w = W({dates[0]: [1.0, 0.0], pd.Timestamp("2020-06-01"): [0.0, 1.0]}, ["A", "B"])
    res = run_backtest(w, mk, tax=TaxSettings(RATES))
    st_gain = 20_000.0
    expected_tax = st_gain * (0.24 + 0.05)                        # 5,800
    years = res.tax_years.set_index("year")
    assert years.loc[2020, "taxable_st"] == pytest.approx(st_gain)
    assert years.loc[2020, "total"] == pytest.approx(expected_tax)
    paid = res.taxes_paid[res.taxes_paid > 0]
    assert paid.index[0] == pd.Timestamp("2021-04-15") and paid.iloc[0] == pytest.approx(expected_tax)
    assert res.equity.iloc[-1] == pytest.approx(120_000 - expected_tax)
    assert res.realized.iloc[0]["term"] == "short"


def test_taxable_dividends_taxed_and_reinvested():
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    tr = pd.Series(100.0, index=dates)
    ex = pd.Timestamp("2020-06-15")
    tr.loc[ex:] = 100.0                                           # total return flat across the ex-date
    y = pd.DataFrame({"A": 0.0}, index=dates)
    y.loc[ex, "A"] = 0.02                                         # 2% dividend reinvested
    mk = market({"A": tr.values}, dates, div_yield=y, tax_character={"A": "qualified"})
    res = run_backtest(W({dates[0]: [1.0]}, ["A"]), mk, tax=TaxSettings(RATES))
    yr = res.tax_years.set_index("year")
    assert yr.loc[2020, "income_qualified"] == pytest.approx(2000)
    assert yr.loc[2020, "total"] == pytest.approx(2000 * 0.20)
    assert res.equity.iloc[-1] == pytest.approx(100_000 - 400)     # paid by selling in April 2021
    # Pre-tax run of the same inputs ignores dividend taxes entirely.
    pre = run_backtest(W({dates[0]: [1.0]}, ["A"]), mk)
    assert pre.equity.iloc[-1] == pytest.approx(100_000)


def test_wash_sale_flag_from_engine():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    px = pd.Series(100.0, index=dates)
    px.loc["2020-02-03":] = 80.0
    mk = market({"A": px.values, "B": [10.0] * len(dates)}, dates, tax_character={"A": "qualified", "B": "qualified"})
    w = W({dates[0]: [1, 0], pd.Timestamp("2020-02-03"): [0, 1], pd.Timestamp("2020-02-14"): [1, 0]}, ["A", "B"])
    res = run_backtest(w, mk, tax=TaxSettings(RATES))
    assert len(res.wash_sales) == 1 and res.wash_sales.iloc[0]["symbol"] == "A"
