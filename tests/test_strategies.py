"""Signal logic for every strategy on small synthetic series."""

import numpy as np
import pandas as pd
import pytest

from src.strategies import (
    GEM, GTAA, BuyAndHold, FactorBlend, SectorMomentum, TrendFilter, build_strategy, strategy_names,
)
from src.strategies.base import above_sma, month_end_dates, sma, trailing_return


def daily_from_monthly(monthly: dict[str, list[float]], start="2000-01-01") -> pd.DataFrame:
    """Daily business-day prices whose month-end closes equal the given values.

    Every day in a month carries that month's value; NaN months stay NaN.
    """
    n = len(next(iter(monthly.values())))
    days = pd.bdate_range(start, periods=n * 23 + 40)
    me = month_end_dates(days)[:n]
    days = days[days <= me[-1]]
    month_pos = (days.year - me[0].year) * 12 + (days.month - me[0].month)
    return pd.DataFrame({sym: np.asarray(vals, dtype=float)[month_pos] for sym, vals in monthly.items()},
                        index=days)


def ones(n, v=100.0):
    return [v] * n


# ------------------------------------------------------------------ helpers

def test_month_end_dates_drop_incomplete_final_month():
    days = pd.bdate_range("2020-01-01", "2020-03-17")
    me = month_end_dates(days)
    assert list(me) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]
    full = month_end_dates(pd.bdate_range("2020-01-01", "2020-03-31"))
    assert full[-1] == pd.Timestamp("2020-03-31")


def test_month_end_uses_last_trading_day_with_holidays():
    days = pd.bdate_range("2020-12-01", "2021-01-29").drop(pd.Timestamp("2020-12-31"))
    assert pd.Timestamp("2020-12-30") in month_end_dates(days)


def test_trailing_return_and_skip_month():
    h = pd.Series([100.0] * 12 + [110.0, 121.0])     # 14 month-end closes; P[-1]=121, P[-2]=110
    assert trailing_return(h, 12) == pytest.approx(121 / 100 - 1)
    assert trailing_return(h, 12, skip=1) == pytest.approx(110 / 100 - 1)   # P[-2] / P[-14]
    assert np.isnan(trailing_return(h.iloc[1:], 12, skip=1))               # needs 14 closes


def test_sma_strictly_above():
    h = pd.Series([10.0] * 10)
    assert sma(h, 10) == 10 and above_sma(h, 10) is False                 # equal is not above
    assert above_sma(pd.Series([10.0] * 9 + [11.0]), 10) is True
    assert above_sma(pd.Series([10.0] * 9), 10) is None


# ------------------------------------------------------------------ strategies

def test_buy_and_hold_single_row_at_first_valid_month():
    px = daily_from_monthly({"SPY": [np.nan, 100, 101, 102]})
    w = BuyAndHold("SPY").generate(px)
    assert len(w) == 1 and w.iloc[0]["SPY"] == 1.0


def test_trend_filter_switches_on_sma_cross():
    spy = [100 + i for i in range(12)] + [90.0, 120.0]
    px = daily_from_monthly({"SPY": spy, "BIL": ones(14)})
    w = TrendFilter("SPY", "BIL", 10).generate(px)
    assert w.index[0] == month_end_dates(px.index)[9]                       # first month with 10 closes
    assert (w["SPY"].iloc[:-2] == 1).all()
    assert w.iloc[-2]["BIL"] == 1.0                                          # 90 is below its SMA
    assert w.iloc[-1]["SPY"] == 1.0


@pytest.mark.parametrize("us_end,intl_end,cash_end,expected", [
    (120, 110, 101, "SPY"),     # US beats T-bills and international
    (110, 125, 101, "EFA"),     # US beats T-bills, international stronger
    (100.5, 125, 101, "AGG"),   # US loses to T-bills -> bonds even though intl is strong
])
def test_gem_rules(us_end, intl_end, cash_end, expected):
    def path(end):
        return [100.0] * 12 + [end]
    px = daily_from_monthly({"SPY": path(us_end), "EFA": path(intl_end), "AGG": ones(13), "BIL": path(cash_end)})
    w = GEM().generate(px)
    assert len(w) == 1 and w.iloc[0][expected] == 1.0


def test_gtaa_sleeves_go_to_cash_individually():
    up = [100.0] * 9 + [110.0]
    down = [100.0] * 9 + [90.0]
    px = daily_from_monthly({"SPY": up, "EFA": down, "IEF": up, "VNQ": down, "DBC": up, "BIL": ones(10)})
    w = GTAA().generate(px).iloc[-1]
    assert w["SPY"] == pytest.approx(0.2) and w["IEF"] == pytest.approx(0.2) and w["DBC"] == pytest.approx(0.2)
    assert w["EFA"] == 0 and w["VNQ"] == 0 and w["BIL"] == pytest.approx(0.4)


def test_gtaa_waits_for_every_asset_history():
    px = daily_from_monthly({"SPY": ones(12), "EFA": ones(12), "IEF": ones(12), "VNQ": ones(12),
                             "DBC": [np.nan] * 5 + ones(7), "BIL": ones(12)})
    assert GTAA().generate(px).empty                                         # DBC has only 7 closes


def test_factor_blend_rebalances_in_december_and_on_drift():
    n = 14  # Jan 2000 .. Feb 2001
    stable = ones(n)
    px = daily_from_monthly({"QUAL": stable, "MTUM": stable, "VLUE": stable, "USMV": stable})
    w = FactorBlend().generate(px)
    assert [d.month for d in w.index] == [1, 12]                             # initial + December only

    # MTUM doubles in April 2000: its weight drifts from 25% to 40% -> rebalance that month.
    mtum = [100.0] * 3 + [200.0] * (n - 3)
    px2 = daily_from_monthly({"QUAL": stable, "MTUM": mtum, "VLUE": stable, "USMV": stable})
    w2 = FactorBlend().generate(px2)
    assert [(d.year, d.month) for d in w2.index] == [(2000, 1), (2000, 4), (2000, 12)]
    assert (w2.to_numpy() == 0.25).all()


def test_factor_blend_small_drift_ignored():
    n = 6
    stable = ones(n)
    px = daily_from_monthly({"QUAL": stable, "MTUM": [100.0, 110, 115, 118, 119, 120], "VLUE": stable, "USMV": stable})
    # weight of MTUM at 120: 120/420 = 28.6% -> drift 3.6pp < 5pp
    assert len(FactorBlend().generate(px)) == 1


SECTORS = ["S1", "S2", "S3", "S4", "S5"]


def sector_prices(final_jump=None, below_sma=(), market_down=False):
    """14 month-end closes. Momentum (skip 1) ranks S1 > S2 > S3 > S4 > S5.

    ``final_jump`` makes one sector jump in the most recent month only, which
    must NOT change the ranking because that month is skipped.
    """
    data = {}
    for k, s in enumerate(SECTORS):
        growth = 1 + 0.05 * (len(SECTORS) - k)          # S1 grows most
        path = list(100 * np.linspace(1, growth, 13))
        last = path[-1] * 1.01
        if s == final_jump:
            last = path[-1] * 3.0
        if s in below_sma:
            last = 50.0                                   # far below its 10-month SMA
        data[s] = path + [last]
    data["CASH"] = ones(14)
    data["MKT"] = [100.0] * 13 + ([50.0] if market_down else [120.0])
    return daily_from_monthly(data)


def test_sector_ranking_top3_equal_weight():
    s = SectorMomentum(SECTORS, cash="CASH", top_n=3, filter="per_sector_sma")
    w = s.generate(sector_prices())
    assert len(w) == 1                                   # needs 14 month-end closes
    row = w.iloc[0]
    assert row[["S1", "S2", "S3"]].tolist() == pytest.approx([1 / 3] * 3)
    assert row[["S4", "S5", "CASH"]].sum() == 0


def test_skip_month_ignores_latest_month():
    s = SectorMomentum(SECTORS, cash="CASH", top_n=3)
    row = s.generate(sector_prices(final_jump="S5")).iloc[0]
    assert row["S5"] == 0                                # last-month surge is skipped
    no_skip = SectorMomentum(SECTORS, cash="CASH", top_n=3, skip_months=0)
    assert no_skip.generate(sector_prices(final_jump="S5")).iloc[-1]["S5"] == pytest.approx(1 / 3)


def test_per_sector_filter_replaces_one_sleeve():
    s = SectorMomentum(SECTORS, cash="CASH", top_n=3, filter="per_sector_sma")
    row = s.generate(sector_prices(below_sma=("S2",))).iloc[0]
    # S2 is still ranked top-3 (its skip-month momentum is unchanged) but fails its SMA
    assert row["S2"] == 0 and row["CASH"] == pytest.approx(1 / 3)
    assert row["S1"] == pytest.approx(1 / 3) and row["S3"] == pytest.approx(1 / 3)


def test_market_filter_moves_everything_to_cash():
    s = SectorMomentum(SECTORS, cash="CASH", top_n=3, filter="market_sma", market="MKT")
    assert s.generate(sector_prices(market_down=True)).iloc[0]["CASH"] == 1.0
    up = s.generate(sector_prices(below_sma=("S2",))).iloc[0]
    assert up["S2"] == pytest.approx(1 / 3)              # market filter ignores per-sector trends


def test_sector_ties_broken_by_listed_order():
    data = {s: ones(14) for s in SECTORS}
    data["CASH"] = ones(14)
    w = SectorMomentum(SECTORS, cash="CASH", top_n=2).generate(daily_from_monthly(data))
    # all momenta equal (0) and prices equal to SMA -> sleeves go to cash; ranking itself is S1, S2
    assert w.iloc[0]["CASH"] == 1.0
    assert SectorMomentum(SECTORS, cash="CASH", top_n=2).ranked(
        daily_from_monthly(data).iloc[-300:].resample("ME").last())[:2] == ["S1", "S2"]


def test_invalid_configs():
    with pytest.raises(ValueError):
        SectorMomentum(SECTORS, filter="bogus")
    with pytest.raises(ValueError):
        SectorMomentum(SECTORS, filter="market_sma")
    with pytest.raises(ValueError):
        SectorMomentum(SECTORS, top_n=9)


# ------------------------------------------------------------------ registry and no look-ahead

def test_registry_builds_configured_defaults(settings):
    names = strategy_names(settings)
    assert names == ["benchmark", "trend_faber", "gem", "gtaa5", "factor_blend",
                     "sector_mom_sector_filter", "sector_mom_market_filter"]
    s5 = build_strategy("sector_mom_sector_filter", settings)
    assert (s5.lookback, s5.skip, s5.top_n, s5.sma_months, s5.filter) == (12, 1, 3, 10, "per_sector_sma")
    s6 = build_strategy("sector_mom_market_filter", settings)
    assert s6.filter == "market_sma" and s6.market == "SPY"
    fb = build_strategy("factor_blend", settings)
    assert fb.rebalance_month == 12 and fb.drift_threshold == 0.05
    with pytest.raises(KeyError):
        build_strategy("nope", settings)


def random_prices(symbols, start="1999-01-01", end="2012-12-31", seed=0):
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    rets = rng.normal(3e-4, 0.012, (len(days), len(symbols)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days, columns=symbols)


@pytest.mark.parametrize("name", ["benchmark", "trend_faber", "gem", "gtaa5", "factor_blend",
                                  "sector_mom_sector_filter", "sector_mom_market_filter"])
def test_no_lookahead_changing_future_data_leaves_past_signals(settings, name):
    strat = build_strategy(name, settings)
    px = random_prices(strat.symbols(), seed=hash(name) % 1000)
    cutoff = pd.Timestamp("2006-06-30")
    base = strat.generate(px)
    shocked = px.copy()
    rng = np.random.default_rng(99)
    after = shocked.index > cutoff
    shocked.loc[after] = shocked.loc[after] * rng.uniform(0.2, 5.0, size=shocked.loc[after].shape)
    alt = build_strategy(name, settings).generate(shocked)
    pd.testing.assert_frame_equal(base[base.index <= cutoff], alt[alt.index <= cutoff])
    assert len(base[base.index <= cutoff]) > 0
    if name not in ("benchmark",):
        assert not base[base.index > cutoff].equals(alt[alt.index > cutoff])   # the shock did matter later


def test_signal_rows_are_sorted_when_holdings_change():
    spy = [100 + i for i in range(12)] + [90.0, 120.0, 80.0, 130.0]
    w = TrendFilter("SPY", "BIL", 10).generate(daily_from_monthly({"SPY": spy, "BIL": ones(16)}))
    assert w.index.is_monotonic_increasing
