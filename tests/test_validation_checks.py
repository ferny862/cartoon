"""Data-quality checks on small synthetic series."""

import numpy as np
import pandas as pd

from src.data import validation as v
from tests.conftest import make_prices

CFG = {
    "extreme_move_threshold": 0.20,
    "stale_run_days": 5,
    "max_staleness_business_days": 5,
    "max_gap_business_days": 3,
    "inception_tolerance_days": 10,
}


def checks(issues):
    return {i.check for i in issues}


def test_clean_series_passes(bdays):
    px = make_prices(bdays, seed=1)
    issues = v.validate_prices("SPY", px, bdays, pd.Timestamp("2020-01-01"), CFG, as_of=bdays[-1])
    assert issues == []


def test_missing_days_detected(bdays):
    px = make_prices(bdays, seed=1).drop(bdays[[10, 11]])
    issues = v.check_missing_days("XLK", px, bdays)
    assert len(issues) == 1 and "2 trading days missing" in issues[0].message


def test_long_gap_detected(bdays):
    px = make_prices(bdays, seed=1).drop(bdays[20:26])
    issues = v.check_calendar_gaps("XLK", px.index, 3)
    assert issues and "6 business days" in issues[0].message


def test_zero_negative_and_nan_prices(bdays):
    px = make_prices(bdays[:20])
    px.iloc[3, px.columns.get_loc("close")] = 0.0
    px.iloc[5, px.columns.get_loc("adj_close")] = -1.0
    px.iloc[7, px.columns.get_loc("adj_close")] = np.nan
    found = checks(v.check_bad_prices("X", px))
    assert {"nonpositive_price", "nan_price"} <= found


def test_extreme_move_flagged(bdays):
    px = make_prices(bdays[:20])
    px.iloc[10:, px.columns.get_loc("adj_close")] *= 1.30
    issues = v.check_extreme_moves("X", px, 0.20)
    assert len(issues) == 1 and issues[0].date == str(bdays[10].date())


def test_stale_run_flagged(bdays):
    px = make_prices(bdays[:30], seed=2)
    px.iloc[5:12, px.columns.get_loc("close")] = 50.0
    issues = v.check_stale_runs("X", px, 5)
    assert len(issues) == 1 and "7 consecutive days" in issues[0].message


def test_stale_last_observation(bdays):
    px = make_prices(bdays[:30])
    assert v.check_staleness("X", px, bdays[-1], 5)
    assert not v.check_staleness("X", px, bdays[31], 5)


def test_inception_mismatch(bdays):
    px = make_prices(bdays)
    early = v.check_inception("X", px, pd.Timestamp("2020-06-01"), 10)
    late = v.check_inception("X", px, pd.Timestamp("2019-06-01"), 10)
    ok = v.check_inception("X", px, pd.Timestamp("2020-01-01"), 10)
    assert early[0].check == "pre_inception_data" and early[0].severity == "error"
    assert late[0].check == "late_first_observation"
    assert ok == []


def test_crosscheck_flags_disagreement(bdays):
    a = make_prices(bdays, seed=3)
    b = a.copy()
    assert v.compare_sources("X", a, b, 0.005, 0.002)[0] == []
    b.loc[b.index >= "2020-06-15", "adj_close"] *= 1.02  # a 2% one-day jump only in source B
    issues, daily = v.compare_sources("X", a, b, 0.005, 0.002)
    assert len(daily) == 1 and daily.index[0] == pd.Timestamp("2020-06-15")
    assert "crosscheck_monthly" in checks(issues)


def test_issues_frame_columns():
    df = v.issues_frame([v.Issue("X", "c", "error", "m", "2020-01-01")])
    assert list(df.columns) == ["symbol", "check", "severity", "message", "date"]
