from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.validation.common import monthly_excess, monthly_returns, monthly_risk_free, slice_equity
from src.validation.walkforward import consistency, expanding_windows, fixed_windows, regime_table


def test_monthly_returns_includes_first_partial_month():
    idx = pd.to_datetime(["2020-01-15", "2020-01-31", "2020-02-28", "2020-03-31"])
    eq = pd.Series([100.0, 110.0, 99.0, 108.9], index=idx)
    r = monthly_returns(eq)
    assert r.tolist() == pytest.approx([0.10, -0.10, 0.10])
    assert r.index[0] == pd.Timestamp("2020-01-31")


def test_monthly_risk_free_and_excess():
    days = pd.bdate_range("2020-01-01", "2020-02-28")
    rf = pd.Series(0.0001, index=days)
    mrf = monthly_risk_free(rf)
    n_jan = (days.month == 1).sum()
    assert mrf.iloc[0] == pytest.approx(1.0001 ** n_jan - 1)
    eq = pd.Series(np.cumprod(np.full(len(days), 1.0001)), index=days)
    assert monthly_excess(eq, rf).iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_slice_equity_requires_live_at_start():
    days = pd.bdate_range("2005-01-03", "2010-12-31")
    eq = pd.Series(np.arange(len(days), dtype=float) + 1, index=days)
    assert slice_equity(eq, "2000-01-01", "2004-12-31") is None        # ends before the curve starts
    late = slice_equity(eq, "2003-01-01", "2006-12-31")                 # started mid-window: from its start
    assert late.index[0] == days[0] and late.index[-1] == pd.Timestamp("2006-12-29")
    assert slice_equity(eq, "2001-01-01", "2005-06-30") is None         # under ~10 months covered
    s = slice_equity(eq, "2006-01-01", "2006-12-31")
    assert s.index[0] == pd.Timestamp("2005-12-30")                    # rebased on the prior close


def _run(eq, bench):
    return SimpleNamespace(pretax=SimpleNamespace(equity=eq), benchmark=SimpleNamespace(equity=bench))


def _curves():
    days = pd.bdate_range("2000-01-03", "2009-12-31")
    bench = pd.Series(100 * np.cumprod(np.full(len(days), 1.0003)), index=days)
    strat = bench.copy()
    # strategy avoids a 30% fall in 2002 but otherwise matches SPY
    crash = (days >= "2002-03-01") & (days <= "2002-09-30")
    b_ret = np.where(crash, -0.002, 0.0003)
    bench = pd.Series(100 * np.cumprod(1 + b_ret), index=days)
    s_ret = np.where(crash, 0.0001, 0.0003)
    strat = pd.Series(100 * np.cumprod(1 + s_ret), index=days)
    return days, strat, bench


def test_fixed_windows_and_consistency():
    days, strat, bench = _curves()
    rf = pd.Series(0.0, index=days)
    t = fixed_windows({"s": _run(strat, bench)}, [("2000-01-01", "2004-12-31"), ("2005-01-01", "2009-12-31"),
                                                   ("2010-01-01", "2014-12-31")], rf)
    w1 = t.iloc[0]
    assert w1["status"] == "ok" and w1["beat_benchmark_return"] and w1["smaller_drawdown"]
    assert t.iloc[1]["excess_cagr"] == pytest.approx(0.0, abs=1e-12)
    assert t.iloc[2]["status"] == "not live"
    assert t.iloc[0]["live_from"] == days[0].date()
    c = consistency(t).iloc[0]
    assert c["windows"] == 2 and c["beat_return"] == 1


def test_expanding_windows_grow_to_the_end():
    days, strat, bench = _curves()
    t = expanding_windows({"s": _run(strat, bench)}, 5, pd.Series(0.0, index=days))
    assert len(t) == 2 and t.iloc[-1]["end"] == days[-1].date()


def test_regime_table_hides_locked_heldout():
    days, strat, bench = _curves()
    regimes = {"bear": ("2000-01-01", "2002-12-31"), "later": ("2022-01-01", "2022-12-31")}
    t = regime_table({"s": _run(strat, bench)}, regimes, pd.Series(0.0, index=days),
                     pd.Timestamp("2021-10-01"), heldout_unlocked=False)
    assert t.iloc[0]["status"] == "ok"
    assert t.iloc[1]["status"] == "held out (locked)" and "cagr" not in t.columns[t.iloc[1].notna()]
