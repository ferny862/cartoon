"""DataStore: inception handling, calendar alignment, cash splice, held-out lock."""

import numpy as np
import pandas as pd
import pytest

from src.data.cache import Cache
from src.data.fred import discount_to_daily_growth
from src.data.store import (
    NS_FRED,
    NS_TIINGO,
    DataStore,
    HeldOutLockedError,
    align_to_calendar,
    apply_inception,
    clip_heldout,
    splice_cash_returns,
)
from tests.conftest import make_prices


def test_apply_inception_blanks_pre_inception_values():
    idx = pd.bdate_range("2007-05-01", "2007-06-29")
    df = pd.DataFrame({"BIL": 1.0, "SPY": 2.0}, index=idx)
    out = apply_inception(df, {"BIL": pd.Timestamp("2007-05-25"), "SPY": pd.Timestamp("1993-01-22")})
    assert out.loc[:"2007-05-24", "BIL"].isna().all()
    assert out.loc["2007-05-25":, "BIL"].notna().all()
    assert out["SPY"].notna().all()


def test_align_never_fills_before_first_value_or_long_gaps():
    cal = pd.bdate_range("2020-01-01", periods=12)
    s = pd.Series([np.nan, np.nan, 10, 11, np.nan, 12, np.nan, np.nan, np.nan, np.nan, 13, np.nan], index=cal)
    out = align_to_calendar(s.to_frame("X"), cal, max_ffill_days=3)["X"]
    assert out.iloc[:2].isna().all()           # no back-fill before first value
    assert out.iloc[4] == 11                    # short interior gap filled
    assert out.iloc[6:9].tolist() == [12, 12, 12] and np.isnan(out.iloc[9])  # limit respected
    assert np.isnan(out.iloc[11])               # trailing gap not filled


def test_splice_cash_uses_fred_before_etf_inception():
    cal = pd.bdate_range("2007-05-21", "2007-06-08")
    fred = pd.Series(0.0001, index=cal)
    etf = pd.Series(np.nan, index=cal, name="BIL")
    etf.loc["2007-05-30":] = 100 * np.cumprod(np.full(len(etf.loc["2007-05-30":]), 1.0002))
    out = splice_cash_returns(etf, fred)
    assert (out.loc[:"2007-05-30", "source"] == "FRED_DTB3").all()  # first ETF return is the day after
    assert (out.loc["2007-05-31":, "source"] == "BIL").all()
    assert out.loc["2007-06-01", "return"] == pytest.approx(0.0002)
    assert out.loc["2007-05-22", "return"] == pytest.approx(0.0001)


def test_clip_heldout(settings):
    idx = pd.bdate_range("2021-09-27", "2021-10-08")
    df = pd.DataFrame({"x": 1.0}, index=idx)
    assert clip_heldout(df, settings).index.max() < pd.Timestamp("2021-10-01")
    with pytest.raises(HeldOutLockedError):
        clip_heldout(df, settings, include_heldout=True)
    unlocked = {**settings, "dates": {**settings["dates"], "heldout_unlocked": True}}
    assert clip_heldout(df, unlocked).index.max() == idx.max()


@pytest.fixture
def store(tmp_path, settings):
    cache = Cache(tmp_path)
    cal = pd.bdate_range("2007-01-02", "2022-12-30")
    cache.write(NS_TIINGO, "SPY", make_prices(cal, seed=1))
    bil_dates = cal[cal >= "2007-05-30"]
    bil = make_prices(bil_dates, daily_return=0.0001)
    # Simulate a bad provider row dated before inception: it must be dropped.
    bad = make_prices(pd.DatetimeIndex(["2007-01-03"]))
    cache.write(NS_TIINGO, "BIL", pd.concat([bad, bil]))
    xlk = make_prices(cal, seed=2)
    xlk.loc["2010-03-01", "div_cash"] = 0.5
    xlk = xlk.drop(pd.Timestamp("2010-03-03"))  # one missing day
    cache.write(NS_TIINGO, "XLK", xlk)
    cache.write(NS_FRED, "DTB3", pd.Series(5.0, index=cal, name="DTB3").to_frame())
    s = {**settings, "data": {**settings["data"], "cache_dir": str(tmp_path)}}
    return DataStore(s, cache=cache, tiingo=object(), yahoo=object())


def test_adjusted_closes_masks_inception_and_heldout(store):
    px = store.adjusted_closes(["SPY", "BIL", "XLK"])
    assert px.index.max() < pd.Timestamp("2021-10-01")
    assert px.loc[:"2007-05-24", "BIL"].isna().all()   # the bad pre-inception row is gone
    assert px.loc["2007-05-30", "BIL"] == pytest.approx(100.0)
    assert not np.isnan(px.loc["2010-03-03", "XLK"])  # one-day gap forward-filled
    assert px.loc["2010-03-03", "XLK"] == px.loc["2010-03-02", "XLK"]


def test_heldout_request_refused_while_locked(store):
    with pytest.raises(HeldOutLockedError):
        store.adjusted_closes(["SPY"], include_heldout=True)


def test_dividends_frame(store):
    div = store.dividends(["XLK", "BIL"])
    assert div.loc["2010-03-01", "XLK"] == 0.5
    assert (div.loc[:"2007-05-24", "BIL"] == 0).all() and not div.isna().any().any()


def test_cash_returns_splice(store):
    cash = store.cash_returns()
    assert cash.loc["2007-01-04", "source"] == "FRED_DTB3"
    assert cash.loc["2007-01-04", "return"] == pytest.approx(discount_to_daily_growth(5.0) - 1)
    assert cash.loc["2008-01-03", "source"] == "BIL"
    assert cash.loc["2008-01-03", "return"] == pytest.approx(0.0001)
    assert cash.index.max() < pd.Timestamp("2021-10-01")


def test_store_downloads_through_mocked_client(tmp_path, settings):
    class FakeTiingo:
        def __init__(self):
            self.calls = []

        def get_daily_prices(self, sym, start):
            self.calls.append(sym)
            return make_prices(pd.bdate_range("2020-01-01", periods=5))

    fake = FakeTiingo()
    s = {**settings, "data": {**settings["data"], "cache_dir": str(tmp_path)}}
    st = DataStore(s, cache=Cache(tmp_path), tiingo=fake)
    assert st.download_prices(["SPY", "XLK"]) == ["SPY", "XLK"]
    assert st.download_prices(["SPY", "XLK"]) == []      # cached and fresh
    assert st.download_prices(["SPY"], refresh=True) == ["SPY"]
    assert fake.calls == ["SPY", "XLK", "SPY"]
