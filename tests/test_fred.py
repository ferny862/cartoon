import numpy as np
import pandas as pd
import pytest

from src.data.fred import discount_to_daily_growth, fetch_fred_series, parse_fred_csv, tbill_returns

CSV = """observation_date,DTB3
2020-01-01,.
2020-01-02,1.52
2020-01-03,1.50
2020-01-06,
2020-01-07,1.54
"""


def test_parse_handles_missing_markers():
    s = parse_fred_csv(CSV)
    assert np.isnan(s.loc["2020-01-01"]) and np.isnan(s.loc["2020-01-06"])
    assert s.loc["2020-01-02"] == 1.52


def test_parse_accepts_old_date_header():
    s = parse_fred_csv(CSV.replace("observation_date", "DATE"))
    assert len(s) == 5


def test_discount_conversion_hand_computed():
    # 5% discount, 91-day bill: price = 1 - 0.05*91/360 = 0.987361...
    price = 1 - 0.05 * 91 / 360
    g = discount_to_daily_growth(5.0)
    assert g ** 91 == pytest.approx(1 / price)
    # Compounded over 365 days: 1.012800 ** (365/91) = exp(4.0110 * 0.012719) = 1.05234,
    # i.e. about 5.23% effective, above the 5% discount quote.
    assert g ** 365 - 1 == pytest.approx(0.05234, abs=1e-4)


def test_tbill_returns_use_prior_rate_and_calendar_days():
    rates = pd.Series([4.0, 8.0], index=pd.to_datetime(["2020-01-03", "2020-01-06"]))
    dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    r = tbill_returns(rates, pd.DatetimeIndex(dates))
    assert np.isnan(r.iloc[0])
    # Friday -> Monday: 3 calendar days at Friday's 4% rate (not Monday's 8%).
    assert r.iloc[1] == pytest.approx(discount_to_daily_growth(4.0) ** 3 - 1)
    assert r.iloc[2] == pytest.approx(discount_to_daily_growth(8.0) - 1)


def test_tbill_returns_no_lookahead():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    rates = pd.Series(1.5, index=dates)
    base = tbill_returns(rates, dates)
    changed = rates.copy()
    changed.loc["2020-03-01":] = 9.0
    after = tbill_returns(changed, dates)
    cutoff = pd.Timestamp("2020-03-02")  # first trading day using the changed rate is the next one
    pd.testing.assert_series_equal(base[base.index <= cutoff], after[after.index <= cutoff])


def test_fetch_uses_session():
    class Resp:
        status_code = 200
        text = CSV

    class Sess:
        def get(self, url, params=None, timeout=None):
            self.params = params
            return Resp()

    sess = Sess()
    s = fetch_fred_series(session=sess)
    assert sess.params == {"id": "DTB3"} and s.notna().sum() == 3
