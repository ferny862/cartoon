import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest, TaxSettings
from src.backtest.market import build_market_data, dividend_yields
from src.backtest.taxes import TaxRates
from src.data.cache import Cache
from src.data.fred import discount_to_daily_growth
from src.data.store import NS_FRED, NS_TIINGO, DataStore
from tests.conftest import make_prices


@pytest.fixture
def store(tmp_path, settings):
    cache = Cache(tmp_path)
    cal = pd.bdate_range("2006-01-02", "2009-12-31")
    cache.write(NS_TIINGO, "SPY", make_prices(cal, seed=1))
    bil = make_prices(cal[cal >= "2007-05-30"], start=50.0, daily_return=0.0001)
    bil.loc["2008-03-03", "div_cash"] = 0.10
    bil.loc["2008-03-03", "close"] = 49.90
    cache.write(NS_TIINGO, "BIL", bil)
    xlk = make_prices(cal, seed=2)
    xlk.loc["2008-06-20", ["close", "div_cash"]] = [19.0, 1.0]
    cache.write(NS_TIINGO, "XLK", xlk)
    cache.write(NS_FRED, "DTB3", pd.Series(4.0, index=cal, name="DTB3").to_frame())
    s = {**settings, "data": {**settings["data"], "cache_dir": str(tmp_path)}}
    return DataStore(s, cache=cache, tiingo=object(), yahoo=object())


def test_dividend_yields():
    div = pd.DataFrame({"A": [0.0, 1.0]})
    raw = pd.DataFrame({"A": [100.0, 99.0]})
    assert dividend_yields(div, raw)["A"].tolist() == pytest.approx([0.0, 0.01])


def test_spliced_cash_asset(store):
    md = build_market_data(store, ["SPY", "BIL"])
    bil = md.prices["BIL"]
    assert bil.notna().all()                                  # cash exists before BIL's inception
    g = discount_to_daily_growth(4.0)
    assert bil.loc["2006-01-04"] / bil.loc["2006-01-03"] == pytest.approx(g)
    assert bil.loc["2008-01-03"] / bil.loc["2008-01-02"] == pytest.approx(1.0001)
    assert md.div_yield.loc["2006-01-04", "BIL"] == pytest.approx((g - 1) / g)   # synthetic distribution
    assert md.div_yield.loc["2008-03-03", "BIL"] == pytest.approx(0.10 / 50.0)
    assert "FRED" in md.notes["BIL"] and "2007-05-31" in md.notes["BIL"]
    assert md.tax_character["BIL"] == "ordinary_treasury"
    assert md.prices.index.max() < pd.Timestamp("2021-10-01")


def test_etf_yields_and_risk_free(store):
    md = build_market_data(store, ["SPY", "XLK"])
    assert md.div_yield.loc["2008-06-20", "XLK"] == pytest.approx(1.0 / 20.0)
    assert md.risk_free.loc["2006-01-04"] == pytest.approx(discount_to_daily_growth(4.0) - 1)


def test_cash_strategy_taxed_as_treasury_interest(store, settings):
    md = build_market_data(store, ["BIL"])
    w = pd.DataFrame({"BIL": [1.0]}, index=[md.prices.index[0]])
    res = run_backtest(w, md, tax=TaxSettings(TaxRates.from_settings(settings)))
    yrs = res.tax_years.set_index("year")
    assert yrs.loc[2006, "income_ordinary_treasury"] > 0
    assert yrs.loc[2006, "state"] == pytest.approx(0.0)           # state-exempt
    assert res.equity.iloc[-1] < run_backtest(w, md).equity.iloc[-1]
