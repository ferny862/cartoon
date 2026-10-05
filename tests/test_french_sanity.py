"""Daily French data parsing and the long-history sanity-check inputs."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.market import french_market_data
from src.data.french import LABEL, factor_returns_daily, industry_returns_daily, parse_french_csv
from src.strategies import build_french_variant

DAILY_IND = """This file was created by CMPT_IND_RETS_DAILY.
Missing data are indicated by -99.99 or -999.

  Average Value Weighted Returns -- Daily
,NoDur,Durbl,Manuf,Enrgy,HiTec,Telcm,Shops,Hlth ,Utils,Other
19260701,    0.06,    0.37,    0.24,    0.66,    0.38,    0.33,    0.05,    0.43,    0.88,    0.12
19260702,    0.38,    0.18,    0.47,    0.56,    0.38,    0.35,    0.18,    0.89,    0.26,    0.43

  Average Equal Weighted Returns -- Daily
,NoDur,Durbl,Manuf,Enrgy,HiTec,Telcm,Shops,Hlth ,Utils,Other
19260701,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00
19260702,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00,    9.00
"""

DAILY_FAC = """This file was created by CMPT_ME_BEME_RETS_DAILY.

,Mkt-RF,SMB,HML,RF
19260701,    0.10,   -0.25,   -0.27,    0.01
19260702,    0.45,   -0.33,   -0.06,    0.01
"""


def test_daily_parsing():
    ind = industry_returns_daily(DAILY_IND)
    assert ind.index[0] == pd.Timestamp("1926-07-01") and len(ind) == 2
    assert ind.loc["1926-07-02", "Hlth"] == pytest.approx(0.0089)      # value-weighted, not equal-weighted
    assert ind.attrs["label"] == LABEL
    fac = factor_returns_daily(DAILY_FAC)
    assert fac.loc["1926-07-01", "RF"] == pytest.approx(0.0001)
    assert len(parse_french_csv(DAILY_IND)) == 2


def test_french_market_data_indices():
    ind = industry_returns_daily(DAILY_IND)
    fac = factor_returns_daily(DAILY_FAC)
    md = french_market_data(ind, fac)
    assert md.prices.loc["1926-07-02", "MKT"] == pytest.approx((1 + 0.0011) * (1 + 0.0046))
    assert md.prices.loc["1926-07-02", "RF"] == pytest.approx(1.0001 ** 2)
    assert "NON-INVESTABLE" in md.notes["universe"]


def synthetic_french(start="1926-07-01", end="1940-12-31", seed=4):
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    cols = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
    ind = pd.DataFrame(rng.normal(3e-4, 0.012, (len(days), 10)), index=days, columns=cols)
    fac = pd.DataFrame({"Mkt-RF": ind.mean(axis=1) - 1e-4, "SMB": 0.0, "HML": 0.0, "RF": 1e-4}, index=days)
    return ind, fac


@pytest.mark.parametrize("name", ["sector_mom_sector_filter", "sector_mom_market_filter"])
def test_french_variants_run_end_to_end(settings, name):
    ind, fac = synthetic_french()
    md = french_market_data(ind, fac)
    strat = build_french_variant(name, settings, list(ind.columns))
    assert strat.name.endswith("__french") and strat.cash == "RF"
    w = strat.generate(md.prices)
    assert w.index[0] == pd.Timestamp("1927-08-31")       # 14 month-end closes from July 1926
    res = run_backtest(w, md, execution_lag=1)
    assert res.equity.notna().all() and len(res.equity) > 3000
    if name == "sector_mom_market_filter":
        assert strat.market == "MKT"


def test_french_variant_only_for_sector_strategies(settings):
    with pytest.raises(ValueError):
        build_french_variant("gem", settings, ["A"])
