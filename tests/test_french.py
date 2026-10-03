import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.french import LABEL, factor_returns, fetch_french_dataset, industry_returns, parse_french_csv

FIX = Path(__file__).parent / "fixtures"
IND = (FIX / "french_industries_sample.csv").read_text()
FAC = (FIX / "french_factors_sample.csv").read_text()


def test_parser_splits_all_tables():
    tables = parse_french_csv(IND)
    titles = list(tables)
    assert len(titles) == 3
    assert "Average Value Weighted Returns -- Monthly" in titles[0]
    assert len(tables[titles[2]]) == 2  # annual table


def test_industry_returns_value_weighted_monthly_in_decimal():
    df = industry_returns(IND)
    assert list(df.columns)[:3] == ["NoDur", "Durbl", "Manuf"]
    assert "Hlth" in df.columns  # header whitespace stripped
    assert df.index[0] == pd.Timestamp("1926-07-31")
    assert df.loc["1926-07-31", "Durbl"] == pytest.approx(0.1555)
    assert len(df) == 3
    assert df.attrs["label"] == LABEL and "NON-INVESTABLE" in LABEL


def test_missing_marker_becomes_nan():
    tables = parse_french_csv(IND)
    ew = tables[[t for t in tables if "Equal" in t][0]]
    assert np.isnan(ew.loc["1926-07-31", "Hlth"])


def test_factor_returns():
    df = factor_returns(FAC)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert df.loc["1926-08-31", "RF"] == pytest.approx(0.0025)
    assert len(df) == 3  # annual table excluded


def test_fetch_unzips():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("10_Industry_Portfolios.CSV", IND)

    class Resp:
        status_code = 200
        content = buf.getvalue()

    class Sess:
        def get(self, url, timeout=None):
            self.url = url
            return Resp()

    sess = Sess()
    text = fetch_french_dataset("10_Industry_Portfolios", session=sess)
    assert sess.url.endswith("/10_Industry_Portfolios_CSV.zip")
    assert len(industry_returns(text)) == 3
