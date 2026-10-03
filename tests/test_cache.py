import pandas as pd

from src.data.cache import Cache
from tests.conftest import make_prices


def test_roundtrip_and_metadata(tmp_path, bdays):
    cache = Cache(tmp_path)
    df = make_prices(bdays[:10])
    cache.write("tiingo", "SPY", df, {"source": "tiingo"})
    back = cache.read("tiingo", "SPY")
    pd.testing.assert_frame_equal(df, back, check_freq=False)
    meta = cache.metadata("tiingo", "SPY")
    assert meta["rows"] == 10 and meta["source"] == "tiingo"
    assert cache.keys("tiingo") == ["SPY"]


def test_freshness(tmp_path, bdays):
    cache = Cache(tmp_path)
    assert not cache.is_fresh("tiingo", "SPY", 1)
    cache.write("tiingo", "SPY", make_prices(bdays[:5]))
    downloaded = cache.metadata("tiingo", "SPY")["downloaded_at"]
    assert cache.is_fresh("tiingo", "SPY", 1, now=downloaded + 3600)
    assert not cache.is_fresh("tiingo", "SPY", 1, now=downloaded + 2 * 86400)
