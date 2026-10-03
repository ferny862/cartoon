"""Tiingo and yfinance sources with all network calls mocked."""

import pandas as pd
import pytest

from src.data.tiingo import TiingoClient, TiingoError
from src.config import MissingSecretError
from src.data.yahoo import YahooSource

KEY = "test-key-0123456789"


class FakeResponse:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.response


TIINGO_ROWS = [
    {"date": "2020-01-02T00:00:00.000Z", "close": 324.87, "adjClose": 300.0, "divCash": 0.0, "splitFactor": 1.0, "volume": 1000},
    {"date": "2020-01-03T00:00:00.000Z", "close": 322.41, "adjClose": 297.7, "divCash": 0.0, "splitFactor": 1.0, "volume": 1100},
    {"date": "2020-01-06T00:00:00.000Z", "close": 323.64, "adjClose": 298.9, "divCash": 1.4, "splitFactor": 1.0, "volume": 900},
]


def test_tiingo_requires_key():
    with pytest.raises(MissingSecretError):
        TiingoClient(session=FakeSession(FakeResponse([])))


def test_tiingo_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", KEY)
    sess = FakeSession(FakeResponse(TIINGO_ROWS))
    TiingoClient(session=sess).get_daily_prices("SPY")
    assert sess.calls[0]["headers"]["Authorization"] == f"Token {KEY}"


def test_tiingo_key_sent_in_header_not_url():
    sess = FakeSession(FakeResponse(TIINGO_ROWS))
    client = TiingoClient(api_key=KEY, session=sess)
    client.get_daily_prices("SPY", start="2020-01-01")
    call = sess.calls[0]
    assert KEY not in call["url"]
    assert KEY not in str(call["params"])
    assert call["url"].endswith("/tiingo/daily/spy/prices")
    assert KEY not in repr(client)


def test_tiingo_parses_to_common_schema():
    df = TiingoClient(api_key=KEY, session=FakeSession(FakeResponse(TIINGO_ROWS))).get_daily_prices("SPY")
    assert list(df.columns) == ["close", "adj_close", "div_cash", "split_factor", "volume"]
    assert df.index[0] == pd.Timestamp("2020-01-02") and df.index.tz is None
    assert df.loc["2020-01-06", "div_cash"] == 1.4
    assert df["adj_close"].iloc[1] == pytest.approx(297.7)


def test_tiingo_http_error_redacts_key():
    resp = FakeResponse(None, status=401, text=f"invalid token {KEY}")
    with pytest.raises(TiingoError) as exc:
        TiingoClient(api_key=KEY, session=FakeSession(resp)).get_daily_prices("SPY")
    assert "401" in str(exc.value) and KEY not in str(exc.value)


def test_tiingo_empty_response_is_error():
    with pytest.raises(TiingoError, match="no rows"):
        TiingoClient(api_key=KEY, session=FakeSession(FakeResponse([]))).get_daily_prices("SPY")


def test_tiingo_uses_rate_limiter():
    class CountingLimiter:
        n = 0

        def acquire(self):
            self.n += 1

    lim = CountingLimiter()
    TiingoClient(api_key=KEY, session=FakeSession(FakeResponse(TIINGO_ROWS)), limiter=lim).get_daily_prices("SPY")
    assert lim.n == 1


def _yahoo_frame():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"]).tz_localize("America/New_York")
    return pd.DataFrame(
        {"Open": [1, 1], "Close": [324.87, 322.41], "Adj Close": [300.1, 297.8],
         "Volume": [10, 11], "Dividends": [0.0, 0.0], "Stock Splits": [0.0, 0.0]},
        index=idx,
    )


def test_yahoo_normalizes_columns_and_timezone():
    df = YahooSource(downloader=lambda s: _yahoo_frame(), sleep=lambda s: None).get_daily_prices("SPY")
    assert df.index[0] == pd.Timestamp("2020-01-02") and df.index.tz is None
    assert (df["split_factor"] == 1.0).all()
    assert df["adj_close"].iloc[0] == pytest.approx(300.1)


def test_yahoo_enforces_minimum_interval():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    src = YahooSource(min_seconds_between_requests=2.0, downloader=lambda s: _yahoo_frame(),
                      clock=lambda: t[0], sleep=sleep)
    src.get_daily_prices("SPY")
    t[0] += 0.5
    src.get_daily_prices("EFA")
    assert slept == [pytest.approx(1.5)]
