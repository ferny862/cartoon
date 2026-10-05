"""yfinance cross-check source.

yfinance is an unofficial wrapper intended for personal use. It is used here
only to cross-check Tiingo, with caching (by the caller) and a minimum delay
between requests. It is imported lazily so tests never touch it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import pandas as pd

from src.data.schema import normalize_price_frame

log = logging.getLogger(__name__)

_COLUMN_MAP = {
    "Close": "close",
    "Adj Close": "adj_close",
    "Dividends": "div_cash",
    "Stock Splits": "split_factor",
    "Volume": "volume",
}


def _default_downloader(symbol: str) -> pd.DataFrame:
    import yfinance as yf  # lazy: optional dependency

    return yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=True)


class YahooSource:
    def __init__(
        self,
        min_seconds_between_requests: float = 2.0,
        downloader: Callable[[str], pd.DataFrame] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = min_seconds_between_requests
        self._downloader = downloader or _default_downloader
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None

    def get_daily_prices(self, symbol: str) -> pd.DataFrame:
        if self._last_request is not None:
            elapsed = self._clock() - self._last_request
            if elapsed < self.min_interval:
                self._sleep(self.min_interval - elapsed)
        self._last_request = self._clock()
        raw = self._downloader(symbol)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {symbol}")
        df = raw.rename(columns=_COLUMN_MAP)
        if "split_factor" in df.columns:
            # yfinance reports 0 on days without a split.
            df["split_factor"] = df["split_factor"].replace(0, 1.0)
        log.info("yfinance: downloaded %d rows for %s", len(df), symbol)
        return normalize_price_frame(df)
