"""Tiingo end-of-day price client (primary source).

* The API key is read from TIINGO_API_KEY and sent in the Authorization
  header, never in the URL, so it cannot leak through logged URLs.
* Requests go through a persistent RateLimiter and SymbolBudget sized to the
  free tier.
* Tiingo's adjusted closes are re-based whenever a new dividend is paid, so a
  refresh always downloads the full history instead of appending.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import get_secret, redact
from src.data.ratelimit import RateLimiter, SymbolBudget
from src.data.schema import normalize_price_frame

log = logging.getLogger(__name__)

_FIELD_MAP = {
    "close": "close",
    "adjClose": "adj_close",
    "divCash": "div_cash",
    "splitFactor": "split_factor",
    "volume": "volume",
}


class TiingoError(RuntimeError):
    pass


class TiingoClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.tiingo.com",
        session: Any | None = None,
        limiter: RateLimiter | None = None,
        symbol_budget: SymbolBudget | None = None,
        timeout: float = 30.0,
    ):
        self._api_key = api_key or get_secret("TIINGO_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.limiter = limiter
        self.symbol_budget = symbol_budget
        self.timeout = timeout

    def __repr__(self) -> str:  # never show the key
        return f"TiingoClient(base_url={self.base_url!r})"

    @classmethod
    def from_settings(cls, settings: dict, state_dir: str | Path, **kwargs: Any) -> "TiingoClient":
        cfg = settings["data"]["tiingo"]
        state_dir = Path(state_dir)
        limiter = RateLimiter(
            [(cfg["max_requests_per_hour"], 3600), (cfg["max_requests_per_day"], 86400)],
            state_path=state_dir / "tiingo_requests.json",
        )
        budget = SymbolBudget(cfg["max_unique_symbols_per_month"], state_path=state_dir / "tiingo_symbols.json")
        return cls(
            base_url=cfg["base_url"],
            limiter=limiter,
            symbol_budget=budget,
            timeout=cfg["timeout_seconds"],
            **kwargs,
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.limiter is not None:
            self.limiter.acquire()
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Token {self._api_key}", "Content-Type": "application/json"}
        try:
            resp = self.session.get(url, params=params or {}, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TiingoError(redact(f"Request to {path} failed: {exc}", (self._api_key,))) from None
        if resp.status_code != 200:
            body = redact(str(getattr(resp, "text", ""))[:200], (self._api_key,))
            raise TiingoError(f"Tiingo returned HTTP {resp.status_code} for {path}: {body}")
        return resp.json()

    def get_metadata(self, symbol: str) -> dict[str, Any]:
        """Ticker metadata, including Tiingo's startDate for the symbol."""
        return self._get(f"/tiingo/daily/{symbol.lower()}")

    def get_daily_prices(self, symbol: str, start: str = "1990-01-01", end: str | None = None) -> pd.DataFrame:
        if self.symbol_budget is not None:
            self.symbol_budget.check_and_add(symbol.upper())
        params = {"startDate": start, "format": "json", "resampleFreq": "daily"}
        if end:
            params["endDate"] = end
        rows = self._get(f"/tiingo/daily/{symbol.lower()}/prices", params)
        if not isinstance(rows, list):
            raise TiingoError(f"Unexpected response type for {symbol}: {type(rows).__name__}")
        if not rows:
            raise TiingoError(f"Tiingo returned no rows for {symbol}")
        df = pd.DataFrame(rows)
        missing = [k for k in ("date", "close", "adjClose") if k not in df.columns]
        if missing:
            raise TiingoError(f"Tiingo response for {symbol} lacks fields {missing}")
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        df = df.set_index("date").rename(columns=_FIELD_MAP)
        log.info("Tiingo: downloaded %d rows for %s (%s to %s)", len(df), symbol, df.index.min().date(), df.index.max().date())
        return normalize_price_frame(df)
