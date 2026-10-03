"""DataStore: download, cache and serve clean data to the rest of the project.

Guarantees:
* An ETF never has a value before its recorded inception date (or before its
  first real observation). Nothing is back-filled.
* Interior gaps up to ``data.max_ffill_days`` are forward-filled; leading and
  trailing gaps never are.
* Held-out data (on or after ``dates.heldout_start``) is withheld from
  analysis frames until ``dates.heldout_unlocked`` is true. Raw cached data
  is still available to the data-quality checks.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.config import heldout_start, inception_dates, instruments
from src.data import french as french_mod
from src.data import fred as fred_mod
from src.data.cache import Cache

log = logging.getLogger(__name__)

NS_TIINGO = "tiingo"
NS_YAHOO = "yfinance"
NS_FRED = "fred"
NS_FRENCH = "french"


class HeldOutLockedError(RuntimeError):
    """Raised when held-out data is requested before it has been unlocked."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------

def apply_inception(df: pd.DataFrame, inception: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Blank out every value dated before each column's inception date."""
    out = df.copy()
    for col in out.columns:
        if col in inception:
            out.loc[out.index < pd.Timestamp(inception[col]), col] = float("nan")
    return out


def align_to_calendar(df: pd.DataFrame, calendar: pd.DatetimeIndex, max_ffill_days: int) -> pd.DataFrame:
    """Reindex to the trading calendar, forward-filling only short interior gaps."""
    out = df.reindex(calendar)
    if max_ffill_days > 0:
        out = out.ffill(limit=max_ffill_days, limit_area="inside")
    return out


def splice_cash_returns(etf_adj_close: pd.Series | None, fred_returns: pd.Series) -> pd.DataFrame:
    """Daily cash returns: the T-bill ETF once it has a return, FRED before that.

    Returns columns ``return`` and ``source`` (the source of each day's value).
    """
    out = pd.DataFrame({"return": fred_returns.astype(float), "source": "FRED_DTB3"})
    if etf_adj_close is not None and etf_adj_close.notna().any():
        etf_ret = etf_adj_close.pct_change(fill_method=None)
        first = etf_ret.first_valid_index()
        if first is not None:
            mask = out.index >= first
            out.loc[mask, "return"] = etf_ret.reindex(out.index[mask]).to_numpy()
            out.loc[mask, "source"] = str(etf_adj_close.name or "ETF")
    return out


def clip_heldout(df: pd.DataFrame | pd.Series, settings: dict, include_heldout: bool | None = None):
    """Drop rows on or after the held-out start unless it has been unlocked."""
    unlocked = bool(settings["dates"].get("heldout_unlocked", False))
    if include_heldout is None:
        include_heldout = unlocked
    if include_heldout and not unlocked:
        raise HeldOutLockedError(
            "Held-out data is locked. Set dates.heldout_unlocked: true only after the final "
            "configuration is approved, and record the date in docs/heldout_log.md."
        )
    if include_heldout:
        return df
    return df[df.index < heldout_start(settings)]


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------

class DataStore:
    def __init__(
        self,
        settings: dict,
        cache: Cache | None = None,
        tiingo: Any | None = None,
        yahoo: Any | None = None,
        session: Any | None = None,
    ):
        self.settings = settings
        self.cfg = settings["data"]
        self.cache = cache or Cache(self.cfg["cache_dir"])
        self._tiingo = tiingo
        self._yahoo = yahoo
        self._session = session
        self.inception = inception_dates(settings)

    # -- clients (created lazily so no key is needed for cached work) --------
    @property
    def tiingo(self):
        if self._tiingo is None:
            from src.data.tiingo import TiingoClient

            self._tiingo = TiingoClient.from_settings(self.settings, self.cache.root / "_state")
        return self._tiingo

    @property
    def yahoo(self):
        if self._yahoo is None:
            from src.data.yahoo import YahooSource

            self._yahoo = YahooSource(self.cfg["yfinance"]["min_seconds_between_requests"])
        return self._yahoo

    def enabled_symbols(self) -> list[str]:
        return list(instruments(self.settings).keys())

    # -- downloads -----------------------------------------------------------
    def download_prices(self, symbols: list[str] | None = None, refresh: bool = False) -> list[str]:
        """Download full daily history from Tiingo for each symbol not freshly cached."""
        symbols = symbols or self.enabled_symbols()
        done = []
        for sym in symbols:
            if not refresh and self.cache.is_fresh(NS_TIINGO, sym, self.cfg["cache_max_age_days"]):
                log.info("Tiingo %s: cache is fresh, skipping", sym)
                continue
            df = self.tiingo.get_daily_prices(sym, start=self.settings["dates"]["data_start"])
            self.cache.write(NS_TIINGO, sym, df, {"source": "tiingo", "symbol": sym})
            done.append(sym)
        return done

    def download_crosscheck(self, symbols: list[str] | None = None, refresh: bool = False) -> list[str]:
        symbols = symbols or self.enabled_symbols()
        done = []
        for sym in symbols:
            if not refresh and self.cache.is_fresh(NS_YAHOO, sym, self.cfg["cache_max_age_days"]):
                continue
            df = self.yahoo.get_daily_prices(sym)
            self.cache.write(NS_YAHOO, sym, df, {"source": "yfinance", "symbol": sym})
            done.append(sym)
        return done

    def download_fred(self, refresh: bool = False) -> None:
        cfg = self.cfg["fred"]
        if not refresh and self.cache.is_fresh(NS_FRED, cfg["series"], self.cfg["cache_max_age_days"]):
            return
        s = fred_mod.fetch_fred_series(cfg["series"], cfg["url"], self._session, cfg["timeout_seconds"])
        self.cache.write(NS_FRED, cfg["series"], s.to_frame(), {"source": "fred", "units": "percent, discount basis"})

    def download_french(self, refresh: bool = False) -> None:
        cfg = self.cfg["french"]
        parsers = {
            "industries_10": french_mod.industry_returns,
            "industries_12": french_mod.industry_returns,
            "factors": french_mod.factor_returns,
            "industries_10_daily": french_mod.industry_returns_daily,
            "factors_daily": french_mod.factor_returns_daily,
        }
        for key, parse in parsers.items():
            if not refresh and self.cache.is_fresh(NS_FRENCH, key, self.cfg["cache_max_age_days"] * 30):
                continue
            text = french_mod.fetch_french_dataset(cfg[key], cfg["base_url"], self._session, cfg["timeout_seconds"])
            df = parse(text)
            self.cache.write(NS_FRENCH, key, df, {"source": "kenneth_french", "dataset": cfg[key], "label": french_mod.LABEL})

    # -- raw access (no held-out clipping; used by data-quality checks) -----
    def raw_prices(self, symbol: str, source: str = NS_TIINGO) -> pd.DataFrame:
        if not self.cache.exists(source, symbol):
            raise FileNotFoundError(f"No cached {source} data for {symbol}; run the download first")
        return self.cache.read(source, symbol)

    def calendar(self) -> pd.DatetimeIndex:
        """Trading days taken from the reference symbol, within the configured dates."""
        ref = self.raw_prices(self.cfg["reference_calendar_symbol"]).index
        start = pd.Timestamp(self.settings["dates"]["data_start"])
        end = self.settings["dates"].get("end")
        cal = ref[ref >= start]
        if end:
            cal = cal[cal <= pd.Timestamp(end)]
        return pd.DatetimeIndex(cal, name="date")

    # -- analysis frames (inception-masked, aligned, held-out clipped) ------
    def _field(self, symbols: list[str], field: str, include_heldout: bool | None,
               _signal_only: bool = False) -> pd.DataFrame:
        cal = self.calendar()
        cols = {}
        for sym in symbols:
            raw = self.raw_prices(sym)[field]
            first_real = raw.first_valid_index()
            if first_real is None:
                raise ValueError(f"Cached data for {sym} has no valid {field} values")
            raw = raw[raw.index >= max(first_real, self.inception.get(sym, first_real))]
            cols[sym] = raw
        df = pd.DataFrame(cols)
        if field == "adj_close":
            df = align_to_calendar(df, cal, self.cfg["max_ffill_days"])
        else:
            df = df.reindex(cal).fillna(0.0)  # no dividend on missing days
        df = apply_inception(df, self.inception)
        if _signal_only:
            return df
        return clip_heldout(df, self.settings, include_heldout)

    def adjusted_closes(self, symbols: list[str], include_heldout: bool | None = None) -> pd.DataFrame:
        """Total-return (split- and dividend-adjusted) closes."""
        return self._field(symbols, "adj_close", include_heldout)

    def dividends(self, symbols: list[str], include_heldout: bool | None = None) -> pd.DataFrame:
        """Cash dividends per share by ex-date (for the tax model)."""
        df = self._field(symbols, "div_cash", include_heldout)
        # Before inception there is no position, so zero rather than NaN.
        return df.fillna(0.0)

    def raw_closes(self, symbols: list[str], include_heldout: bool | None = None) -> pd.DataFrame:
        """Unadjusted closes (needed to turn per-share dividends into yields)."""
        return self._field(symbols, "close", include_heldout).replace(0.0, float("nan"))

    def cash_returns(self, include_heldout: bool | None = None) -> pd.DataFrame:
        """Daily cash returns: BIL after its first return, FRED DTB3 before."""
        cal = self.calendar()
        rates = self.cache.read(NS_FRED, self.cfg["fred"]["series"]).iloc[:, 0]
        fred_ret = fred_mod.tbill_returns(rates, cal)
        etf = self.settings["cash"]["etf"]
        etf_px = None
        if self.cache.exists(NS_TIINGO, etf):
            etf_px = self._field([etf], "adj_close", include_heldout=self._unlocked())[etf]
            etf_px = etf_px.reindex(cal)
        out = splice_cash_returns(etf_px, fred_ret)
        return clip_heldout(out, self.settings, include_heldout)

    def signal_data(self, symbols: list[str], purpose: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(adjusted closes, raw closes) including recent dates, for paper trading ONLY.

        This is the single approved exception to the held-out lock: paper
        trading needs today's prices to compute this month's target weights
        and to measure fill slippage. It must never be used to compute or
        report backtest performance. Every call is logged to
        logs/heldout_signal_access.log.
        """
        from datetime import datetime, timezone

        from src.config import PROJECT_ROOT

        log.warning("Held-out signal-only access for %s: %s", purpose, ", ".join(symbols))
        log_dir = PROJECT_ROOT / self.settings["project"]["log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "heldout_signal_access.log", "a", encoding="utf-8") as fh:
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            fh.write(f"{stamp}\t{purpose}\t{','.join(symbols)}\n")
        adj = self._field(symbols, "adj_close", None, _signal_only=True)
        raw = self._field(symbols, "close", None, _signal_only=True).replace(0.0, float("nan"))
        return adj, raw

    def risk_free_returns(self, include_heldout: bool | None = None) -> pd.Series:
        """Daily FRED 3-month T-bill returns (used as the risk-free rate in metrics)."""
        cal = self.calendar()
        rates = self.cache.read(NS_FRED, self.cfg["fred"]["series"]).iloc[:, 0]
        rf = fred_mod.tbill_returns(rates, cal).fillna(0.0).rename("risk_free")
        return clip_heldout(rf, self.settings, include_heldout)

    def _unlocked(self) -> bool:
        return bool(self.settings["dates"].get("heldout_unlocked", False))

    def french_industries(self, n: int = 10) -> pd.DataFrame:
        df = self.cache.read(NS_FRENCH, f"industries_{n}")
        df.attrs["label"] = french_mod.LABEL
        return clip_heldout(df, self.settings)

    def french_factors(self) -> pd.DataFrame:
        df = self.cache.read(NS_FRENCH, "factors")
        df.attrs["label"] = french_mod.LABEL
        return clip_heldout(df, self.settings)

    def french_daily(self, key: str, include_heldout: bool | None = None) -> pd.DataFrame:
        """Daily French table (``industries_10_daily`` or ``factors_daily``), decimal returns."""
        df = self.cache.read(NS_FRENCH, key)
        df.attrs["label"] = french_mod.LABEL
        return clip_heldout(df, self.settings, include_heldout)
