"""Build engine inputs (MarketData) from the DataStore.

The cash symbol (BIL by default) is replaced by a spliced series: before
BIL's first return, cash earns the FRED 3-month T-bill rate, modeled as a
$1-NAV fund that distributes its interest daily; afterwards it is BIL itself.
``MarketData.notes`` records the splice date so reports can label it.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.engine import MarketData
from src.config import instruments
from src.data.store import DataStore


def dividend_yields(div_cash: pd.DataFrame, raw_close: pd.DataFrame) -> pd.DataFrame:
    """dividend / (ex-date close + dividend): the share of value paid out on each ex-date."""
    y = div_cash / (raw_close + div_cash)
    return y.where(div_cash > 0, 0.0).fillna(0.0)


def build_market_data(store: DataStore, symbols: list[str], include_heldout: bool | None = None) -> MarketData:
    settings = store.settings
    cash_sym = settings["cash"]["etf"]
    etfs = [s for s in symbols if s != cash_sym]
    if etfs:
        prices = store.adjusted_closes(etfs, include_heldout)
        raw = store.raw_closes(etfs, include_heldout)
        yields = dividend_yields(store.dividends(etfs, include_heldout), raw)
    else:
        rf_index = store.risk_free_returns(include_heldout).index
        prices = pd.DataFrame(index=rf_index)
        raw = pd.DataFrame(index=rf_index)
        yields = pd.DataFrame(index=rf_index)
    notes: dict[str, str] = {}

    if cash_sym in symbols:
        cash = store.cash_returns(include_heldout).reindex(prices.index)
        r = cash["return"].fillna(0.0)
        prices[cash_sym] = (1.0 + r).cumprod()
        synthetic = cash["source"] != cash_sym
        if store.cache.exists("tiingo", cash_sym):
            etf_raw = store.raw_closes([cash_sym], include_heldout)[cash_sym].reindex(prices.index)
            etf_div = store.dividends([cash_sym], include_heldout)[cash_sym].reindex(prices.index).fillna(0.0)
            etf_y = dividend_yields(etf_div.to_frame(), etf_raw.to_frame())[cash_sym]
        else:
            etf_raw = pd.Series(float("nan"), index=prices.index)
            etf_y = pd.Series(0.0, index=prices.index)
        raw[cash_sym] = etf_raw.where(~synthetic)
        yields[cash_sym] = (r / (1.0 + r)).where(synthetic, etf_y)
        if synthetic.any() and (~synthetic).any():
            switch = cash.index[~synthetic][0]
            notes[cash_sym] = f"FRED 3-month T-bill rate before {switch.date()}, {cash_sym} total return from then on"
        elif synthetic.all():
            notes[cash_sym] = "FRED 3-month T-bill rate (no ETF data)"

    characters = {s: i.tax_character for s, i in instruments(settings, include_disabled=True).items()}
    return MarketData(
        prices=prices[symbols],
        risk_free=store.risk_free_returns(include_heldout),
        div_yield=yields[symbols],
        raw_close=raw[symbols],
        tax_character=characters,
        notes=notes,
    )


def french_market_data(industries: pd.DataFrame, factors: pd.DataFrame,
                       cash_symbol: str = "RF", market_symbol: str = "MKT") -> MarketData:
    """MarketData for the long-history sanity check from French daily returns.

    Each industry becomes a total-return index; ``market_symbol`` is the
    French market (Mkt-RF + RF) and ``cash_symbol`` compounds the French
    risk-free rate. All series are GROSS and NON-INVESTABLE.
    """
    common = industries.index.intersection(factors.index)
    ind = industries.loc[common]
    fac = factors.loc[common]
    rets = ind.copy()
    rets[market_symbol] = fac["Mkt-RF"] + fac["RF"]
    rets[cash_symbol] = fac["RF"]
    prices = (1.0 + rets).cumprod()
    note = "GROSS, NON-INVESTABLE Kenneth French daily portfolios; no fees, costs or taxes"
    return MarketData(
        prices=prices,
        risk_free=fac["RF"].rename("risk_free"),
        div_yield=None,
        raw_close=None,
        tax_character={},
        notes={"universe": note},
    )


def build_french_market_data(store: DataStore, include_heldout: bool | None = None) -> MarketData:
    cfg = store.settings["french_sanity"]
    if cfg.get("industries", 10) != 10 or cfg.get("frequency", "daily") != "daily":
        raise ValueError("Only the daily 10-industry French data is configured")
    return french_market_data(
        store.french_daily("industries_10_daily", include_heldout),
        store.french_daily("factors_daily", include_heldout),
        cfg["cash_symbol"], cfg["market_symbol"],
    )
