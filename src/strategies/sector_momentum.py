"""Strategies 5 and 6: sector momentum with a trend filter.

Each month rank the sectors by trailing total return over ``lookback_months``
months, skipping the most recent ``skip_months`` month(s), and hold the top
``top_n`` at equal weight.

* ``filter="per_sector_sma"`` (strategy 5): a held sector whose month-end
  close is not above its own 10-month SMA has its sleeve moved to cash.
* ``filter="market_sma"`` (strategy 6): the whole portfolio moves to cash
  whenever the market (SPY) is not above its 10-month SMA.

Ties in momentum are broken by the order of ``sectors`` (stable sort), so
results are deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, above_sma, trailing_return

FILTERS = ("per_sector_sma", "market_sma")


class SectorMomentum(Strategy):
    name = "sector_momentum"

    def __init__(self, sectors: list[str], cash: str = "BIL", lookback_months: int = 12, skip_months: int = 1,
                 top_n: int = 3, filter: str = "per_sector_sma", sma_months: int = 10,
                 market: str | None = None, name: str | None = None):
        if filter not in FILTERS:
            raise ValueError(f"filter must be one of {FILTERS}")
        if filter == "market_sma" and not market:
            raise ValueError("market_sma filter needs a market symbol")
        if not 0 < top_n <= len(sectors):
            raise ValueError("top_n must be between 1 and the number of sectors")
        sectors = list(sectors)
        super().__init__(name, sectors=sectors, cash=cash, lookback_months=lookback_months, skip_months=skip_months,
                         top_n=top_n, filter=filter, sma_months=sma_months, market=market)
        self.sectors, self.cash, self.lookback, self.skip = sectors, cash, lookback_months, skip_months
        self.top_n, self.filter, self.sma_months, self.market = top_n, filter, sma_months, market

    def symbols(self) -> list[str]:
        syms = self.sectors + [self.cash]
        if self.filter == "market_sma":
            syms.append(self.market)
        return syms

    def momentum(self, history: pd.DataFrame) -> pd.Series:
        return pd.Series({s: trailing_return(history[s], self.lookback, self.skip) for s in self.sectors})

    def ranked(self, history: pd.DataFrame) -> list[str] | None:
        mom = self.momentum(history)
        if mom.isna().any():
            return None                                   # wait until every sector has enough history
        order = np.argsort(-mom.to_numpy(), kind="stable")
        return [self.sectors[i] for i in order]

    def target(self, history: pd.DataFrame, date: pd.Timestamp):
        if pd.isna(history[self.cash].iloc[-1]):
            return None
        ranking = self.ranked(history)
        if ranking is None:
            return None
        top = ranking[: self.top_n]
        sleeve = 1.0 / self.top_n

        if self.filter == "market_sma":
            up = above_sma(history[self.market], self.sma_months)
            if up is None:
                return None
            return {s: sleeve for s in top} if up else {self.cash: 1.0}

        filters = [above_sma(history[s], self.sma_months) for s in top]
        if any(f is None for f in filters):
            return None
        w: dict[str, float] = {}
        for s, up in zip(top, filters):
            key = s if up else self.cash
            w[key] = w.get(key, 0.0) + sleeve
        return w
