"""Strategy 3: Faber Global Tactical Asset Allocation, 5-asset version.

Equal sleeves in each asset. A sleeve holds its asset only when the asset's
month-end close is above its 10-month SMA; otherwise that sleeve holds cash.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy, above_sma


class GTAA(Strategy):
    name = "gtaa5"

    def __init__(self, assets: list[str] | tuple[str, ...] = ("SPY", "EFA", "IEF", "VNQ", "DBC"),
                 cash: str = "BIL", sma_months: int = 10, name: str | None = None):
        assets = list(assets)
        super().__init__(name, assets=assets, cash=cash, sma_months=sma_months)
        self.assets, self.cash, self.sma_months = assets, cash, sma_months

    def symbols(self) -> list[str]:
        return self.assets + [self.cash]

    def target(self, history: pd.DataFrame, date: pd.Timestamp):
        if pd.isna(history[self.cash].iloc[-1]):
            return None
        sleeve = 1.0 / len(self.assets)
        w: dict[str, float] = {}
        for a in self.assets:
            up = above_sma(history[a], self.sma_months)
            if up is None:
                return None                     # wait until every asset has enough history
            key = a if up else self.cash
            w[key] = w.get(key, 0.0) + sleeve
        return w
