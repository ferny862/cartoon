"""Strategy 1: Faber trend filter.

Hold the risk asset (SPY) when its month-end close is above its 10-month
simple moving average; otherwise hold cash (BIL, or the FRED T-bill series
before BIL existed).
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy, above_sma


class TrendFilter(Strategy):
    name = "trend_faber"

    def __init__(self, risk_asset: str = "SPY", cash: str = "BIL", sma_months: int = 10, name: str | None = None):
        super().__init__(name, risk_asset=risk_asset, cash=cash, sma_months=sma_months)
        self.risk_asset, self.cash, self.sma_months = risk_asset, cash, sma_months

    def symbols(self) -> list[str]:
        return [self.risk_asset, self.cash]

    def target(self, history: pd.DataFrame, date: pd.Timestamp):
        up = above_sma(history[self.risk_asset], self.sma_months)
        if up is None or pd.isna(history[self.cash].iloc[-1]):
            return None
        return {self.risk_asset: 1.0} if up else {self.cash: 1.0}
