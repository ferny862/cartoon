"""Strategy 0 (benchmark): buy SPY once and hold it, dividends reinvested."""

from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class BuyAndHold(Strategy):
    name = "benchmark"

    def __init__(self, symbol: str = "SPY", name: str | None = None):
        super().__init__(name, symbol=symbol)
        self.symbol = symbol
        self._bought = False

    def symbols(self) -> list[str]:
        return [self.symbol]

    def reset(self) -> None:
        self._bought = False

    def target(self, history: pd.DataFrame, date: pd.Timestamp):
        if self._bought or pd.isna(history[self.symbol].iloc[-1]):
            return None
        self._bought = True
        return {self.symbol: 1.0}
