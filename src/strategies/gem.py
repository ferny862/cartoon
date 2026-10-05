"""Strategy 2: Global Equities Momentum (Gary Antonacci).

Each month compare 12-month total returns. If US stocks beat Treasury bills
(absolute momentum), hold whichever of US or non-US stocks has the higher
12-month return (relative momentum). Otherwise hold aggregate bonds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, trailing_return


class GEM(Strategy):
    name = "gem"

    def __init__(self, us: str = "SPY", intl: str = "EFA", bonds: str = "AGG", cash: str = "BIL",
                 lookback_months: int = 12, name: str | None = None):
        super().__init__(name, us=us, intl=intl, bonds=bonds, cash=cash, lookback_months=lookback_months)
        self.us, self.intl, self.bonds, self.cash, self.lookback = us, intl, bonds, cash, lookback_months

    def symbols(self) -> list[str]:
        return [self.us, self.intl, self.bonds, self.cash]

    def target(self, history: pd.DataFrame, date: pd.Timestamp):
        r_us = trailing_return(history[self.us], self.lookback)
        r_intl = trailing_return(history[self.intl], self.lookback)
        r_cash = trailing_return(history[self.cash], self.lookback)
        if not all(np.isfinite([r_us, r_intl, r_cash])) or pd.isna(history[self.bonds].iloc[-1]):
            return None
        if r_us > r_cash:
            return {self.us: 1.0} if r_us >= r_intl else {self.intl: 1.0}
        return {self.bonds: 1.0}
