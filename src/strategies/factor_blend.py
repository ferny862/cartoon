"""Strategy 4: static equal-weight blend of US factor ETFs.

Equal weight in quality, momentum, value and minimum-volatility ETFs.
Rebalance at each December month-end, and at any month-end when a holding's
drifted weight differs from its target by more than ``drift_threshold``
(absolute). Drift is tracked from month-end closes since the last
rebalance signal, so it uses only past data.

Most of these ETFs start around 2013, so this strategy has a much shorter
history than the others.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Strategy


class FactorBlend(Strategy):
    name = "factor_blend"

    def __init__(self, assets: list[str] | tuple[str, ...] = ("QUAL", "MTUM", "VLUE", "USMV"),
                 drift_threshold: float = 0.05, rebalance_month: int = 12, name: str | None = None):
        assets = list(assets)
        super().__init__(name, assets=assets, drift_threshold=drift_threshold, rebalance_month=rebalance_month)
        self.assets, self.drift_threshold, self.rebalance_month = assets, drift_threshold, rebalance_month
        self._last_prices: np.ndarray | None = None
        self._last_weights: np.ndarray | None = None

    def symbols(self) -> list[str]:
        return list(self.assets)

    def reset(self) -> None:
        self._last_prices = None
        self._last_weights = None

    def drifted_weights(self, prices: np.ndarray) -> np.ndarray:
        grown = self._last_weights * prices / self._last_prices
        return grown / grown.sum()

    def target(self, history: pd.DataFrame, date: pd.Timestamp):
        px = history[self.assets].iloc[-1].to_numpy(dtype=float)
        if not np.isfinite(px).all():
            return None
        target = np.full(len(self.assets), 1.0 / len(self.assets))
        if self._last_prices is None:
            rebalance = True
        else:
            drift = np.abs(self.drifted_weights(px) - target).max()
            rebalance = date.month == self.rebalance_month or drift > self.drift_threshold
        if not rebalance:
            return None
        self._last_prices, self._last_weights = px, target
        return dict(zip(self.assets, target.tolist()))
