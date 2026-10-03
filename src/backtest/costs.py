"""Transaction cost model.

Every trade pays a bid-ask spread plus slippage (in basis points of traded
notional, per side) and any commission. Sales additionally pay the SEC
Section 31 fee (per dollar of proceeds) and the FINRA Trading Activity Fee
(per share, capped per trade).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CostBreakdown:
    spread: float = 0.0
    commission: float = 0.0
    sec_fee: float = 0.0
    finra_taf: float = 0.0

    @property
    def total(self) -> float:
        return self.spread + self.commission + self.sec_fee + self.finra_taf


@dataclass(frozen=True)
class CostModel:
    spread_slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    sec_fee_per_million: float = 0.0
    finra_taf_per_share: float = 0.0
    finra_taf_max_per_trade: float = float("inf")

    @classmethod
    def from_settings(cls, settings: dict, spread_slippage_bps: float | None = None) -> "CostModel":
        c = settings["costs"]
        return cls(
            spread_slippage_bps=c["spread_slippage_bps"] if spread_slippage_bps is None else spread_slippage_bps,
            commission_per_trade=float(c.get("commission_per_trade") or 0.0),
            sec_fee_per_million=float(c.get("sec_fee_per_million") or 0.0),
            finra_taf_per_share=float(c.get("finra_taf_per_share") or 0.0),
            finra_taf_max_per_trade=float(c.get("finra_taf_max_per_trade") or float("inf")),
        )

    def with_spread(self, bps: float) -> "CostModel":
        return replace(self, spread_slippage_bps=bps)

    def trade_cost(self, notional: float, shares: float | None = None) -> CostBreakdown:
        """Cost of one trade. ``notional`` > 0 is a buy, < 0 a sale.

        ``shares`` is the number of shares traded (for the per-share FINRA fee);
        it defaults to zero if unknown.
        """
        size = abs(notional)
        if size == 0:
            return CostBreakdown()
        spread = size * self.spread_slippage_bps / 1e4
        commission = self.commission_per_trade
        sec = taf = 0.0
        if notional < 0:
            sec = size * self.sec_fee_per_million / 1e6
            taf = min(abs(shares or 0.0) * self.finra_taf_per_share, self.finra_taf_max_per_trade)
        return CostBreakdown(spread, commission, sec, taf)
