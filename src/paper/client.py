"""Thin wrapper around the official alpaca-py TradingClient, paper endpoint only.

The wrapper exposes only what the rebalance job needs and converts Alpaca's
string-typed fields to numbers. Tests inject a fake trading client; the real
one is imported lazily so the package works without alpaca-py installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.config import get_secret
from src.paper.safety import PAPER_BASE_URL, PaperSafetyError, check_paper_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    status: str
    trading_blocked: bool


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float
    filled_avg_price: float | None
    filled_at: str | None
    submitted_at: str | None


def _real_trading_client(api_key: str, secret_key: str):
    from alpaca.trading.client import TradingClient  # lazy: optional dependency

    return TradingClient(api_key, secret_key, paper=True, url_override=PAPER_BASE_URL)


def _f(x) -> float:
    return float(x) if x not in (None, "") else 0.0


def _enum(x) -> str:
    return str(getattr(x, "value", x))


class PaperBroker:
    """Paper-only broker. Construction runs every safety check."""

    def __init__(self, settings: dict, environ: dict | None = None,
                 client_factory: Callable[[str, str], Any] | None = None):
        url = check_paper_settings(settings, environ)
        api_key = get_secret("ALPACA_API_KEY")
        secret = get_secret("ALPACA_SECRET_KEY")
        self._client = (client_factory or _real_trading_client)(api_key, secret)
        actual = str(getattr(self._client, "_base_url", url)).rstrip("/")
        if actual != url:
            raise PaperSafetyError(f"Broker client points at {actual!r}, not the paper endpoint; refusing to continue.")
        log.info("Connected to Alpaca paper trading at %s", url)

    def __repr__(self) -> str:
        return f"PaperBroker(base_url={PAPER_BASE_URL!r})"

    def account(self) -> AccountSnapshot:
        a = self._client.get_account()
        return AccountSnapshot(_f(a.equity), _f(a.cash), _f(a.buying_power), _enum(a.status),
                               bool(getattr(a, "trading_blocked", False)))

    def positions(self) -> dict[str, float]:
        """Shares held per symbol."""
        return {p.symbol: _f(p.qty) for p in self._client.get_all_positions()}

    def position_prices(self) -> dict[str, float]:
        """Latest price Alpaca reports for each held symbol."""
        out = {}
        for p in self._client.get_all_positions():
            qty = _f(p.qty)
            price = _f(getattr(p, "current_price", None)) or (_f(p.market_value) / qty if qty else 0.0)
            out[p.symbol] = price
        return out

    def clock(self):
        return self._client.get_clock()

    def submit_market_order(self, symbol: str, qty: float, side: str, time_in_force: str,
                            client_order_id: str) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce(time_in_force),
                                 client_order_id=client_order_id)
        return self._to_result(self._client.submit_order(order_data=req))

    def order(self, order_id: str) -> OrderResult:
        return self._to_result(self._client.get_order_by_id(order_id))

    @staticmethod
    def _to_result(o) -> OrderResult:
        avg = getattr(o, "filled_avg_price", None)
        return OrderResult(
            order_id=str(o.id), client_order_id=str(getattr(o, "client_order_id", "")), symbol=o.symbol,
            side=_enum(o.side), qty=_f(o.qty), status=_enum(o.status), filled_qty=_f(getattr(o, "filled_qty", 0)),
            filled_avg_price=float(avg) if avg not in (None, "") else None,
            filled_at=str(o.filled_at) if getattr(o, "filled_at", None) else None,
            submitted_at=str(o.submitted_at) if getattr(o, "submitted_at", None) else None,
        )
