"""Monthly rebalance job for the paper account.

Timeline (matches the backtest: signal at the month-end close, fill at the
next trading day's close):

1. ``plan``      after the month-end close: compute target weights from the
                 latest month-end prices, compare with paper positions, and
                 write the intended orders to ``<state_dir>/plan_<date>.json``.
2. ``submit``    the next trading day before the cutoff (default 15:45 ET):
                 send each order as a market-on-close order. Needs
                 ``paper.dry_run: false`` AND ``--execute``.
3. ``reconcile`` after that close: record fills and compare each fill price
                 with the close the backtest would have used (slippage).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import PROJECT_ROOT
from src.paper.safety import PaperSafetyError, check_can_submit
from src.strategies import build_strategy
from src.strategies.base import month_end_dates

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class PlannedOrder:
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    est_price: float
    est_notional: float
    current_qty: float
    target_qty: float
    target_weight: float


@dataclass
class RebalancePlan:
    strategy: str
    signal_date: str
    created_at: str
    equity: float
    cash_buffer: float
    targets: dict[str, float]
    orders: list[PlannedOrder]
    note: str
    prices: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RebalancePlan":
        d = json.loads(text)
        d["orders"] = [PlannedOrder(**o) for o in d["orders"]]
        return cls(**d)

    def describe(self) -> str:
        lines = [f"Strategy {self.strategy} · signal {self.signal_date} · equity ${self.equity:,.2f} · {self.note}",
                 "Targets: " + (", ".join(f"{k} {v:.1%}" for k, v in self.targets.items()) or "none")]
        if not self.orders:
            lines.append("No orders needed.")
        for o in self.orders:
            lines.append(f"  {o.side.upper():4s} {o.qty:>8g} {o.symbol:<5s} ~${o.est_notional:>12,.2f} "
                         f"(now {o.current_qty:g} -> {o.target_qty:g}, target {o.target_weight:.1%})")
        return "\n".join(lines)


def state_dir(settings: dict) -> Path:
    p = Path(settings["paper"]["state_dir"])
    p = p if p.is_absolute() else PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------- planning

def latest_targets(strategy, prices: pd.DataFrame, holding_nothing: bool):
    """(signal_date, weights or None, note) for the latest complete month-end."""
    weights = strategy.generate(prices)
    month_ends = month_end_dates(prices.index)
    if len(month_ends) == 0:
        raise ValueError("No complete month in the price data")
    last = month_ends[-1]
    if last in weights.index:
        row = weights.loc[last]
        return last, {k: float(v) for k, v in row.items() if v > 0}, "rebalance signal"
    if holding_nothing and len(weights):
        row = weights.iloc[-1]
        return last, {k: float(v) for k, v in row.items() if v > 0}, \
            f"initial funding from the standing target set on {weights.index[-1].date()}"
    return last, None, "no rebalance signal this month: hold current positions"


def plan_orders(targets: dict[str, float], positions: dict[str, float], prices: dict[str, float], equity: float,
                cash_buffer: float = 0.005, min_trade_value: float = 25.0, whole_shares: bool = True
                ) -> list[PlannedOrder]:
    """Orders that move ``positions`` to ``targets``; sells first, then buys."""
    investable = equity * (1.0 - cash_buffer)
    orders = []
    for sym in sorted(set(targets) | {s for s, q in positions.items() if q}):
        price = prices.get(sym)
        if price is None or not math.isfinite(price) or price <= 0:
            raise ValueError(f"No usable price for {sym}")
        w = targets.get(sym, 0.0)
        raw_qty = w * investable / price
        target_qty = float(math.floor(raw_qty + 1e-9)) if whole_shares else round(raw_qty, 6)
        current = float(positions.get(sym, 0.0))
        delta = target_qty - current
        closing = target_qty == 0 and current > 0
        if delta == 0 or (abs(delta) * price < min_trade_value and not closing):
            continue
        orders.append(PlannedOrder(sym, "buy" if delta > 0 else "sell", abs(delta), price, abs(delta) * price,
                                   current, target_qty, w))
    return sorted(orders, key=lambda o: (o.side != "sell", o.symbol))


def make_plan(settings: dict, store, broker, today: pd.Timestamp | None = None) -> RebalancePlan:
    cfg = settings["paper"]
    name = cfg.get("strategy")
    if not name:
        raise PaperSafetyError("No strategy chosen: set paper.strategy in config/settings.yaml after validation.")
    strategy = build_strategy(name, settings)
    adj, raw = store.signal_data(strategy.symbols(), purpose=f"paper plan for {name}")
    today = pd.Timestamp(today or datetime.now(ET).date())
    age = (today - adj.index[-1]).days
    if age > cfg.get("max_data_age_days", 5):
        raise PaperSafetyError(f"Newest price is {adj.index[-1].date()} ({age} days old); run the data download first.")
    positions = broker.positions()
    held = {s: q for s, q in positions.items() if q}
    signal_date, targets, note = latest_targets(strategy, adj, holding_nothing=not held)
    account = broker.account()
    if account.trading_blocked:
        raise PaperSafetyError("The paper account reports trading_blocked.")
    if targets is None:
        orders, targets_out = [], {}
    else:
        prices = {s: float(raw.loc[signal_date, s]) for s in targets}
        extra = {s for s in held if s not in prices}
        if extra:
            prices.update({s: p for s, p in broker.position_prices().items() if s in extra})
        orders = plan_orders(targets, positions, prices, account.equity, cfg["cash_buffer"], cfg["min_trade_value"],
                             whole_shares=not cfg.get("fractional_shares", False))
        targets_out = targets
    return RebalancePlan(
        strategy=name, signal_date=str(signal_date.date()), created_at=datetime.now(ET).isoformat(timespec="seconds"),
        equity=account.equity, cash_buffer=cfg["cash_buffer"], targets=targets_out, orders=orders, note=note,
        prices={o.symbol: o.est_price for o in orders},
    )


def save_plan(plan: RebalancePlan, settings: dict) -> Path:
    path = state_dir(settings) / f"plan_{plan.strategy}_{plan.signal_date}.json"
    path.write_text(plan.to_json(), encoding="utf-8")
    return path


def latest_plan_path(settings: dict, strategy: str) -> Path:
    plans = sorted(state_dir(settings).glob(f"plan_{strategy}_*.json"))
    if not plans:
        raise FileNotFoundError("No saved plan; run `python -m src.paper plan` first.")
    return plans[-1]


# --------------------------------------------------------------------- submission

def check_submission_window(plan: RebalancePlan, clock, now_et: datetime, cutoff: str) -> None:
    """Market open, before the cutoff, and after the signal date (next-day fill)."""
    if not getattr(clock, "is_open", False):
        raise PaperSafetyError("The market is closed; submit on the next trading day before the cutoff.")
    hh, mm = (int(x) for x in cutoff.split(":"))
    if now_et.time() > time(hh, mm):
        raise PaperSafetyError(f"It is past the {cutoff} ET cutoff for market-on-close orders.")
    if now_et.date() <= pd.Timestamp(plan.signal_date).date():
        raise PaperSafetyError("Orders fill on the trading day AFTER the signal date, as in the backtest.")


def submit_plan(plan: RebalancePlan, broker, settings: dict, execute: bool,
                now_et: datetime | None = None, force: bool = False) -> list[dict]:
    """Send the plan's orders as market-on-close orders. Raises PaperSafetyError in dry run."""
    check_can_submit(settings, execute)
    now_et = now_et or datetime.now(ET)
    check_submission_window(plan, broker.clock(), now_et, settings["paper"]["submit_cutoff_et"])
    record = state_dir(settings) / f"submitted_{plan.strategy}_{plan.signal_date}.json"
    if record.exists() and not force:
        raise PaperSafetyError(f"Orders for {plan.signal_date} were already submitted ({record.name}).")
    tif = settings["paper"]["time_in_force"]
    results = []
    for o in plan.orders:
        coid = f"{plan.strategy}-{plan.signal_date}-{o.symbol}-{o.side}"[:48]
        res = broker.submit_market_order(o.symbol, o.qty, o.side, tif, coid)
        log.info("Submitted %s %s %s (%s): %s", o.side, o.qty, o.symbol, tif, res.status)
        results.append({**asdict(res), "est_price": o.est_price})
    record.write_text(json.dumps({"plan": json.loads(plan.to_json()), "orders": results}, indent=2), encoding="utf-8")
    return results


# --------------------------------------------------------------------- reconciliation

def slippage_bps(side: str, fill_price: float, reference_price: float) -> float:
    """Positive = worse than the backtest's assumed price (paid more / received less)."""
    if not reference_price or not fill_price:
        return float("nan")
    sign = 1.0 if side == "buy" else -1.0
    return sign * (fill_price - reference_price) / reference_price * 1e4


def reconcile(broker, submission: dict, closes: pd.DataFrame | None) -> pd.DataFrame:
    """Refresh each submitted order and compare fills with that day's close."""
    rows = []
    for o in submission["orders"]:
        cur = broker.order(o["order_id"])
        fill_date = None
        if cur.filled_at:
            ts = pd.Timestamp(cur.filled_at)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            fill_date = ts.tz_convert(ET).date()
        ref = float("nan")
        if closes is not None and fill_date is not None and cur.symbol in closes.columns:
            ts = pd.Timestamp(fill_date)
            if ts in closes.index:
                ref = float(closes.loc[ts, cur.symbol])
        rows.append({
            "strategy": submission["plan"]["strategy"], "signal_date": submission["plan"]["signal_date"],
            "order_id": cur.order_id, "symbol": cur.symbol, "side": cur.side, "qty": cur.qty, "status": cur.status,
            "filled_qty": cur.filled_qty, "fill_price": cur.filled_avg_price, "fill_date": fill_date,
            "backtest_price": ref, "slippage_bps": slippage_bps(cur.side, cur.filled_avg_price or 0.0, ref),
            "plan_price": o.get("est_price"),
        })
    return pd.DataFrame(rows)


def append_fills(df: pd.DataFrame, settings: dict) -> Path:
    path = state_dir(settings) / "fills.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old["order_id"].isin(df["order_id"])], df], ignore_index=True)
    df.to_csv(path, index=False)
    return path
