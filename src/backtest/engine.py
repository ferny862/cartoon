"""Transparent daily backtest engine.

Converts target weights on signal dates into trades, holdings, cash and
portfolio value.

Conventions
-----------
* Prices are total-return (dividend- and split-adjusted) closes, so
  dividends are reinvested in the paying fund on the ex-date at no cost.
* A target set on signal date ``s`` is executed at the close
  ``execution_lag`` trading days later. The engine only ever reads prices up
  to the current day, so it cannot look ahead.
* Each execution trades every position to its target weight of the portfolio
  value net of that execution's trading costs. Trades smaller than
  ``min_trade_value`` dollars are skipped.
* In taxable mode, every trade and reinvested dividend goes through a FIFO
  tax-lot ledger. Each year's tax is computed at year end and paid on the
  first trading day on or after the configured payment date (default
  April 15) of the following year by selling holdings pro rata.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.taxes import TaxLedger, TaxRates, wash_sale_flags

log = logging.getLogger(__name__)

CASH = "CASH"   # residual uninvested cash (earns nothing; normally ~0)


@dataclass
class MarketData:
    """Aligned daily inputs for the engine."""

    prices: pd.DataFrame                    # total-return closes
    risk_free: pd.Series                    # per-period risk-free return (for metrics)
    div_yield: pd.DataFrame | None = None   # dividend / (ex-date close + dividend); 0 on other days
    raw_close: pd.DataFrame | None = None   # unadjusted closes, for share counts
    tax_character: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        idx = self.prices.index
        if not idx.is_monotonic_increasing or idx.has_duplicates:
            raise ValueError("prices index must be sorted and unique")
        self.risk_free = self.risk_free.reindex(idx)
        if self.div_yield is not None:
            self.div_yield = self.div_yield.reindex(idx).fillna(0.0)
        if self.raw_close is not None:
            self.raw_close = self.raw_close.reindex(idx)


@dataclass(frozen=True)
class TaxSettings:
    rates: TaxRates
    payment_month: int = 4
    payment_day: int = 15
    wash_sale_groups: tuple[tuple[str, ...], ...] = ()
    wash_sale_window_days: int = 30

    @classmethod
    def from_settings(cls, settings: dict) -> "TaxSettings":
        t = settings["taxes"]
        return cls(
            rates=TaxRates.from_settings(settings),
            wash_sale_groups=tuple(tuple(g) for g in t.get("substantially_identical", [])),
            wash_sale_window_days=int(t.get("wash_sale_window_days", 30)),
        )


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series                 # daily portfolio value (after costs, and after taxes paid if taxable)
    weights: pd.DataFrame             # daily actual weights, including CASH
    trades: pd.DataFrame              # one row per symbol traded
    costs: pd.Series                  # daily trading costs
    initial_capital: float
    target_weights: pd.DataFrame      # targets indexed by execution date
    taxable: bool = False
    taxes_paid: pd.Series | None = None
    tax_years: pd.DataFrame | None = None
    realized: pd.DataFrame | None = None
    wash_sales: pd.DataFrame | None = None
    unpaid_tax_at_end: float = 0.0
    after_tax_value_holding: float | None = None
    after_tax_value_liquidated: float | None = None
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change(fill_method=None).iloc[1:]

    @property
    def start(self) -> pd.Timestamp:
        return self.equity.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.equity.index[-1]


def execution_dates(signal_dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, lag: int) -> pd.Series:
    """Map each signal date to its execution date ``lag`` trading days later.

    Signals whose execution date would fall after the end of the data are dropped.
    """
    pos = calendar.get_indexer(signal_dates)
    if (pos < 0).any():
        bad = signal_dates[pos < 0]
        raise ValueError(f"Signal dates not in the trading calendar: {list(bad[:5])}")
    exec_pos = pos + lag
    keep = exec_pos < len(calendar)
    if (~keep).any():
        log.info("Dropping %d signal(s) whose execution date is beyond the data", int((~keep).sum()))
    return pd.Series(calendar[exec_pos[keep]], index=signal_dates[keep])


def _validate_weights(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in weights.columns if c not in prices.columns]
    if missing:
        raise ValueError(f"No prices for symbols {missing}")
    w = weights.sort_index().fillna(0.0).astype(float)
    if w.index.has_duplicates:
        raise ValueError("Duplicate signal dates in weights")
    if (w < -1e-12).to_numpy().any():
        raise ValueError("Negative weights are not allowed (long-only)")
    sums = w.sum(axis=1)
    if (sums > 1 + 1e-9).any():
        raise ValueError(f"Weights sum above 1 on {list(sums[sums > 1 + 1e-9].index[:5])}")
    return w.clip(lower=0.0)


class _Simulator:
    def __init__(self, market: MarketData, symbols: list[str], costs: CostModel, ledger: TaxLedger | None,
                 allow_fractional: bool, min_trade_value: float):
        self.dates = market.prices.index
        self.symbols = symbols
        self.P = market.prices[symbols].to_numpy(dtype=float)
        self.R = market.raw_close[symbols].to_numpy(dtype=float) if market.raw_close is not None else None
        self.Y = market.div_yield[symbols].to_numpy(dtype=float) if (
            market.div_yield is not None and ledger is not None) else None
        self.costs = costs
        self.ledger = ledger
        self.allow_fractional = allow_fractional
        self.min_trade_value = min_trade_value
        self.units = np.zeros(len(symbols))
        self.cash = 0.0
        self.trades: list[dict] = []

    # -- helpers ---------------------------------------------------------------
    def value(self, t: int) -> float:
        held = self.units != 0
        px = self.P[t]
        if np.isnan(px[held]).any():
            sym = [s for s, h, p in zip(self.symbols, held, px) if h and np.isnan(p)]
            raise ValueError(f"No price for held position(s) {sym} on {self.dates[t].date()}")
        return float(np.dot(self.units[held], px[held]) + self.cash)

    def _shares(self, t: int, i: int, notional: float) -> float:
        raw = self.R[t, i] if self.R is not None else np.nan
        price = raw if (raw == raw and raw > 0) else self.P[t, i]
        return abs(notional) / price

    def _trade_costs(self, t: int, trade: np.ndarray) -> np.ndarray:
        return np.array([self.costs.trade_cost(x, self._shares(t, i, x)).total if x else 0.0
                         for i, x in enumerate(trade)])

    def _execute(self, t: int, trade: np.ndarray, cost: np.ndarray, reason: str) -> None:
        date = self.dates[t]
        px = self.P[t]
        order = sorted(range(len(trade)), key=lambda i: trade[i])  # sells first
        for i in order:
            x = trade[i]
            if x == 0:
                continue
            du = x / px[i]
            if self.ledger is not None:
                if x < 0:
                    self.ledger.sell(self.symbols[i], date, -du, -x - cost[i])
                else:
                    self.ledger.buy(self.symbols[i], date, du, x + cost[i])
            self.units[i] += du
            if abs(self.units[i] * px[i]) < 1e-9:
                self.units[i] = 0.0
            self.trades.append({"date": date, "symbol": self.symbols[i], "units": du, "price": px[i],
                                "notional": x, "cost": cost[i], "reason": reason})
        self.cash -= float(trade.sum() + cost.sum())

    # -- events ----------------------------------------------------------------
    def dividends(self, t: int) -> None:
        if self.Y is None:
            return
        y = self.Y[t]
        for i in np.nonzero((y > 0) & (self.units > 0))[0]:
            amount = self.units[i] * self.P[t, i] * y[i]
            self.ledger.dividend(self.symbols[i], self.dates[t], amount, y[i])

    def rebalance(self, t: int, w: np.ndarray, reason: str = "rebalance") -> float:
        px = self.P[t]
        needed = w > 0
        if np.isnan(px[needed]).any():
            sym = [s for s, n, p in zip(self.symbols, needed, px) if n and np.isnan(p)]
            raise ValueError(f"Target includes {sym} with no price on {self.dates[t].date()} "
                             f"(before inception or missing data)")
        V = self.value(t)
        cur = np.where(self.units != 0, self.units * np.nan_to_num(px), 0.0)
        cost_total = 0.0
        trade = np.zeros_like(w)
        cost = np.zeros_like(w)
        for _ in range(20):
            target = w * (V - cost_total)
            if not self.allow_fractional:
                for i in np.nonzero(needed)[0]:
                    share_px = target[i] / self._shares(t, i, target[i]) if target[i] > 0 else 0.0
                    target[i] = math.floor(self._shares(t, i, target[i]) + 1e-9) * share_px if share_px else 0.0
            trade = target - cur
            trade[np.abs(trade) < self.min_trade_value] = 0.0
            cost = self._trade_costs(t, trade)
            if abs(cost.sum() - cost_total) < 1e-10:
                break
            cost_total = float(cost.sum())
        self._execute(t, trade, cost, reason)
        return float(cost.sum())

    def raise_cash(self, t: int, amount: float) -> tuple[float, float]:
        """Sell holdings pro rata so that ``amount`` can be paid. Returns (paid, cost)."""
        from_cash = min(max(self.cash, 0.0), amount)
        self.cash -= from_cash
        need = amount - from_cash
        if need <= 1e-9:
            return amount, 0.0
        px = np.nan_to_num(self.P[t])
        pos_val = self.units * px
        held_value = float(pos_val.sum())
        if held_value <= 0:
            return from_cash, 0.0
        frac = min(1.0, need / held_value)
        trade = cost = np.zeros_like(pos_val)
        for _ in range(20):
            trade = -pos_val * frac
            cost = self._trade_costs(t, trade)
            new_frac = min(1.0, (need + cost.sum()) / held_value)
            if abs(new_frac - frac) < 1e-12:
                break
            frac = new_frac
        before = self.cash
        self._execute(t, trade, cost, "tax_payment")
        raised = self.cash - before            # net proceeds
        paid = from_cash + min(need, raised)
        self.cash -= min(need, raised)
        return paid, float(cost.sum())


def run_backtest(
    weights: pd.DataFrame,
    market: MarketData,
    costs: CostModel | None = None,
    initial_capital: float = 100_000.0,
    execution_lag: int = 1,
    tax: TaxSettings | None = None,
    allow_fractional: bool = True,
    min_trade_value: float = 1.0,
    name: str = "",
) -> BacktestResult:
    """Simulate a long-only portfolio that trades to ``weights`` after each signal.

    ``weights`` is indexed by signal date with one column per symbol; rows sum
    to at most 1 (any remainder stays as uninvested cash).
    """
    costs = costs or CostModel(spread_slippage_bps=0.0)
    w = _validate_weights(weights, market.prices)
    symbols = list(w.columns)
    calendar = market.prices.index
    exec_map = execution_dates(pd.DatetimeIndex(w.index), calendar, execution_lag)
    if exec_map.empty:
        raise ValueError("No executable signals in the data range")
    targets = w.loc[exec_map.index].set_axis(pd.DatetimeIndex(exec_map.values, name="date"))
    targets = targets[~targets.index.duplicated(keep="last")]
    target_rows = {calendar.get_loc(d): targets.loc[d].to_numpy(dtype=float) for d in targets.index}

    ledger = TaxLedger(tax.rates, market.tax_character) if tax is not None else None
    sim = _Simulator(market, symbols, costs, ledger, allow_fractional, min_trade_value)

    start = calendar.get_loc(targets.index[0])
    n = len(calendar) - start
    equity = np.empty(n)
    cost_series = np.zeros(n)
    tax_paid = np.zeros(n)
    wts = np.zeros((n, len(symbols) + 1))
    pending: list[tuple[pd.Timestamp, float, int]] = []   # (due date, amount, tax year)

    sim.cash = float(initial_capital)
    for k, t in enumerate(range(start, len(calendar))):
        date = calendar[t]
        if ledger is not None:
            if k > 0 and date.year != calendar[t - 1].year:
                year = calendar[t - 1].year
                due_amount = ledger.close_year(year).total
                due = pd.Timestamp(year=year + 1, month=tax.payment_month, day=tax.payment_day)
                if due_amount > 0:
                    pending.append((due, due_amount, year))
            sim.dividends(t)
            while pending and date >= pending[0][0]:
                _, amount, year = pending.pop(0)
                paid, c = sim.raise_cash(t, amount)
                tax_paid[k] += paid
                cost_series[k] += c
                if paid < amount - 1e-6:
                    log.warning("Portfolio could not cover the %d tax bill on %s", year, date.date())
        if t in target_rows:
            cost_series[k] += sim.rebalance(t, target_rows[t])
        V = sim.value(t)
        equity[k] = V
        if V > 0:
            px = np.nan_to_num(sim.P[t])
            wts[k, :-1] = sim.units * px / V
            wts[k, -1] = sim.cash / V

    idx = calendar[start:]
    result = BacktestResult(
        name=name,
        equity=pd.Series(equity, index=idx, name=name or "equity"),
        weights=pd.DataFrame(wts, index=idx, columns=symbols + [CASH]),
        trades=pd.DataFrame(sim.trades, columns=["date", "symbol", "units", "price", "notional", "cost", "reason"]),
        costs=pd.Series(cost_series, index=idx, name="costs"),
        initial_capital=float(initial_capital),
        target_weights=targets,
        taxable=ledger is not None,
        notes=dict(market.notes),
    )
    if ledger is not None:
        _finish_taxable(result, sim, ledger, tax, pending, tax_paid, idx)
    return result


def _finish_taxable(result: BacktestResult, sim: _Simulator, ledger: TaxLedger, tax: TaxSettings,
                    pending: list, tax_paid: np.ndarray, idx: pd.DatetimeIndex) -> None:
    """After-tax values at the end: still holding, and fully liquidated."""
    t_end = len(sim.dates) - 1
    final_year = sim.dates[t_end].year
    unpaid_prior = sum(a for _, a, _ in pending)
    V = result.equity.iloc[-1]

    accrued = ledger.preview_year(final_year)
    result.after_tax_value_holding = V - unpaid_prior - accrued.total

    liq = ledger.copy()
    px = np.nan_to_num(sim.P[t_end])
    proceeds = sim.cash
    for i, u in enumerate(sim.units):
        if u > 0:
            gross = u * px[i]
            c = sim.costs.trade_cost(-gross, sim._shares(t_end, i, gross)).total
            liq.sell(sim.symbols[i], sim.dates[t_end], u, gross - c)
            proceeds += gross - c
    final_tax = liq.preview_year(final_year).total
    result.after_tax_value_liquidated = proceeds - unpaid_prior - final_tax
    result.unpaid_tax_at_end = unpaid_prior + accrued.total

    years = ledger.years_frame()
    partial = pd.DataFrame([{**accrued.as_dict(), "open_at_end": True}])
    years = pd.concat([years.assign(open_at_end=False), partial], ignore_index=True) if len(years) else partial
    result.tax_years = years
    result.taxes_paid = pd.Series(tax_paid, index=idx, name="taxes_paid")
    result.realized = ledger.realized_frame()
    result.wash_sales = wash_sale_flags(result.realized, ledger.purchases_frame(),
                                        [list(g) for g in tax.wash_sale_groups], tax.wash_sale_window_days)
