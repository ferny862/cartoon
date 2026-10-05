"""Tax-lot model for a taxable account.

* First-in-first-out lots. A gain is long-term only if the sale date is more
  than one year after the acquisition date ("held one year or less" is
  short-term).
* Positions are tracked in total-return units. A dividend reinvested on its
  ex-date becomes a new lot (basis = dividend amount, acquired that day). To
  keep total units unchanged, existing lots are scaled down by
  ``1 - yield`` — exactly what happens in share terms when the price drops by
  the dividend and new shares are bought.
* Dividends from funds marked ``qualified`` must also pass the holding-period
  test: shares held more than 60 days in the 121-day window that starts 60
  days before the ex-date. Shares failing it are taxed as ordinary income.
* Each year short- and long-term results are netted the way Schedule D does,
  net losses are carried forward keeping their character, and federal plus
  state tax is computed at flat rates.
* Wash sales are flagged, not adjusted: the report shows losses whose
  deduction might be disallowed.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

import pandas as pd

QUALIFIED = "qualified"
NONQUALIFIED = "nonqualified"           # qualified-fund dividends failing the holding test
INCOME_CHARACTERS = ("qualified", "nonqualified", "ordinary", "ordinary_treasury", "ordinary_reit", "commodity_pool")
UNIT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TaxRates:
    short_term: float
    long_term: float
    qualified_dividend: float
    ordinary: float
    state: float
    treasury_state_exempt: bool = True
    carry_forward_losses: bool = True

    @classmethod
    def from_settings(cls, settings: dict) -> "TaxRates":
        t = settings["taxes"]
        if t.get("ordinary_income_loss_offset", 0):
            raise ValueError("ordinary_income_loss_offset is not supported (wages are not modeled); set it to 0")
        if t.get("lot_method", "fifo") != "fifo":
            raise ValueError("Only FIFO lots are supported")
        f = t["federal"]
        return cls(
            short_term=f["short_term"],
            long_term=f["long_term"],
            qualified_dividend=f["qualified_dividend"],
            ordinary=f["ordinary"],
            state=t["state"]["rate"],
            treasury_state_exempt=t["state"].get("treasury_interest_exempt", True),
            carry_forward_losses=t.get("carry_forward_losses", True),
        )


def is_long_term(acquired: pd.Timestamp, sold: pd.Timestamp) -> bool:
    """More than one year: sold after the one-year anniversary of the acquisition."""
    return pd.Timestamp(sold) > pd.Timestamp(acquired) + pd.DateOffset(years=1)


def qualified_holding_days(acquired: pd.Timestamp, disposed: pd.Timestamp, ex_date: pd.Timestamp) -> int:
    """Days held within the 121-day window starting 60 days before the ex-date.

    Counts the disposal day but not the acquisition day, as the IRS does.
    """
    ex = pd.Timestamp(ex_date)
    start = max(pd.Timestamp(acquired), ex - pd.Timedelta(days=61))
    end = min(pd.Timestamp(disposed), ex + pd.Timedelta(days=60))
    return max(0, (end - start).days)


def net_capital_gains(st_net: float, lt_net: float, carry_st: float = 0.0, carry_lt: float = 0.0):
    """Schedule D-style netting.

    ``carry_st`` / ``carry_lt`` are prior-year loss carryforwards (<= 0).
    Returns ``(taxable_st, taxable_lt, new_carry_st, new_carry_lt)``; the
    carries are <= 0 and keep their short/long character.
    """
    st = st_net + carry_st
    lt = lt_net + carry_lt
    if st >= 0 and lt >= 0:
        return st, lt, 0.0, 0.0
    if st < 0 and lt < 0:
        return 0.0, 0.0, st, lt
    combined = st + lt
    if st < 0:  # short-term loss against long-term gain
        return (0.0, combined, 0.0, 0.0) if combined >= 0 else (0.0, 0.0, combined, 0.0)
    # long-term loss against short-term gain
    return (combined, 0.0, 0.0, 0.0) if combined >= 0 else (0.0, 0.0, 0.0, combined)


@dataclass
class YearTax:
    year: int
    st_net: float
    lt_net: float
    taxable_st: float
    taxable_lt: float
    income: dict[str, float]
    federal: float
    state: float
    carry_st: float
    carry_lt: float

    @property
    def total(self) -> float:
        return self.federal + self.state

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("year", "st_net", "lt_net", "taxable_st", "taxable_lt",
                                             "federal", "state", "carry_st", "carry_lt")}
        d.update({f"income_{k}": v for k, v in self.income.items()})
        d["total"] = self.total
        return d


def compute_year_tax(
    year: int,
    st_net: float,
    lt_net: float,
    income: dict[str, float],
    rates: TaxRates,
    carry_st: float = 0.0,
    carry_lt: float = 0.0,
) -> YearTax:
    """Federal and state tax for one year at flat rates."""
    if not rates.carry_forward_losses:
        carry_st = carry_lt = 0.0
    tst, tlt, cst, clt = net_capital_gains(st_net, lt_net, carry_st, carry_lt)
    if not rates.carry_forward_losses:
        cst = clt = 0.0
    inc = {k: float(income.get(k, 0.0)) for k in INCOME_CHARACTERS}
    ordinary_like = inc["nonqualified"] + inc["ordinary"] + inc["ordinary_treasury"] + inc["ordinary_reit"] + inc["commodity_pool"]
    federal = (
        tst * rates.short_term
        + tlt * rates.long_term
        + inc["qualified"] * rates.qualified_dividend
        + ordinary_like * rates.ordinary
    )
    state_base = tst + tlt + inc["qualified"] + ordinary_like
    if rates.treasury_state_exempt:
        state_base -= inc["ordinary_treasury"]
    state = state_base * rates.state
    return YearTax(year, st_net, lt_net, tst, tlt, inc, federal, state, cst, clt)


@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float
    basis: float
    acquired: pd.Timestamp
    from_dividend: bool = False
    # (date, fraction of the lot's units sold on that date) for the holding test
    sales: list[tuple[pd.Timestamp, float]] = field(default_factory=list)


class TaxLedger:
    """FIFO lot ledger for one taxable account."""

    def __init__(self, rates: TaxRates, tax_character: dict[str, str], merge_dividend_lots_within_month: bool = True):
        self.rates = rates
        self.tax_character = tax_character
        self.merge_dividend_lots = merge_dividend_lots_within_month
        self.lots: dict[str, list[Lot]] = {}
        self._all_lots: dict[int, Lot] = {}
        self._ids = itertools.count(1)
        self.realized: list[dict] = []
        self.purchases: list[dict] = []
        self._qualified_events: list[dict] = []
        self._other_income: dict[tuple[int, str], float] = {}
        self.carry_st = 0.0
        self.carry_lt = 0.0
        self.closed_years: dict[int, YearTax] = {}

    # -- positions -----------------------------------------------------------
    def units(self, symbol: str) -> float:
        return sum(l.units for l in self.lots.get(symbol, []))

    def basis(self, symbol: str) -> float:
        return sum(l.basis for l in self.lots.get(symbol, []))

    def _new_lot(self, symbol: str, units: float, basis: float, date: pd.Timestamp, from_dividend: bool) -> Lot:
        lot = Lot(next(self._ids), symbol, units, basis, pd.Timestamp(date), from_dividend)
        self.lots.setdefault(symbol, []).append(lot)
        self._all_lots[lot.lot_id] = lot
        return lot

    def buy(self, symbol: str, date: pd.Timestamp, units: float, basis: float) -> None:
        """Record a purchase. ``basis`` includes trading costs."""
        if units <= 0:
            return
        lot = self._new_lot(symbol, units, basis, date, from_dividend=False)
        self.purchases.append({"date": pd.Timestamp(date), "symbol": symbol, "units": units, "basis": basis,
                               "lot_id": lot.lot_id, "source": "trade"})

    def sell(self, symbol: str, date: pd.Timestamp, units: float, proceeds: float) -> float:
        """Sell ``units`` FIFO for ``proceeds`` (net of trading costs). Returns the realized gain."""
        if units <= 0:
            return 0.0
        date = pd.Timestamp(date)
        held = self.units(symbol)
        if units > held * (1 + 1e-7) + UNIT_TOLERANCE:
            raise ValueError(f"Cannot sell {units} units of {symbol}; only {held} held")
        units = min(units, held)
        price = proceeds / units
        remaining = units
        total_gain = 0.0
        queue = self.lots[symbol]
        while remaining > UNIT_TOLERANCE and queue:
            lot = queue[0]
            take = min(lot.units, remaining)
            frac = take / lot.units
            basis = lot.basis * frac
            gain = take * price - basis
            self.realized.append({
                "date": date, "symbol": symbol, "lot_id": lot.lot_id, "acquired": lot.acquired,
                "units": take, "proceeds": take * price, "basis": basis, "gain": gain,
                "term": "long" if is_long_term(lot.acquired, date) else "short",
            })
            lot.sales.append((date, frac))
            total_gain += gain
            lot.units -= take
            lot.basis -= basis
            remaining -= take
            if lot.units <= UNIT_TOLERANCE * max(1.0, take):
                queue.pop(0)
        if not queue:
            self.lots.pop(symbol, None)
        return total_gain

    def dividend(self, symbol: str, date: pd.Timestamp, amount: float, yield_fraction: float) -> None:
        """Record a dividend reinvested on its ex-date.

        ``yield_fraction`` = dividend / (ex-date close + dividend): the share of
        the position's value that the reinvested dividend now represents.
        """
        lots = self.lots.get(symbol)
        if not lots or amount <= 0:
            return
        date = pd.Timestamp(date)
        total_units = sum(l.units for l in lots)
        character = self.tax_character.get(symbol, "ordinary")
        if character == QUALIFIED:
            parts = [(l.lot_id, amount * l.units / total_units) for l in lots if l.acquired < date]
            self._qualified_events.append({"date": date, "symbol": symbol, "amount": amount, "parts": parts})
        else:
            key = (date.year, character)
            self._other_income[key] = self._other_income.get(key, 0.0) + amount

        new_units = total_units * yield_fraction
        for l in lots:
            l.units *= 1.0 - yield_fraction
        last = lots[-1]
        if (self.merge_dividend_lots and last.from_dividend and not last.sales
                and (last.acquired.year, last.acquired.month) == (date.year, date.month)):
            # Merge into this month's earlier dividend lot, dated conservatively at the later date.
            last.units += new_units
            last.basis += amount
            last.acquired = date
            lot = last
        else:
            lot = self._new_lot(symbol, new_units, amount, date, from_dividend=True)
        self.purchases.append({"date": date, "symbol": symbol, "units": new_units, "basis": amount,
                               "lot_id": lot.lot_id, "source": "dividend_reinvestment"})

    # -- annual tax ----------------------------------------------------------
    def _qualified_split(self, event: dict) -> tuple[float, float]:
        """Split a qualified-fund dividend into (qualified, nonqualified) amounts."""
        ex = event["date"]
        q = nq = 0.0
        for lot_id, amount in event["parts"]:
            lot = self._all_lots[lot_id]
            remaining = 1.0
            for sale_date, frac in lot.sales:
                if sale_date <= ex:
                    continue  # already reflected in the lot's units on the ex-date
                portion = remaining * frac
                if qualified_holding_days(lot.acquired, sale_date, ex) > 60:
                    q += amount * portion
                else:
                    nq += amount * portion
                remaining -= portion
            if remaining > 0:
                # Still held: assume held through the end of the window.
                if qualified_holding_days(lot.acquired, ex + pd.Timedelta(days=60), ex) > 60:
                    q += amount * remaining
                else:
                    nq += amount * remaining
        return q, nq

    def year_income(self, year: int) -> dict[str, float]:
        income = {k: 0.0 for k in INCOME_CHARACTERS}
        for (y, character), amount in self._other_income.items():
            if y == year:
                income[character if character in income else "ordinary"] += amount
        for ev in self._qualified_events:
            if ev["date"].year == year:
                q, nq = self._qualified_split(ev)
                income["qualified"] += q
                income["nonqualified"] += nq
        return income

    def year_gains(self, year: int) -> tuple[float, float]:
        st = sum(r["gain"] for r in self.realized if r["date"].year == year and r["term"] == "short")
        lt = sum(r["gain"] for r in self.realized if r["date"].year == year and r["term"] == "long")
        return st, lt

    def preview_year(self, year: int) -> YearTax:
        """Tax for ``year`` given everything recorded so far, without closing the year."""
        st, lt = self.year_gains(year)
        return compute_year_tax(year, st, lt, self.year_income(year), self.rates, self.carry_st, self.carry_lt)

    def close_year(self, year: int) -> YearTax:
        """Finalize ``year``: compute its tax and roll loss carryforwards."""
        if year in self.closed_years:
            raise ValueError(f"Year {year} already closed")
        result = self.preview_year(year)
        self.carry_st, self.carry_lt = result.carry_st, result.carry_lt
        self.closed_years[year] = result
        return result

    def copy(self) -> "TaxLedger":
        return copy.deepcopy(self)

    # -- reporting -----------------------------------------------------------
    def realized_frame(self) -> pd.DataFrame:
        cols = ["date", "symbol", "lot_id", "acquired", "units", "proceeds", "basis", "gain", "term"]
        return pd.DataFrame(self.realized, columns=cols)

    def purchases_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.purchases, columns=["date", "symbol", "units", "basis", "lot_id", "source"])

    def years_frame(self) -> pd.DataFrame:
        return pd.DataFrame([y.as_dict() for y in self.closed_years.values()])


def wash_sale_flags(
    realized: pd.DataFrame,
    purchases: pd.DataFrame,
    groups: list[list[str]] | None = None,
    window_days: int = 30,
) -> pd.DataFrame:
    """Flag losing sales with a purchase of the same or a substantially identical
    fund within ``window_days`` before or after the sale.

    Purchases whose lots were themselves sold in that sale are not replacements
    and are ignored. Flags are informational: no basis adjustment is made.
    """
    cols = ["date", "symbol", "loss", "trigger_dates", "trigger_symbols", "by_trade", "by_dividend_reinvestment"]
    if realized.empty:
        return pd.DataFrame(columns=cols)
    group_of: dict[str, frozenset[str]] = {}
    for g in groups or []:
        for s in g:
            group_of[s] = frozenset(g)
    window = pd.Timedelta(days=window_days)
    rows = []
    sales = realized.groupby(["date", "symbol"])
    for (date, symbol), sale in sales:
        loss = sale["gain"].sum()
        if loss >= 0:
            continue
        identical = group_of.get(symbol, frozenset([symbol]))
        sold_lots = set(sale["lot_id"])
        cand = purchases[
            purchases["symbol"].isin(identical)
            & ((purchases["date"] - date).abs() <= window)
            & ~purchases["lot_id"].isin(sold_lots)
        ]
        if cand.empty:
            continue
        rows.append({
            "date": date, "symbol": symbol, "loss": loss,
            "trigger_dates": ", ".join(sorted({str(d.date()) for d in cand["date"]})),
            "trigger_symbols": ", ".join(sorted(set(cand["symbol"]))),
            "by_trade": bool((cand["source"] == "trade").any()),
            "by_dividend_reinvestment": bool((cand["source"] == "dividend_reinvestment").any()),
        })
    return pd.DataFrame(rows, columns=cols)
