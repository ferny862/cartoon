"""Tax-lot model: FIFO, holding periods, netting, dividends, wash sales."""

import pandas as pd
import pytest

from src.backtest.taxes import (
    TaxLedger,
    TaxRates,
    compute_year_tax,
    is_long_term,
    net_capital_gains,
    qualified_holding_days,
    wash_sale_flags,
)

T = pd.Timestamp
RATES = TaxRates(short_term=0.24, long_term=0.15, qualified_dividend=0.15, ordinary=0.24, state=0.05)
CHAR = {"SPY": "qualified", "BIL": "ordinary_treasury", "AGG": "ordinary", "VNQ": "ordinary_reit",
        "IVV": "qualified", "XLK": "qualified"}


def ledger():
    return TaxLedger(RATES, CHAR)


def test_one_year_boundary_is_short_term():
    assert not is_long_term(T("2020-01-15"), T("2021-01-15"))  # exactly one year: short
    assert is_long_term(T("2020-01-15"), T("2021-01-16"))


def test_fifo_lot_order_and_terms():
    lg = ledger()
    lg.buy("SPY", T("2020-01-02"), units=10, basis=1000)   # $100/unit
    lg.buy("SPY", T("2020-12-01"), units=10, basis=1500)   # $150/unit
    gain = lg.sell("SPY", T("2021-03-01"), units=15, proceeds=15 * 200)
    # FIFO: 10 units from lot 1 (LT, gain 1000) + 5 from lot 2 (ST, gain 250)
    assert gain == pytest.approx(1250)
    r = lg.realized_frame()
    assert list(r["term"]) == ["long", "short"]
    assert r["gain"].tolist() == pytest.approx([1000, 250])
    assert lg.units("SPY") == pytest.approx(5) and lg.basis("SPY") == pytest.approx(750)


def test_cannot_oversell():
    lg = ledger()
    lg.buy("SPY", T("2020-01-02"), 1, 100)
    with pytest.raises(ValueError):
        lg.sell("SPY", T("2020-02-03"), 2, 300)


@pytest.mark.parametrize("st,lt,cst,clt,expected", [
    (100, 200, 0, 0, (100, 200, 0, 0)),
    (-100, 200, 0, 0, (0, 100, 0, 0)),         # ST loss absorbs part of LT gain
    (-300, 200, 0, 0, (0, 0, -100, 0)),        # net loss carried as short-term
    (300, -500, 0, 0, (0, 0, 0, -200)),        # net loss carried as long-term
    (-50, -70, 0, 0, (0, 0, -50, -70)),
    (500, 0, -200, -100, (200, 0, 0, 0)),      # carryforwards used up
])
def test_netting(st, lt, cst, clt, expected):
    assert net_capital_gains(st, lt, cst, clt) == pytest.approx(expected)


def test_year_tax_hand_computed():
    income = {"qualified": 1000, "ordinary": 200, "ordinary_treasury": 300, "ordinary_reit": 100}
    y = compute_year_tax(2021, st_net=2000, lt_net=4000, income=income, rates=RATES)
    federal = 2000 * .24 + 4000 * .15 + 1000 * .15 + (200 + 300 + 100) * .24
    state = (2000 + 4000 + 1000 + 200 + 100) * .05        # Treasury interest exempt
    assert y.federal == pytest.approx(federal)            # 480 + 600 + 150 + 144 = 1374
    assert y.state == pytest.approx(state)                # 365
    assert y.total == pytest.approx(1739)


def test_carryforward_across_years():
    lg = ledger()
    lg.buy("SPY", T("2020-01-02"), 10, 1000)
    lg.sell("SPY", T("2020-06-01"), 10, 700)              # -300 short-term
    y1 = lg.close_year(2020)
    assert y1.total == 0 and lg.carry_st == pytest.approx(-300)
    lg.buy("SPY", T("2021-01-04"), 10, 1000)
    lg.sell("SPY", T("2021-06-01"), 10, 1500)             # +500 short-term
    y2 = lg.close_year(2021)
    assert y2.taxable_st == pytest.approx(200)
    assert y2.total == pytest.approx(200 * (.24 + .05))
    assert lg.carry_st == 0


def test_carryforward_disabled():
    rates = TaxRates(0.24, 0.15, 0.15, 0.24, 0.05, carry_forward_losses=False)
    y = compute_year_tax(2021, 500, 0, {}, rates, carry_st=-200)
    assert y.taxable_st == 500 and y.carry_st == 0


def test_dividend_reinvestment_creates_lot_and_scales_old_lots():
    lg = ledger()
    lg.buy("SPY", T("2020-01-02"), units=100, basis=10_000)
    # Ex-date: close 98 + dividend 2 -> yield fraction 2/100 = 0.02; position worth 10,000.
    lg.dividend("SPY", T("2020-03-20"), amount=200, yield_fraction=0.02)
    assert lg.units("SPY") == pytest.approx(100)         # total units unchanged
    lots = lg.lots["SPY"]
    assert lots[0].units == pytest.approx(98) and lots[0].basis == pytest.approx(10_000)
    assert lots[1].units == pytest.approx(2) and lots[1].basis == pytest.approx(200)
    assert lots[1].acquired == T("2020-03-20") and lots[1].from_dividend


def test_qualified_holding_days_rule():
    ex = T("2020-06-15")
    assert qualified_holding_days(T("2020-06-14"), T("2020-08-14"), ex) == 61   # qualifies (>60)
    assert qualified_holding_days(T("2020-06-14"), T("2020-08-13"), ex) == 60   # fails
    assert qualified_holding_days(T("2019-01-01"), T("2021-01-01"), ex) == 121


def test_dividend_holding_test_splits_qualified():
    lg = ledger()
    lg.buy("SPY", T("2020-01-02"), 50, 5000)               # long held: qualifies
    lg.buy("SPY", T("2020-06-01"), 50, 5000)               # bought 14 days before ex-date
    lg.dividend("SPY", T("2020-06-15"), amount=100, yield_fraction=0.01)
    # FIFO sells lot 1 entirely and half of lot 2 on 2020-07-01, 16 days after the ex-date.
    lg.sell("SPY", T("2020-07-01"), units=49.5 + 24.75, proceeds=7425)
    inc = lg.year_income(2020)
    # Lot 1 (held since Jan): its $50 qualifies even though sold.
    # Lot 2: sold half after 30 window days -> fails; remaining half assumed held -> qualifies.
    assert inc["qualified"] == pytest.approx(50 + 25)
    assert inc["nonqualified"] == pytest.approx(25)


def test_non_qualified_characters():
    lg = ledger()
    lg.buy("BIL", T("2020-01-02"), 100, 10_000)
    lg.buy("VNQ", T("2020-01-02"), 100, 10_000)
    lg.dividend("BIL", T("2020-02-03"), 20, 0.002)
    lg.dividend("VNQ", T("2020-03-20"), 80, 0.008)
    inc = lg.year_income(2020)
    assert inc["ordinary_treasury"] == pytest.approx(20) and inc["ordinary_reit"] == pytest.approx(80)
    y = lg.close_year(2020)
    assert y.state == pytest.approx(80 * 0.05)              # BIL interest state-exempt


def test_dividend_lots_merge_within_month():
    lg = ledger()
    lg.buy("BIL", T("2020-01-02"), 100, 10_000)
    lg.dividend("BIL", T("2020-02-03"), 1, 0.0001)
    lg.dividend("BIL", T("2020-02-04"), 1, 0.0001)
    lg.dividend("BIL", T("2020-03-02"), 1, 0.0001)
    lots = lg.lots["BIL"]
    assert len(lots) == 3                                    # original + Feb + Mar
    assert lots[1].basis == pytest.approx(2) and lots[1].acquired == T("2020-02-04")


def test_close_year_twice_raises():
    lg = ledger()
    lg.close_year(2020)
    with pytest.raises(ValueError):
        lg.close_year(2020)


def test_wash_sale_flags():
    lg = ledger()
    lg.buy("SPY", T("2020-01-02"), 10, 1000)
    lg.sell("SPY", T("2020-03-02"), 10, 800)                # loss
    lg.buy("IVV", T("2020-03-20"), 5, 400)                  # substantially identical, 18 days later
    lg.buy("XLK", T("2020-01-02"), 10, 1000)
    lg.sell("XLK", T("2020-03-02"), 10, 900)                # loss, no repurchase
    lg.buy("XLK", T("2020-05-01"), 10, 900)                 # 60 days later: outside window
    flags = wash_sale_flags(lg.realized_frame(), lg.purchases_frame(), [["SPY", "IVV", "VOO"]], 30)
    assert list(flags["symbol"]) == ["SPY"]
    assert flags.iloc[0]["trigger_symbols"] == "IVV" and flags.iloc[0]["by_trade"]


def test_wash_sale_purchase_before_sale_counts_but_sold_lots_do_not():
    lg = ledger()
    lg.buy("SPY", T("2020-01-02"), 10, 1000)
    lg.buy("SPY", T("2020-02-20"), 10, 1000)                # bought 11 days before the loss sale
    lg.sell("SPY", T("2020-03-02"), 10, 800)                # FIFO sells lot 1 only
    flags = wash_sale_flags(lg.realized_frame(), lg.purchases_frame(), None, 30)
    assert len(flags) == 1 and "2020-02-20" in flags.iloc[0]["trigger_dates"]

    lg2 = ledger()
    lg2.buy("SPY", T("2020-02-20"), 10, 1000)
    lg2.sell("SPY", T("2020-03-02"), 10, 800)               # the only purchase was the lot sold
    assert wash_sale_flags(lg2.realized_frame(), lg2.purchases_frame(), None, 30).empty


def test_rates_from_settings(settings):
    r = TaxRates.from_settings(settings)
    assert (r.short_term, r.long_term, r.qualified_dividend, r.ordinary, r.state) == (0.24, 0.15, 0.15, 0.24, 0.05)
    bad = {**settings, "taxes": {**settings["taxes"], "ordinary_income_loss_offset": 3000}}
    with pytest.raises(ValueError):
        TaxRates.from_settings(bad)
