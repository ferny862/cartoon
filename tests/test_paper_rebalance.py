"""Paper rebalance job with a fake broker (no network)."""

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.paper import __main__ as pcli
from src.paper.client import AccountSnapshot, OrderResult
from src.paper.rebalance import (
    ET, PlannedOrder, RebalancePlan, latest_targets, make_plan, plan_orders, reconcile, save_plan, slippage_bps,
    submit_plan,
)
from src.paper.safety import PaperSafetyError
from src.strategies import FactorBlend, BuyAndHold
from tests.conftest import build_synthetic_store


class FakeBroker:
    def __init__(self, positions=None, equity=100_000.0, is_open=True, prices=None):
        self._positions = dict(positions or {})
        self._equity = equity
        self._open = is_open
        self._prices = prices or {}
        self.submitted = []
        self.fills = {}

    def account(self):
        return AccountSnapshot(self._equity, self._equity, 2 * self._equity, "ACTIVE", False)

    def positions(self):
        return dict(self._positions)

    def position_prices(self):
        return dict(self._prices)

    def clock(self):
        return SimpleNamespace(is_open=self._open)

    def submit_market_order(self, symbol, qty, side, tif, coid):
        self.submitted.append((symbol, qty, side, tif, coid))
        oid = f"id-{len(self.submitted)}"
        self.fills[oid] = (symbol, side, qty)
        return OrderResult(oid, coid, symbol, side, qty, "accepted", 0.0, None, None, "2023-01-03T15:00:00Z")

    def order(self, oid):
        symbol, side, qty = self.fills[oid]
        price = {"buy": 101.0, "sell": 99.0}[side]
        return OrderResult(oid, "", symbol, side, qty, "filled", qty, price, "2023-01-03 21:00:05+00:00", None)


# ------------------------------------------------------------------ order planning

def test_plan_orders_initial_buy_hand_computed():
    orders = plan_orders({"SPY": 1.0}, {}, {"SPY": 400.0}, 100_000, cash_buffer=0.005)
    assert len(orders) == 1
    o = orders[0]
    assert (o.side, o.qty, o.target_qty) == ("buy", 248.0, 248.0)        # floor(99,500 / 400)
    assert o.est_notional == pytest.approx(99_200)


def test_plan_orders_switch_sells_first():
    orders = plan_orders({"BIL": 1.0}, {"SPY": 248}, {"SPY": 400.0, "BIL": 91.5}, 100_000, 0.005)
    assert [(o.side, o.symbol, o.qty) for o in orders] == [("sell", "SPY", 248.0), ("buy", "BIL", 1087.0)]


def test_plan_orders_small_trades_skipped_but_closes_always_sent():
    orders = plan_orders({"SPY": 0.5, "EFA": 0.5}, {"SPY": 124, "EFA": 663, "XLK": 0.3},
                         {"SPY": 400.0, "EFA": 75.0, "XLK": 50.0}, 100_000, 0.005, min_trade_value=25)
    # SPY target 124 (no trade), EFA target 663 (no trade), XLK not in targets -> closed despite $15 size
    assert [(o.side, o.symbol, o.qty) for o in orders] == [("sell", "XLK", 0.3)]


def test_plan_orders_needs_prices():
    with pytest.raises(ValueError, match="No usable price"):
        plan_orders({"SPY": 1.0}, {}, {"SPY": float("nan")}, 100_000)


def test_plan_never_exceeds_investable_cash():
    rng = np.random.default_rng(0)
    for _ in range(50):
        syms = ["A", "B", "C"]
        w = rng.dirichlet(np.ones(3))
        prices = dict(zip(syms, rng.uniform(5, 900, 3)))
        orders = plan_orders(dict(zip(syms, w)), {}, prices, 50_000, 0.005)
        assert sum(o.est_notional for o in orders) <= 50_000 * 0.995 + 1e-6


# ------------------------------------------------------------------ targets

def _prices(n_months=16):
    days = pd.bdate_range("2020-01-01", periods=n_months * 22)
    me = pd.Series(days, index=days).groupby([days.year, days.month]).last().values
    days = days[days <= me[n_months - 1]]
    px = pd.DataFrame({s: 100 * np.cumprod(np.full(len(days), 1 + 0.0002 * (k + 1)))
                       for k, s in enumerate(["QUAL", "MTUM", "VLUE", "USMV", "SPY"])}, index=days)
    return px


def test_latest_targets_hold_when_no_signal():
    px = _prices()
    date, targets, note = latest_targets(FactorBlend(), px, holding_nothing=False)
    assert targets is None and "hold" in note
    date, targets, note = latest_targets(FactorBlend(), px, holding_nothing=True)
    assert targets == pytest.approx({"QUAL": .25, "MTUM": .25, "VLUE": .25, "USMV": .25}) and "initial" in note


def test_latest_targets_buy_and_hold_only_funds_once():
    px = _prices()
    assert latest_targets(BuyAndHold("SPY"), px, holding_nothing=False)[1] is None
    assert latest_targets(BuyAndHold("SPY"), px, holding_nothing=True)[1] == {"SPY": 1.0}


# ------------------------------------------------------------------ full plan with the data store

@pytest.fixture
def paper_env(tmp_path, settings):
    store, s = build_synthetic_store(tmp_path, settings, start="2018-01-02", end="2022-12-30")
    s["paper"] = {**s["paper"], "strategy": "trend_faber", "state_dir": str(tmp_path / "paper"),
                  "log_file": str(tmp_path / "paper.log")}
    return store, s


def test_make_plan_uses_signal_only_access(paper_env, tmp_path):
    store, s = paper_env
    broker = FakeBroker()
    plan = make_plan(s, store, broker, today=pd.Timestamp("2023-01-02"))
    assert plan.signal_date == "2022-12-30"                              # inside the held-out period
    assert plan.orders and plan.orders[0].side == "buy"
    assert sum(plan.targets.values()) == pytest.approx(1.0)
    access = (tmp_path / "logs" / "heldout_signal_access.log").read_text()
    assert "paper plan for trend_faber" in access
    assert store.adjusted_closes(["SPY"]).index.max() < pd.Timestamp("2021-10-01")   # lock still holds elsewhere
    path = save_plan(plan, s)
    assert RebalancePlan.from_json(path.read_text()) == plan


def test_make_plan_refusals(paper_env):
    store, s = paper_env
    with pytest.raises(PaperSafetyError, match="days old"):
        make_plan(s, store, FakeBroker(), today=pd.Timestamp("2023-02-15"))
    no_strategy = {**s, "paper": {**s["paper"], "strategy": None}}
    with pytest.raises(PaperSafetyError, match="No strategy"):
        make_plan(no_strategy, store, FakeBroker(), today=pd.Timestamp("2023-01-02"))


# ------------------------------------------------------------------ submission

def _plan():
    return RebalancePlan("trend_faber", "2022-12-30", "x", 100_000, 0.005, {"BIL": 1.0},
                         [PlannedOrder("SPY", "sell", 10, 100.0, 1000.0, 10, 0, 0.0),
                          PlannedOrder("BIL", "buy", 10, 100.0, 1000.0, 0, 10, 1.0)], "rebalance signal")


def _live_cfg(paper_env):
    _, s = paper_env
    return {**s, "paper": {**s["paper"], "dry_run": False}}


NEXT_DAY_NOON = datetime(2023, 1, 3, 12, 0, tzinfo=ET)


def test_submit_is_dry_run_by_default(paper_env):
    _, s = paper_env
    broker = FakeBroker()
    with pytest.raises(PaperSafetyError, match="Dry run"):
        submit_plan(_plan(), broker, s, execute=True, now_et=NEXT_DAY_NOON)
    with pytest.raises(PaperSafetyError, match="Dry run"):
        submit_plan(_plan(), broker, _live_cfg(paper_env), execute=False, now_et=NEXT_DAY_NOON)
    assert broker.submitted == []


@pytest.mark.parametrize("broker,now,match", [
    (FakeBroker(is_open=False), NEXT_DAY_NOON, "closed"),
    (FakeBroker(), datetime(2023, 1, 3, 15, 50, tzinfo=ET), "cutoff"),
    (FakeBroker(), datetime(2022, 12, 30, 12, 0, tzinfo=ET), "AFTER the signal date"),
])
def test_submission_window(paper_env, broker, now, match):
    with pytest.raises(PaperSafetyError, match=match):
        submit_plan(_plan(), broker, _live_cfg(paper_env), execute=True, now_et=now)
    assert broker.submitted == []


def test_submit_sends_moc_orders_once(paper_env):
    cfg = _live_cfg(paper_env)
    broker = FakeBroker()
    results = submit_plan(_plan(), broker, cfg, execute=True, now_et=NEXT_DAY_NOON)
    assert [(s, side, tif) for s, _, side, tif, _ in broker.submitted] == [("SPY", "sell", "cls"), ("BIL", "buy", "cls")]
    assert len(results) == 2 and broker.submitted[0][4] == "trend_faber-2022-12-30-SPY-sell"
    with pytest.raises(PaperSafetyError, match="already submitted"):
        submit_plan(_plan(), broker, cfg, execute=True, now_et=NEXT_DAY_NOON)


# ------------------------------------------------------------------ reconciliation

def test_slippage_sign_convention():
    assert slippage_bps("buy", 101, 100) == pytest.approx(100)      # paid more: worse
    assert slippage_bps("sell", 99, 100) == pytest.approx(100)      # received less: worse
    assert slippage_bps("sell", 101, 100) == pytest.approx(-100)    # better than the close
    assert np.isnan(slippage_bps("buy", 101, float("nan")))


def test_reconcile_compares_fills_with_close(paper_env):
    cfg = _live_cfg(paper_env)
    broker = FakeBroker()
    submit_plan(_plan(), broker, cfg, execute=True, now_et=NEXT_DAY_NOON)
    record = json.loads(next((pd.io.common.Path(cfg["paper"]["state_dir"])).glob("submitted_*.json")).read_text())
    closes = pd.DataFrame({"SPY": [100.0], "BIL": [100.0]}, index=[pd.Timestamp("2023-01-03")])
    fills = reconcile(broker, record, closes).set_index("symbol")
    assert fills.loc["SPY", "fill_date"] == pd.Timestamp("2023-01-03").date()       # 21:00 UTC = 16:00 ET
    assert fills.loc["SPY", "slippage_bps"] == pytest.approx(100)
    assert fills.loc["BIL", "slippage_bps"] == pytest.approx(100)


# ------------------------------------------------------------------ CLI

def test_cli_plan_and_dry_run_submit(paper_env, monkeypatch, capsys):
    store, s = paper_env
    monkeypatch.setattr(pcli, "load_dotenv_if_present", lambda: None)
    monkeypatch.setattr(pcli, "load_settings", lambda path=None: s)
    broker = FakeBroker()
    import src.paper.rebalance as rb
    real_make_plan = rb.make_plan
    monkeypatch.setattr(pcli, "make_plan", lambda st, sto, b: real_make_plan(st, sto, b, today=pd.Timestamp("2023-01-02")))
    assert pcli.main(["plan"], broker_factory=lambda st: broker, store_factory=lambda st: store) == 0
    assert "Saved plan" in capsys.readouterr().out
    assert pcli.main(["submit", "--execute"], broker_factory=lambda st: broker, store_factory=lambda st: store) == 0
    assert "DRY RUN" in capsys.readouterr().out and broker.submitted == []


def test_cli_refuses_without_paper_flag(settings, monkeypatch, capsys):
    monkeypatch.setattr(pcli, "load_dotenv_if_present", lambda: None)
    monkeypatch.setattr(pcli, "setup_logging", lambda *a, **k: None)
    monkeypatch.delenv("PAPER_TRADING", raising=False)
    assert pcli.main(["status"]) == 2
    assert "REFUSED" in capsys.readouterr().out
