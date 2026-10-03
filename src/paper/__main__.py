"""Paper-trading command line (Alpaca paper endpoint only).

    python -m src.paper status
    python -m src.paper plan                 # after the month-end close
    python -m src.paper submit [--execute]   # next trading day, before 15:45 ET
    python -m src.paper reconcile            # after that day's close (download prices first)

``submit`` is a dry run unless BOTH paper.dry_run is false in
config/settings.yaml AND --execute is given. Everything is logged to
paper.log_file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import load_dotenv_if_present, load_settings
from src.data.store import DataStore
from src.logging_setup import setup_logging
from src.paper.client import PaperBroker
from src.paper.rebalance import (
    RebalancePlan, append_fills, latest_plan_path, make_plan, reconcile, save_plan, state_dir, submit_plan,
)
from src.paper.safety import PaperSafetyError

log = logging.getLogger("src.paper")


def cmd_status(settings, broker, store, args) -> int:
    a = broker.account()
    print(f"Paper account: equity ${a.equity:,.2f} · cash ${a.cash:,.2f} · status {a.status}")
    for sym, qty in sorted(broker.positions().items()):
        print(f"  {sym:<6s} {qty:g}")
    return 0


def cmd_plan(settings, broker, store, args) -> int:
    plan = make_plan(settings, store, broker)
    path = save_plan(plan, settings)
    print(plan.describe())
    print(f"\nSaved plan: {path}")
    log.info("Plan saved to %s with %d orders", path, len(plan.orders))
    return 0


def cmd_submit(settings, broker, store, args) -> int:
    strategy = settings["paper"].get("strategy")
    if not strategy:
        raise PaperSafetyError("No strategy chosen: set paper.strategy in config/settings.yaml.")
    plan = RebalancePlan.from_json(latest_plan_path(settings, strategy).read_text())
    print(plan.describe())
    try:
        results = submit_plan(plan, broker, settings, execute=args.execute, force=args.force)
    except PaperSafetyError as exc:
        if "Dry run" in str(exc):
            print(f"\nDRY RUN, nothing submitted. {exc}")
            log.info("Dry run for plan %s: %s", plan.signal_date, exc)
            return 0
        raise
    print(f"\nSubmitted {len(results)} market-on-close order(s).")
    return 0


def cmd_reconcile(settings, broker, store, args) -> int:
    strategy = settings["paper"].get("strategy")
    pattern = f"submitted_{strategy}_{args.signal_date or '*'}.json"
    records = sorted(state_dir(settings).glob(pattern))
    if not records:
        raise FileNotFoundError(f"No submission record matching {pattern}")
    submission = json.loads(records[-1].read_text())
    symbols = sorted({o["symbol"] for o in submission["orders"]})
    closes = None
    if symbols:
        try:
            closes = store.signal_data(symbols, purpose="paper fill reconciliation")[1]
        except FileNotFoundError as exc:
            log.warning("No cached closes (%s); slippage left blank. Download prices and re-run.", exc)
    fills = reconcile(broker, submission, closes)
    path = append_fills(fills, settings)
    if len(fills):
        print(fills[["symbol", "side", "qty", "status", "fill_price", "backtest_price", "slippage_bps"]].to_string(index=False))
        print(f"\nMean slippage vs the backtest's close: {fills['slippage_bps'].mean():.1f} bp (positive = worse)")
    print(f"Fills recorded in {path}")
    return 0


def main(argv: list[str] | None = None, broker_factory=None, store_factory=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.paper")
    parser.add_argument("--settings", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the paper account")
    sub.add_parser("plan", help="compute this month's targets and intended orders (no orders sent)")
    sp = sub.add_parser("submit", help="send the saved plan as market-on-close orders")
    sp.add_argument("--execute", action="store_true", help="actually submit (also needs paper.dry_run: false)")
    sp.add_argument("--force", action="store_true", help="allow re-submitting a plan already submitted")
    rp = sub.add_parser("reconcile", help="record fills and slippage vs the backtest's close")
    rp.add_argument("--signal-date", default=None)
    args = parser.parse_args(argv)

    load_dotenv_if_present()
    settings = load_settings(args.settings)
    setup_logging(Path(settings["paper"]["log_file"]))
    try:
        broker = (broker_factory or PaperBroker)(settings)
        store = (store_factory or DataStore)(settings)
        handler = {"status": cmd_status, "plan": cmd_plan, "submit": cmd_submit, "reconcile": cmd_reconcile}
        return handler[args.command](settings, broker, store, args)
    except PaperSafetyError as exc:
        print(f"REFUSED: {exc}")
        log.error("Refused: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
