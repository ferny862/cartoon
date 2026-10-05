"""Run all strategies in-sample and every validation check.

    python -m src.validation [--no-tax]

Saves tables and a plain-English summary under logs/validation_<date>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from src.backtest.market import build_market_data
from src.backtest.runner import required_symbols, run_strategy
from src.config import PROJECT_ROOT, load_dotenv_if_present, load_settings
from src.data.store import DataStore
from src.logging_setup import setup_logging
from src.strategies import strategy_names
from src.validation.suite import run_validation
from src.validation.trials import TrialLog

log = logging.getLogger("src.validation")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.validation")
    parser.add_argument("--settings", default=None)
    parser.add_argument("--no-tax", action="store_true", help="skip taxable runs (validation uses pre-tax returns)")
    args = parser.parse_args(argv)

    load_dotenv_if_present()
    settings = load_settings(args.settings)
    setup_logging(Path(settings["project"]["log_dir"]) / "validation.log")
    if settings["dates"].get("heldout_unlocked"):
        log.warning("HELD-OUT PERIOD IS UNLOCKED: validation includes data from %s onward",
                    settings["dates"]["heldout_start"])
    store = DataStore(settings)
    names = strategy_names(settings)
    market = build_market_data(store, required_symbols(names, settings))
    trials = TrialLog(settings["validation"]["trial_log"])
    runs = {n: run_strategy(n, settings, market, taxable=not args.no_tax, trial_log=trials) for n in names}
    report = run_validation(runs, market, settings, trials)
    out = report.save(PROJECT_ROOT / settings["project"]["log_dir"] / f"validation_{date.today().isoformat()}")
    print("\n".join(report.explanations))
    log.info("Saved validation tables to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
