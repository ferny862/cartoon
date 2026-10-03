"""Run backtests from the command line (in-sample unless the held-out period is unlocked).

    python -m src.backtest run                          # all strategies, 5 bp
    python -m src.backtest run --strategies gem trend_faber --cost-bps 2 5 10
    python -m src.backtest run --no-tax
    python -m src.backtest french                       # strategies 5-6 on French industries

Writes a CSV summary to the log directory and prints a table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.market import build_french_market_data, build_market_data
from src.backtest.runner import required_symbols, run_french, run_strategy
from src.config import PROJECT_ROOT, load_dotenv_if_present, load_settings
from src.data.store import DataStore
from src.logging_setup import setup_logging
from src.strategies import strategy_names
from src.validation.trials import TrialLog

log = logging.getLogger("src.backtest")

COLUMNS = ["strategy", "universe", "cost_bps", "start", "end", "cagr_pretax", "benchmark_cagr_pretax",
           "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "beta", "annual_turnover", "cost_drag",
           "cagr_aftertax_holding", "cagr_aftertax_liquidated", "cagr_traditional_ira",
           "pct_5y_windows_beating_benchmark", "wash_sale_flags"]


def _print(df: pd.DataFrame) -> None:
    show = df[[c for c in COLUMNS if c in df.columns]].copy()
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.4f}".format):
        print(show.to_string(index=False))


def _save(df: pd.DataFrame, settings: dict, tag: str) -> Path:
    out_dir = PROJECT_ROOT / settings["project"]["log_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backtest_{tag}_{date.today().isoformat()}.csv"
    df.to_csv(path, index=False)
    return path


def cmd_run(store: DataStore, settings: dict, args) -> int:
    names = args.strategies or strategy_names(settings)
    market = build_market_data(store, required_symbols(names, settings))
    trials = TrialLog(settings["validation"]["trial_log"])
    rows = []
    for bps in args.cost_bps or [settings["costs"]["spread_slippage_bps"]]:
        for name in names:
            run = run_strategy(name, settings, market, cost_bps=bps, taxable=not args.no_tax, trial_log=trials)
            rows.append(run.summary)
    df = pd.DataFrame(rows)
    _print(df)
    log.info("Saved %s", _save(df, settings, "etf"))
    return 0


def cmd_french(store: DataStore, settings: dict, args) -> int:
    market = build_french_market_data(store)
    trials = TrialLog(settings["validation"]["trial_log"])
    rows = [run_french(n, settings, market, trials).summary for n in settings["french_sanity"]["strategies"]]
    df = pd.DataFrame(rows)
    print("GROSS, NON-INVESTABLE Kenneth French industry portfolios (no costs, fees or taxes)")
    _print(df)
    log.info("Saved %s", _save(df, settings, "french"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.backtest")
    parser.add_argument("--settings", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run ETF strategies against SPY")
    run.add_argument("--strategies", nargs="*")
    run.add_argument("--cost-bps", nargs="*", type=float)
    run.add_argument("--no-tax", action="store_true", help="skip the taxable-account runs")
    sub.add_parser("french", help="run strategies 5-6 on French industry portfolios")
    args = parser.parse_args(argv)

    load_dotenv_if_present()
    settings = load_settings(args.settings)
    setup_logging(Path(settings["project"]["log_dir"]) / "backtest.log")
    if settings["dates"].get("heldout_unlocked"):
        log.warning("HELD-OUT PERIOD IS UNLOCKED: results include data from %s onward", settings["dates"]["heldout_start"])
    store = DataStore(settings)
    return {"run": cmd_run, "french": cmd_french}[args.command](store, settings, args)


if __name__ == "__main__":
    sys.exit(main())
