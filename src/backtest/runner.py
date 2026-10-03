"""Run configured strategies against the benchmark over the same period.

Each run produces a pre-tax result (also the Roth IRA result), optionally a
taxable result, and a buy-and-hold SPY benchmark that starts on the same
signal date as the strategy so comparisons cover identical periods. Every
run is appended to the trial log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import BacktestResult, MarketData, TaxSettings, run_backtest
from src.backtest.metrics import summarize
from src.strategies import build_french_variant, build_strategy, strategy_params
from src.validation.trials import TrialLog

log = logging.getLogger(__name__)


@dataclass
class StrategyRun:
    name: str
    universe: str
    weights: pd.DataFrame
    pretax: BacktestResult
    benchmark: BacktestResult
    taxable: BacktestResult | None
    summary: dict
    cost_bps: float


def _engine_kwargs(settings: dict) -> dict:
    return {
        "initial_capital": float(settings["project"]["starting_capital"]),
        "execution_lag": int(settings["execution"]["execution_lag_days"]),
        "allow_fractional": bool(settings["execution"].get("allow_fractional_shares", True)),
    }


def _run(strategy, settings: dict, market: MarketData, benchmark_symbol: str, cost_bps: float | None,
         taxable: bool, universe: str, trial_log: TrialLog | None, params: dict) -> StrategyRun:
    costs = CostModel.from_settings(settings, cost_bps)
    kw = _engine_kwargs(settings)
    weights = strategy.generate(market.prices)
    if weights.empty:
        raise ValueError(f"{strategy.name}: no signals in the available data")
    pre = run_backtest(weights, market, costs, name=strategy.name, **kw)
    bench_w = pd.DataFrame({benchmark_symbol: [1.0]}, index=[weights.index[0]])
    bench = run_backtest(bench_w, market, costs, name=f"{benchmark_symbol} buy and hold", **kw)
    tax_res = None
    if taxable:
        tax_res = run_backtest(weights, market, costs, tax=TaxSettings.from_settings(settings),
                               name=f"{strategy.name} (taxable)", **kw)
    summary = summarize(pre, bench, market.risk_free, taxable=tax_res,
                        traditional_ira_rate=settings["taxes"].get("traditional_ira_withdrawal_rate"))
    summary["strategy"] = strategy.name
    summary["universe"] = universe
    summary["cost_bps"] = costs.spread_slippage_bps
    summary["benchmark_cagr_pretax"] = summarize(bench, None, market.risk_free)["cagr_pretax"]
    if trial_log is not None:
        trial_log.record(strategy.name, params, {
            "universe": universe, "cost_bps": costs.spread_slippage_bps, "taxable": taxable,
            "start": str(pre.start.date()), "end": str(pre.end.date()),
            "heldout_unlocked": bool(settings["dates"].get("heldout_unlocked", False)),
        })
    return StrategyRun(strategy.name, universe, weights, pre, bench, tax_res, summary, costs.spread_slippage_bps)


def run_strategy(name: str, settings: dict, market: MarketData, cost_bps: float | None = None,
                 taxable: bool = True, trial_log: TrialLog | None = None) -> StrategyRun:
    """Run one configured ETF strategy and its same-period SPY benchmark."""
    strategy = build_strategy(name, settings)
    return _run(strategy, settings, market, settings["benchmark"], cost_bps, taxable, "etf",
                trial_log, strategy_params(name, settings))


def run_french(name: str, settings: dict, market: MarketData, trial_log: TrialLog | None = None) -> StrategyRun:
    """Run strategy 5 or 6 on French industries (gross, non-investable, no taxes)."""
    cfg = settings["french_sanity"]
    industries = [c for c in market.prices.columns if c not in (cfg["cash_symbol"], cfg["market_symbol"])]
    strategy = build_french_variant(name, settings, industries)
    params = {**strategy_params(name, settings), "sectors": industries, "cash": cfg["cash_symbol"]}
    return _run(strategy, settings, market, cfg["market_symbol"], cfg["cost_bps"], False, "french_gross",
                trial_log, params)


def required_symbols(names: list[str], settings: dict) -> list[str]:
    syms: list[str] = [settings["benchmark"]]
    for n in names:
        for s in build_strategy(n, settings).symbols():
            if s not in syms:
                syms.append(s)
    return syms
