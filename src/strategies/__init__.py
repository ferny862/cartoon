"""Strategies: one module per strategy, all returning target weights per signal date.

``build_strategy(name, settings)`` constructs a strategy from its entry under
``strategies:`` in config/settings.yaml. Parameters come only from that file,
so every configuration that is run can be logged.
"""

from __future__ import annotations

from src.strategies.base import Strategy, month_end_dates
from src.strategies.buy_and_hold import BuyAndHold
from src.strategies.factor_blend import FactorBlend
from src.strategies.gem import GEM
from src.strategies.gtaa import GTAA
from src.strategies.sector_momentum import SectorMomentum
from src.strategies.trend_filter import TrendFilter

_IGNORED_KEYS = {"module", "note"}


def _buy_and_hold(name, p):
    return BuyAndHold(symbol=p["symbol"], name=name)


def _trend(name, p):
    return TrendFilter(risk_asset=p["risk_asset"], cash=p["cash"], sma_months=p["sma_months"], name=name)


def _gem(name, p):
    return GEM(us=p["us"], intl=p["intl"], bonds=p["bonds"], cash=p["cash"],
               lookback_months=p["lookback_months"], name=name)


def _gtaa(name, p):
    return GTAA(assets=p["assets"], cash=p["cash"], sma_months=p["sma_months"], name=name)


def _factor(name, p):
    if p.get("rebalance", "annual") != "annual":
        raise ValueError("factor_blend supports rebalance: annual only")
    return FactorBlend(assets=p["assets"], drift_threshold=p["drift_threshold"],
                       rebalance_month=p.get("rebalance_month", 12), name=name)


def _sector(name, p):
    return SectorMomentum(sectors=p["sectors"], cash=p["cash"], lookback_months=p["lookback_months"],
                          skip_months=p["skip_months"], top_n=p["top_n"], filter=p["filter"],
                          sma_months=p["sma_months"], market=p.get("market"), name=name)


BUILDERS = {
    "buy_and_hold": _buy_and_hold,
    "trend_filter": _trend,
    "gem": _gem,
    "gtaa": _gtaa,
    "factor_blend": _factor,
    "sector_momentum": _sector,
}


def strategy_names(settings: dict) -> list[str]:
    return list(settings["strategies"].keys())


def strategy_params(name: str, settings: dict) -> dict:
    """The configured parameters for one strategy (without bookkeeping keys)."""
    spec = settings["strategies"][name]
    return {k: v for k, v in spec.items() if k not in _IGNORED_KEYS}


def build_strategy(name: str, settings: dict) -> Strategy:
    if name not in settings["strategies"]:
        raise KeyError(f"Unknown strategy {name!r}; configured: {strategy_names(settings)}")
    spec = settings["strategies"][name]
    module = spec["module"]
    if module not in BUILDERS:
        raise ValueError(f"Strategy {name!r} uses unknown module {module!r}")
    return BUILDERS[module](name, spec)


def build_french_variant(name: str, settings: dict, industries: list[str]) -> Strategy:
    """Strategy 5 or 6 with identical rules, run on French industry portfolios.

    Sectors become the French industries, cash becomes the French risk-free
    rate and the market filter uses the French market return.
    """
    cfg = settings["french_sanity"]
    if name not in cfg["strategies"]:
        raise ValueError(f"{name} is not configured for the French sanity check")
    spec = dict(settings["strategies"][name])
    spec["sectors"] = list(industries)
    spec["cash"] = cfg["cash_symbol"]
    if spec.get("filter") == "market_sma":
        spec["market"] = cfg["market_symbol"]
    return BUILDERS[spec["module"]](f"{name}__french", spec)


__all__ = ["build_french_variant", "Strategy", "build_strategy", "strategy_names", "strategy_params", "month_end_dates",
           "BuyAndHold", "TrendFilter", "GEM", "GTAA", "FactorBlend", "SectorMomentum"]
