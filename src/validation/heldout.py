"""One-time held-out evaluation of the approved final strategies.

Simulates an investor who starts each strategy on the first trading day of
the held-out period (using the target set at the last close before it) and
compares it with SPY bought the same day: pre-tax, taxable, block bootstrap,
and the 2022 downturn.

Refuses to run unless the held-out period is unlocked AND
docs/heldout_log.md names every strategy being evaluated.

    python -m src.validation.heldout --strategies trend_faber sector_mom_market_filter
"""

from __future__ import annotations

import argparse
import html
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import MarketData, TaxSettings, run_backtest
from src.backtest.metrics import summarize
from src.config import PROJECT_ROOT, heldout_start, load_dotenv_if_present, load_settings
from src.data.store import HeldOutLockedError
from src.reporting import charts
from src.reporting.build import fmt
from src.reporting.downturns import describe_downturn_row, downturn_table
from src.reporting.names import display_name
from src.strategies import build_strategy, strategy_params
from src.validation.bootstrap import bootstrap_difference, explain_bootstrap
from src.validation.common import monthly_returns, monthly_risk_free
from src.validation.trials import TrialLog

log = logging.getLogger(__name__)
HELDOUT_LOG = PROJECT_ROOT / "docs" / "heldout_log.md"


@dataclass
class HeldOutRun:
    name: str
    pretax: object
    taxable: object
    benchmark: object
    benchmark_taxable: object
    summary: dict
    bootstrap: object


def check_heldout_allowed(settings: dict, names: list[str], log_path: Path = HELDOUT_LOG) -> None:
    if not settings["dates"].get("heldout_unlocked", False):
        raise HeldOutLockedError("The held-out period is locked (dates.heldout_unlocked is false).")
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    missing = [n for n in names if f"`{n}`" not in text]
    if missing:
        raise HeldOutLockedError(f"docs/heldout_log.md does not record an evaluation of {missing}; "
                                 "record it before looking at held-out results.")


def heldout_weights(weights: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Targets from the last signal before ``start`` onward, so trading begins on the first held-out day."""
    before = weights.index[weights.index < start]
    if len(before) == 0:
        raise ValueError("The strategy has no signal before the held-out period")
    return weights.loc[weights.index >= before[-1]]


def run_heldout(names: list[str], settings: dict, market: MarketData, trial_log: TrialLog | None = None,
                log_path: Path = HELDOUT_LOG) -> dict[str, HeldOutRun]:
    check_heldout_allowed(settings, names, log_path)
    start = heldout_start(settings)
    costs = CostModel.from_settings(settings)
    tax = TaxSettings.from_settings(settings)
    kw = {"initial_capital": float(settings["project"]["starting_capital"]),
          "execution_lag": int(settings["execution"]["execution_lag_days"]),
          "allow_fractional": bool(settings["execution"].get("allow_fractional_shares", True))}
    ira = settings["taxes"].get("traditional_ira_withdrawal_rate")
    bcfg = settings["validation"]["bootstrap"]
    out: dict[str, HeldOutRun] = {}
    for name in names:
        w = heldout_weights(build_strategy(name, settings).generate(market.prices), start)
        bench_w = pd.DataFrame({settings["benchmark"]: [1.0]}, index=[w.index[0]])
        pre = run_backtest(w, market, costs, name=name, **kw)
        taxable = run_backtest(w, market, costs, tax=tax, name=f"{name} (taxable)", **kw)
        bench = run_backtest(bench_w, market, costs, name="SPY", **kw)
        bench_tax = run_backtest(bench_w, market, costs, tax=tax, name="SPY (taxable)", **kw)
        if pre.start < start:
            raise RuntimeError("Held-out run started before the held-out period")
        s = summarize(pre, bench, market.risk_free, taxable=taxable, traditional_ira_rate=ira)
        b = summarize(bench, None, market.risk_free, taxable=bench_tax, traditional_ira_rate=ira)
        s.update({f"benchmark_{k}": b[k] for k in ("cagr_pretax", "sharpe", "sortino", "max_drawdown", "volatility",
                                                   "cagr_aftertax_liquidated", "cagr_traditional_ira", "calmar")})
        s["strategy"] = name
        boot = bootstrap_difference(monthly_returns(pre.equity).iloc[1:], monthly_returns(bench.equity).iloc[1:],
                                    monthly_risk_free(market.risk_free), block=int(bcfg["block_months"]),
                                    n_samples=int(bcfg["n_samples"]), seed=int(bcfg["seed"]))
        if trial_log is not None:
            trial_log.record(name, strategy_params(name, settings),
                             {"universe": "etf", "heldout_evaluation": True, "start": str(pre.start.date()),
                              "end": str(pre.end.date())})
        out[name] = HeldOutRun(name, pre, taxable, bench, bench_tax, s, boot)
    return out


def downturns_2022(runs: dict[str, HeldOutRun], market: MarketData, settings: dict) -> pd.DataFrame:
    first = next(iter(runs.values()))
    view = {"benchmark": SimpleNamespace(pretax=first.benchmark)}
    view.update({n: SimpleNamespace(pretax=r.pretax) for n, r in runs.items()})
    cfg = {**settings, "report": {**settings["report"],
                                  "downturns": {k: v for k, v in settings["report"]["downturns"].items()
                                                if pd.Timestamp(v[0]) >= heldout_start(settings)}}}
    return downturn_table(view, market, cfg, {n: strategy_params(n, settings) for n in runs})


ROWS = [
    ("cagr_pretax", "benchmark_cagr_pretax", "Return per year (pre-tax / Roth IRA)", "pct"),
    ("cagr_aftertax_liquidated", "benchmark_cagr_aftertax_liquidated", "Return per year, taxable (liquidated)", "pct"),
    ("cagr_traditional_ira", "benchmark_cagr_traditional_ira", "Return per year, traditional IRA", "pct"),
    ("volatility", "benchmark_volatility", "Volatility", "pct"),
    ("sharpe", "benchmark_sharpe", "Sharpe ratio", "num"),
    ("sortino", "benchmark_sortino", "Sortino ratio", "num"),
    ("max_drawdown", "benchmark_max_drawdown", "Maximum drawdown", "pct"),
    ("calmar", "benchmark_calmar", "Calmar ratio", "num"),
]


def render(runs: dict[str, HeldOutRun], downturns: pd.DataFrame, settings: dict) -> tuple[str, str]:
    """(html, markdown) for the held-out results."""
    first = next(iter(runs.values()))
    period = f"{first.pretax.start.date()} to {first.pretax.end.date()}"
    md = ["# Held-out test", "", f"Period: **{period}** (never used during development). "
          "Each strategy starts with $100,000 on the first held-out day; SPY is bought the same day.", ""]
    html_parts = []
    for name, r in runs.items():
        s = r.summary
        md += [f"## {display_name(name)}", "", "| Measure | Strategy | SPY |", "|---|---|---|"]
        rows_html = []
        for k, bk, label, kind in ROWS:
            md.append(f"| {label} | {fmt(s.get(k), kind)} | {fmt(s.get(bk), kind)} |")
            rows_html.append(f"<tr><td>{html.escape(label)}</td><td class='n'>{fmt(s.get(k), kind)}</td>"
                             f"<td class='n'>{fmt(s.get(bk), kind)}</td></tr>")
        extra = (f"Turnover {fmt(s['annual_turnover'], 'pct')} per year; beta {fmt(s['beta'], 'num')}; "
                 f"wash-sale flags {s.get('wash_sale_flags', 0)}.")
        boot = explain_bootstrap(display_name(name), r.bootstrap)
        md += ["", extra, "", boot, ""]
        html_parts.append(f"<h2>{html.escape(display_name(name))}</h2><div class='table-wrap'><table><thead><tr>"
                          f"<th>Measure</th><th class='n'>Strategy</th><th class='n'>SPY</th></tr></thead><tbody>"
                          f"{''.join(rows_html)}</tbody></table></div><p>{html.escape(extra)}</p>"
                          f"<p>{html.escape(boot)}</p>")
    live = downturns[downturns["strategy"].notna()] if len(downturns) else downturns
    dt_lines = [describe_downturn_row(row, display_name) for _, row in live.iterrows()]
    if len(live):
        r0 = live.iloc[0]
        spy = (f"SPY fell {abs(r0['spy_drawdown']):.0%} from {r0['spy_peak']} to {r0['spy_trough']}"
               + (f" and regained its peak on {r0['spy_recovery']}." if r0["spy_recovered"] else "."))
        md += ["## 2022 bear market", "", spy, ""] + [f"- {t}" for t in dt_lines]
        html_parts.append("<h2>2022 bear market</h2><p>" + html.escape(spy) + "</p><ul>"
                          + "".join(f"<li>{html.escape(t)}</li>" for t in dt_lines) + "</ul>")
    view = {"benchmark": SimpleNamespace(pretax=first.benchmark)}
    view.update({n: SimpleNamespace(pretax=r.pretax, benchmark=r.benchmark) for n, r in runs.items()})
    figs = []
    for key, fn, cap in (("equity", charts.equity_chart, "Growth of $1 from the start of the held-out period"),
                         ("dd", charts.drawdown_chart, "Drawdowns in the held-out period")):
        pair = charts.both_themes(fn, view)
        figs.append(f"<figure><picture><source media='(prefers-color-scheme: dark)' "
                    f"srcset='data:image/png;base64,{pair['dark']}'><img alt='{html.escape(cap)}' "
                    f"src='data:image/png;base64,{pair['light']}'></picture><figcaption>{html.escape(cap)}"
                    f"</figcaption></figure>")
    template = (PROJECT_ROOT / "src" / "reporting" / "templates" / "report.html.j2").read_text(encoding="utf-8")
    css = template[template.index("<style>"): template.index("</style>") + len("</style>")]
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>Held-Out Test</title>{css}"
            f"</head><body><main><h1>Held-out test</h1><p class='meta'>Period {period} · generated "
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p><div class='banner'><strong>Out-of-sample.</strong> "
            f"These five years were never used while the strategies were chosen. Each strategy starts with "
            f"$100,000 on the first held-out day; SPY is bought the same day.</div>"
            + "".join(figs) + "".join(html_parts) + "</main></body></html>")
    return page, "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    from src.backtest.market import build_market_data
    from src.data.store import DataStore
    from src.logging_setup import setup_logging

    parser = argparse.ArgumentParser(prog="python -m src.validation.heldout")
    parser.add_argument("--settings", default=None)
    parser.add_argument("--strategies", nargs="+", required=True)
    args = parser.parse_args(argv)
    load_dotenv_if_present()
    settings = load_settings(args.settings)
    setup_logging(Path(settings["project"]["log_dir"]) / "heldout.log")
    check_heldout_allowed(settings, args.strategies)
    syms = [settings["benchmark"]]
    for n in args.strategies:
        syms += [s for s in build_strategy(n, settings).symbols() if s not in syms]
    market = build_market_data(DataStore(settings), syms)
    runs = run_heldout(args.strategies, settings, market, TrialLog(settings["validation"]["trial_log"]))
    page, md = render(runs, downturns_2022(runs, market, settings), settings)
    out = PROJECT_ROOT / settings["report"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (out / f"heldout_{stamp}.html").write_text(page, encoding="utf-8")
    (out / f"heldout_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {out / f'heldout_{stamp}.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
