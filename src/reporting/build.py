"""Gather every result the report needs, then render HTML and Markdown.

``build_report(settings, store)`` runs the strategies (base cost with taxes,
plus pre-tax cost-sensitivity runs), the validation suite, the downturn
analysis, the French appendix and the pre-registered verdict.
``assemble_report`` does the same from runs already computed (used by tests).
"""

from __future__ import annotations

import html
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.backtest.market import build_french_market_data, build_market_data
from src.backtest.runner import required_symbols, run_french, run_strategy
from src.config import heldout_start
from src.reporting import charts
from src.reporting.downturns import describe_downturn_row, downturn_name, downturn_table
from src.reporting.names import display_name
from src.reporting.verdict import evaluate, verdict_text
from src.strategies import strategy_names, strategy_params
from src.validation.suite import ValidationReport, run_validation
from src.validation.trials import TrialLog

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"


@dataclass
class ReportData:
    generated: str
    in_sample: bool
    data_start: str
    data_end: str
    settings: dict
    summaries: pd.DataFrame
    sensitivity: pd.DataFrame
    validation: ValidationReport
    downturns: pd.DataFrame
    downturn_sentences: dict[str, list[str]]
    verdict_table: pd.DataFrame
    verdict_paragraphs: list[str]
    charts: dict[str, dict[str, str]]
    french_summaries: pd.DataFrame | None
    notes: dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------- formatting

def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (x is pd.NaT)


def fmt(x, kind: str) -> str:
    if _missing(x):
        return "–"
    if kind == "pct":
        return f"{x:.1%}"
    if kind == "pct_signed":
        return f"{x:+.1%}"
    if kind == "num":
        return f"{x:.2f}"
    if kind == "num_signed":
        return f"{x:+.2f}"
    if kind == "int":
        return f"{int(x):,}"
    if kind == "prob":
        return f"{x:.0%}"
    if kind == "name":
        return display_name(str(x))
    if kind == "bool":
        return "pass" if x else "fail"
    return str(x)


def html_table(df: pd.DataFrame, columns: list[tuple[str, str, str]], caption: str | None = None) -> str:
    """Render selected columns of ``df`` as an escaped HTML table."""
    numeric = {"pct", "pct_signed", "num", "num_signed", "int", "prob"}
    head = "".join(f'<th class="{"n" if k in numeric else ""}">{html.escape(label)}</th>' for _, label, k in columns)
    body = []
    for _, row in df.iterrows():
        cells = "".join(
            f'<td class="{"n" if k in numeric else ""}">{html.escape(fmt(row.get(key), k))}</td>'
            for key, _, k in columns)
        body.append(f"<tr>{cells}</tr>")
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return (f'<div class="table-wrap"><table>{cap}<thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label, _ in columns) + " |",
             "|" + "---|" * len(columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row.get(key), k) for key, _, k in columns) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------- assembly

def _summaries_frame(runs: dict, validation: ValidationReport) -> pd.DataFrame:
    df = pd.DataFrame([r.summary for r in runs.values()])
    dsr = validation.dsr.set_index("strategy")["dsr"] if len(validation.dsr) else pd.Series(dtype=float)
    df["dsr"] = df["strategy"].map(dsr)
    pbo_main = validation.pbo.get("main")
    df["pbo_main"] = pbo_main.pbo if pbo_main else float("nan")
    return df


def assemble_report(runs: dict, market, settings: dict, trial_log: TrialLog | None,
                    sensitivity_runs: list | None = None, french_runs: dict | None = None,
                    render_charts: bool = True) -> ReportData:
    validation = run_validation(runs, market, settings, trial_log)
    summaries = _summaries_frame(runs, validation)
    params = {n: strategy_params(n, settings) for n in runs}
    downturns = downturn_table(runs, market, settings, params)
    sentences: dict[str, list[str]] = {}
    for dname, grp in downturns.groupby("downturn", sort=False):
        live = grp[grp["strategy"].notna()]
        sentences[dname] = [describe_downturn_row(r, display_name) for _, r in live.iterrows()]
    pbo_main = validation.pbo.get("main")
    vtable = evaluate(summaries, validation.dsr, validation.bootstrap, validation.consistency,
                      pbo_main.pbo if pbo_main else None, settings["report"]["verdict_criteria"])
    in_sample = not settings["dates"].get("heldout_unlocked", False)
    paragraphs = verdict_text(vtable, display_name, in_sample=in_sample)

    sens = pd.DataFrame([r.summary for r in (sensitivity_runs or [])] + [r.summary for r in runs.values()])
    french = pd.DataFrame([r.summary for r in french_runs.values()]) if french_runs else None

    chart_set: dict[str, dict[str, str]] = {}
    if render_charts:
        chart_set = {
            "equity": charts.both_themes(charts.equity_chart, runs),
            "drawdowns": charts.both_themes(charts.drawdown_chart, runs),
            "rolling_excess": charts.both_themes(charts.rolling_excess_chart, runs),
            "annual_returns": charts.both_themes(charts.annual_returns_chart, runs),
            "allocation": charts.both_themes(charts.allocation_chart, runs),
        }
        if french_runs:
            chart_set["french"] = charts.both_themes(charts.french_chart, french_runs)

    idx = market.prices.index
    return ReportData(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        in_sample=in_sample,
        data_start=str(idx[0].date()), data_end=str(idx[-1].date()),
        settings=settings, summaries=summaries, sensitivity=sens, validation=validation,
        downturns=downturns, downturn_sentences=sentences, verdict_table=vtable,
        verdict_paragraphs=paragraphs, charts=chart_set, french_summaries=french,
        notes=dict(market.notes),
    )


def build_report(settings: dict, store) -> ReportData:
    names = strategy_names(settings)
    market = build_market_data(store, required_symbols(names, settings))
    trials = TrialLog(settings["validation"]["trial_log"])
    base = settings["costs"]["spread_slippage_bps"]
    runs = {n: run_strategy(n, settings, market, cost_bps=base, taxable=True, trial_log=trials) for n in names}
    sensitivity = [run_strategy(n, settings, market, cost_bps=bps, taxable=False, trial_log=trials)
                   for bps in settings["costs"]["sensitivity_bps"] if bps != base for n in names]
    french_runs = None
    try:
        fmarket = build_french_market_data(store)
        french_runs = {f"{n}__french": run_french(n, settings, fmarket, trials)
                       for n in settings["french_sanity"]["strategies"]}
    except FileNotFoundError as exc:
        log.warning("French data not cached; appendix omitted (%s)", exc)
    return assemble_report(runs, market, settings, trials, sensitivity, french_runs)


# ----------------------------------------------------------------- rendering

HEADLINE = [
    ("strategy", "Strategy", "name"), ("start", "From", "text"), ("end", "To", "text"),
    ("cagr_pretax", "CAGR", "pct"), ("benchmark_cagr_pretax", "SPY CAGR", "pct"),
    ("volatility", "Volatility", "pct"), ("sharpe", "Sharpe", "num"), ("benchmark_sharpe", "SPY Sharpe", "num"),
    ("sortino", "Sortino", "num"), ("max_drawdown", "Max drawdown", "pct"),
    ("benchmark_max_drawdown", "SPY max drawdown", "pct"), ("max_drawdown_days", "Longest drawdown (days)", "int"),
    ("calmar", "Calmar", "num"), ("beta", "Beta", "num"), ("tracking_error", "Tracking error", "pct"),
    ("information_ratio", "Information ratio", "num"), ("annual_turnover", "Turnover / yr", "pct"),
    ("cost_drag", "Cost drag / yr", "pct"), ("cagr_aftertax_liquidated", "After-tax CAGR", "pct"),
    ("best_year", "Best year", "pct"), ("worst_year", "Worst year", "pct"),
    ("pct_5y_windows_beating_benchmark", "5-yr windows beating SPY", "prob"),
    ("dsr", "Deflated Sharpe", "prob"), ("pbo_main", "PBO (main set)", "prob"),
]
ACCOUNTS = [
    ("strategy", "Strategy", "name"), ("cagr_pretax", "Roth IRA (pre-tax)", "pct"),
    ("benchmark_cagr_pretax", "SPY", "pct"), ("cagr_traditional_ira", "Traditional IRA", "pct"),
    ("benchmark_cagr_traditional_ira", "SPY", "pct"), ("cagr_aftertax_holding", "Taxable, still holding", "pct"),
    ("cagr_aftertax_liquidated", "Taxable, liquidated", "pct"),
    ("benchmark_cagr_aftertax_liquidated", "SPY taxable, liquidated", "pct"),
    ("annual_turnover", "Turnover / yr", "pct"), ("wash_sale_flags", "Wash-sale flags", "int"),
]
WINDOWS = [
    ("strategy", "Strategy", "name"), ("window", "Window", "text"), ("status", "Status", "text"),
    ("live_from", "From", "text"), ("live_to", "To", "text"), ("cagr", "CAGR", "pct"),
    ("benchmark_cagr", "SPY CAGR", "pct"), ("excess_cagr", "Difference", "pct_signed"),
    ("max_drawdown", "Max drawdown", "pct"), ("benchmark_max_drawdown", "SPY max drawdown", "pct"),
]
DOWNTURN = [
    ("strategy", "Strategy", "name"), ("status", "Status", "text"), ("drawdown", "Fell", "pct"),
    ("days_to_recover", "Days to recover", "int"), ("exits", "Exits", "int"), ("reentries", "Re-entries", "int"),
    ("whipsaws", "Whipsaws", "int"), ("label", "Result", "text"),
    ("holdings_into", "Held going in", "text"), ("holdings_during", "Held on average, peak to trough", "text"),
]
DSR = [
    ("strategy", "Strategy", "name"), ("sharpe_annualized", "Sharpe (annualized)", "num"), ("n_obs", "Months", "int"),
    ("skew", "Skew", "num"), ("kurtosis", "Kurtosis", "num"), ("n_trials", "Configurations tried", "int"),
    ("psr_vs_zero", "P(Sharpe > 0)", "prob"), ("dsr", "Deflated Sharpe", "prob"),
]
PBO = [
    ("candidate_set", "Candidate set", "text"), ("n_strategies", "Candidates", "int"), ("n_obs", "Months", "int"),
    ("start", "From", "text"), ("end", "To", "text"), ("pbo", "PBO", "prob"),
    ("prob_oos_loss", "Selected trails T-bills out of sample", "prob"), ("degradation_slope", "Degradation slope", "num"),
]
BOOT = [
    ("strategy", "Strategy", "name"), ("n_obs", "Months", "int"), ("return_diff", "Return difference / yr", "pct_signed"),
    ("return_diff_ci_low", "95% low", "pct_signed"), ("return_diff_ci_high", "95% high", "pct_signed"),
    ("return_diff_prob_positive", "P(> 0)", "prob"), ("sharpe_diff", "Sharpe difference", "num_signed"),
    ("sharpe_diff_ci_low", "95% low", "num_signed"), ("sharpe_diff_ci_high", "95% high", "num_signed"),
    ("sharpe_diff_prob_positive", "P(> 0)", "prob"),
]
VERDICT = [
    ("strategy", "Strategy", "name"), ("sharpe_significant", "Bootstrap Sharpe > 0", "bool"),
    ("dsr_ok", "DSR ≥ 0.95", "bool"), ("pbo_ok", "PBO < 0.5", "bool"),
    ("drawdown_ok", "Smaller drawdowns", "bool"), ("return_ok_ira", "Return (IRA)", "bool"),
    ("return_ok_taxable", "Return (taxable)", "bool"), ("verdict_ira", "Verdict: IRA", "text"),
    ("verdict_taxable", "Verdict: taxable", "text"),
]
FRENCH = [
    ("strategy", "Strategy", "name"), ("start", "From", "text"), ("end", "To", "text"),
    ("cagr_pretax", "CAGR (gross)", "pct"), ("benchmark_cagr_pretax", "Market CAGR", "pct"),
    ("sharpe", "Sharpe", "num"), ("benchmark_sharpe", "Market Sharpe", "num"),
    ("max_drawdown", "Max drawdown", "pct"), ("benchmark_max_drawdown", "Market max drawdown", "pct"),
    ("annual_turnover", "Turnover / yr", "pct"),
]


def _sensitivity_table(sens: pd.DataFrame) -> str:
    if sens.empty:
        return ""
    rows = []
    for name, g in sens.groupby("strategy", sort=False):
        row = {"strategy": name}
        for _, r in g.iterrows():
            bps = int(r["cost_bps"])
            row[f"cagr_{bps}"] = r["cagr_pretax"]
            row[f"sharpe_{bps}"] = r["sharpe"]
        rows.append(row)
    df = pd.DataFrame(rows)
    bps_list = sorted({int(b) for b in sens["cost_bps"]})
    cols = [("strategy", "Strategy", "name")]
    cols += [(f"cagr_{b}", f"CAGR at {b} bp", "pct") for b in bps_list]
    cols += [(f"sharpe_{b}", f"Sharpe at {b} bp", "num") for b in bps_list]
    return html_table(df, cols)


def render_html(data: ReportData) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    template = env.get_template("report.html.j2")
    v = data.validation
    downturn_blocks = []
    for dname, grp in data.downturns.groupby("downturn", sort=False):
        first = grp.iloc[0]
        block = {"name": downturn_name(dname), "status": first.get("status")}
        if first.get("status") in ("held out (locked)", "no SPY drawdown found in data"):
            block["locked"] = True
        else:
            block.update({
                "locked": False,
                "spy": (f"SPY fell {abs(first['spy_drawdown']):.0%} from its peak on {first['spy_peak']} to "
                        f"{first['spy_trough']}; " + (f"it regained the peak on {first['spy_recovery']} "
                        f"({int(first['spy_days_to_recover'])} days)." if first["spy_recovered"]
                        else "it had not regained the peak by the end of the data.")),
                "table": html_table(grp[grp["strategy"].notna()], DOWNTURN),
                "sentences": data.downturn_sentences.get(dname, []),
            })
        downturn_blocks.append(block)
    crit = data.settings["report"]["verdict_criteria"]
    return template.render(
        d=data,
        heldout_start=str(heldout_start(data.settings).date()),
        headline=html_table(data.summaries, HEADLINE),
        accounts=html_table(data.summaries, ACCOUNTS),
        sensitivity=_sensitivity_table(data.sensitivity),
        windows=html_table(v.windows, WINDOWS),
        expanding=html_table(v.expanding, WINDOWS),
        regimes=html_table(v.regimes, WINDOWS),
        downturns=downturn_blocks,
        dsr=html_table(v.dsr, DSR),
        pbo=html_table(v.pbo_table(), PBO) if v.pbo else "<p>PBO could not be computed.</p>",
        bootstrap=html_table(v.bootstrap, BOOT),
        verdict_checks=html_table(data.verdict_table, VERDICT),
        french=html_table(data.french_summaries, FRENCH) if data.french_summaries is not None else None,
        criteria=crit,
        costs=data.settings["costs"], taxes=data.settings["taxes"],
    )


def render_markdown(data: ReportData) -> str:
    cols = [("strategy", "Strategy", "name"), ("start", "From", "text"), ("cagr_pretax", "CAGR", "pct"),
            ("benchmark_cagr_pretax", "SPY CAGR", "pct"), ("sharpe", "Sharpe", "num"),
            ("max_drawdown", "Max drawdown", "pct"), ("benchmark_max_drawdown", "SPY max drawdown", "pct"),
            ("cagr_aftertax_liquidated", "After-tax CAGR", "pct"), ("dsr", "Deflated Sharpe", "prob")]
    scope = (f"IN-SAMPLE ONLY: data before {heldout_start(data.settings).date()}" if data.in_sample
             else "INCLUDES THE HELD-OUT PERIOD")
    lines = [f"# ETF strategy research: summary", "",
             f"Generated {data.generated}. Data {data.data_start} to {data.data_end}. **{scope}.**", "",
             "## Headline results (each strategy vs SPY over the same period)", "",
             markdown_table(data.summaries, cols), "", "## Verdict", ""]
    lines += [f"- {p}" for p in data.verdict_paragraphs]
    lines += ["", "See the HTML report for charts, downturns, validation details and limitations."]
    return "\n".join(lines) + "\n"


def write_report(data: ReportData, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = data.generated[:10]
    html_path = out / f"report_{stamp}.html"
    md_path = out / f"summary_{stamp}.md"
    html_path.write_text(render_html(data), encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    return html_path, md_path
