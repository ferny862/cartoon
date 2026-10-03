"""Apply the pre-registered verdict criteria (config: report.verdict_criteria).

Per strategy and account type:
* "Preferable to holding SPY": every statistical test passes, drawdowns are
  smaller than SPY's in most evaluated windows, and CAGR is within the
  allowed shortfall of SPY's.
* "Lower drawdowns, but not distinguishable from luck": drawdowns are
  smaller in most windows, but a statistical test fails.
* "Hold SPY": anything else.
"""

from __future__ import annotations

import pandas as pd

PREFERABLE = "Preferable to holding SPY"
LOWER_DD = "Lower drawdowns, but not distinguishable from luck"
HOLD_SPY = "Hold SPY"


def evaluate(summaries: pd.DataFrame, dsr: pd.DataFrame, bootstrap: pd.DataFrame, consistency: pd.DataFrame,
             pbo_main: float | None, criteria: dict) -> pd.DataFrame:
    """One row per non-benchmark strategy with each check and the verdicts."""
    s = summaries.set_index("strategy")
    d = dsr.set_index("strategy") if len(dsr) else pd.DataFrame()
    b = bootstrap.set_index("strategy") if len(bootstrap) else pd.DataFrame()
    c = consistency.set_index("strategy") if len(consistency) else pd.DataFrame()
    rows = []
    for name in s.index:
        if name == "benchmark":
            continue
        r = s.loc[name]
        sharpe_low = b.loc[name, "sharpe_diff_ci_low"] if name in b.index else float("nan")
        dsr_val = d.loc[name, "dsr"] if name in d.index else float("nan")
        windows = int(c.loc[name, "windows"]) if name in c.index else 0
        smaller = int(c.loc[name, "smaller_drawdown"]) if name in c.index else 0
        checks = {
            "sharpe_significant": bool(sharpe_low > criteria["sharpe_ci_low_above"]),
            "dsr_ok": bool(dsr_val >= criteria["min_dsr"]),
            "pbo_ok": bool(pbo_main is not None and pbo_main < criteria["max_pbo_main"]),
            "drawdown_ok": bool(windows > 0 and smaller > windows / 2) if criteria.get("drawdown_majority", True) else True,
            "return_ok_ira": bool(r["cagr_pretax"] >= r["benchmark_cagr_pretax"] - criteria["max_cagr_shortfall"]),
        }
        if "cagr_aftertax_liquidated" in r and "benchmark_cagr_aftertax_liquidated" in r:
            checks["return_ok_taxable"] = bool(r["cagr_aftertax_liquidated"]
                                               >= r["benchmark_cagr_aftertax_liquidated"] - criteria["max_cagr_shortfall"])
        else:
            checks["return_ok_taxable"] = False
        statistical = checks["sharpe_significant"] and checks["dsr_ok"] and checks["pbo_ok"]

        def verdict(return_ok: bool) -> str:
            if statistical and checks["drawdown_ok"] and return_ok:
                return PREFERABLE
            if checks["drawdown_ok"] and not statistical:
                return LOWER_DD
            return HOLD_SPY

        failed = [k for k, v in checks.items() if not v]
        rows.append({"strategy": name, **checks, "statistical_tests_pass": statistical,
                     "windows_evaluated": windows, "windows_smaller_drawdown": smaller,
                     "sharpe_diff_ci_low": sharpe_low, "dsr": dsr_val, "pbo_main": pbo_main,
                     "verdict_ira": verdict(checks["return_ok_ira"]),
                     "verdict_taxable": verdict(checks["return_ok_taxable"]),
                     "failed_checks": ", ".join(failed) if failed else ""})
    return pd.DataFrame(rows)


CHECK_TEXT = {
    "sharpe_significant": "its Sharpe-ratio advantage over SPY is not clearly above zero in the bootstrap",
    "dsr_ok": "its Deflated Sharpe Ratio is below the 0.95 bar",
    "pbo_ok": "the overfitting test (PBO) says picking the best backtest here is unreliable",
    "drawdown_ok": "it did not have smaller drawdowns than SPY in most periods",
    "return_ok_ira": "its pre-tax return trailed SPY by more than the allowed 1 point a year",
    "return_ok_taxable": "its after-tax return trailed SPY by more than the allowed 1 point a year",
}


def verdict_text(table: pd.DataFrame, display, in_sample: bool = True) -> list[str]:
    """Plain-English verdict paragraphs."""
    out: list[str] = []
    pref_ira = table[table["verdict_ira"] == PREFERABLE]["strategy"].tolist()
    pref_tax = table[table["verdict_taxable"] == PREFERABLE]["strategy"].tolist()
    lower = table[(table["verdict_ira"] == LOWER_DD) | (table["verdict_taxable"] == LOWER_DD)]["strategy"].tolist()
    scope = "on in-sample data (before October 2021)" if in_sample else "including the held-out period"
    if not pref_ira and not pref_tax:
        out.append(f"No strategy met the pre-registered bar {scope}. Based on this evidence, simply buying and "
                   f"holding an S&P 500 index fund is the better choice: it is cheaper, simpler, more tax-efficient, "
                   f"and none of the rules-based alternatives showed an advantage that can be told apart from luck.")
    else:
        if pref_ira:
            out.append("Preferable to SPY in an IRA (pre-tax): " + ", ".join(display(n) for n in pref_ira) + ".")
        if pref_tax:
            out.append("Preferable to SPY in a taxable account: " + ", ".join(display(n) for n in pref_tax) + ".")
        out.append(f"These results are {scope}. They become a recommendation only if they also hold in the "
                   f"held-out period, which has not been evaluated yet.")
    if lower:
        lt = table.set_index("strategy")
        close = [n for n in lower if lt.loc[n, "return_ok_ira"]]
        costly = [n for n in lower if not lt.loc[n, "return_ok_ira"]]
        if close:
            out.append("Lower drawdowns than SPY in most periods with a similar return, but not distinguishable "
                       "from luck: " + ", ".join(display(n) for n in close)
                       + ". These may suit an investor who values a smoother ride, accepting that the evidence is weak.")
        if costly:
            out.append("Lower drawdowns than SPY in most periods, but at the cost of a clearly lower return, and not "
                       "distinguishable from luck: " + ", ".join(display(n) for n in costly) + ".")
    for _, r in table.iterrows():
        if r["verdict_ira"] == PREFERABLE and r["verdict_taxable"] == PREFERABLE:
            continue
        reasons = [CHECK_TEXT[k] for k in r["failed_checks"].split(", ") if k in CHECK_TEXT]
        if reasons:
            out.append(f"{display(r['strategy'])}: " + "; ".join(reasons) + ".")
    out.append("The sector-rotation strategies trade more often and realize mostly short-term gains, so if used at "
               "all they are better suited to an IRA than a taxable account.")
    return out
