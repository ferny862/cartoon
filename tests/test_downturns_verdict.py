from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import CASH
from src.reporting.downturns import (
    defensive_symbols, describe_downturn_row, downturn_table, exits_and_whipsaws, exposure, spy_episode,
)
from src.reporting.names import display_name
from src.reporting.verdict import HOLD_SPY, LOWER_DD, PREFERABLE, evaluate, verdict_text


def test_spy_episode_picks_deepest_with_trough_in_range():
    idx = pd.bdate_range("2000-01-03", periods=10)
    eq = pd.Series([100, 90, 100, 110, 60, 70, 115, 100, 120, 121], index=idx, dtype=float)
    ep = spy_episode(eq, idx[0], idx[-1])
    assert ep["peak"] == idx[3] and ep["trough"] == idx[4] and ep["recovery"] == idx[6]
    assert ep["depth"] == pytest.approx(60 / 110 - 1)
    assert spy_episode(eq, idx[1], idx[1])["trough"] == idx[1]
    assert spy_episode(eq, "2010-01-01", "2010-12-31") is None


def test_exits_reentries_and_whipsaw():
    idx = pd.bdate_range("2008-01-01", periods=6)
    expo = pd.Series([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], index=idx)
    spy = pd.Series([100, 90, 80, 85, 95, 90], index=idx, dtype=float)
    # exit at 90 -> re-entry at 85 (not a whipsaw); exit at 95 -> re-entry at 90 (not a whipsaw)
    assert exits_and_whipsaws(expo, spy, 0.3) == (2, 2, 0)
    spy2 = pd.Series([100, 90, 80, 95, 95, 99], index=idx, dtype=float)
    # exit at 90 -> re-entry at 95 (whipsaw); exit at 95 -> re-entry at 99 (whipsaw)
    assert exits_and_whipsaws(expo, spy2, 0.3) == (2, 2, 2)
    partial = pd.Series([1.0, 2 / 3, 2 / 3, 1.0, 1.0, 1.0], index=idx)
    assert exits_and_whipsaws(partial, spy, 0.3) == (1, 1, 0)      # one-third sleeve moves count


def test_exposure_excludes_cash_and_bonds():
    w = pd.DataFrame({"SPY": [1.0, 0.0], "AGG": [0.0, 1.0], "BIL": [0.0, 0.0], CASH: [0.0, 0.0]})
    defensive = defensive_symbols({"cash": "BIL", "bonds": "AGG", "us": "SPY"})
    assert defensive == {"BIL", "AGG", CASH}
    assert exposure(w, defensive).tolist() == [1.0, 0.0]


def _run(eq, weights):
    return SimpleNamespace(pretax=SimpleNamespace(equity=eq, weights=weights))


def test_downturn_table_labels(settings):
    days = pd.bdate_range("2007-06-01", "2012-12-31")
    n = len(days)
    t = np.arange(n)
    crash = (days >= "2008-01-02") & (days <= "2009-03-09")
    spy_ret = np.where(crash, -0.0025, 0.0006)
    spy = pd.Series(100 * np.cumprod(1 + spy_ret), index=days)
    # The strategy sits in cash for the whole crash.
    in_cash = (days >= "2008-01-15") & (days <= "2009-06-01")
    s_ret = np.where(in_cash, 0.0001, spy_ret)
    strat = pd.Series(100 * np.cumprod(1 + s_ret), index=days)
    w = pd.DataFrame({"SPY": np.where(in_cash, 0.0, 1.0), "BIL": np.where(in_cash, 1.0, 0.0), CASH: 0.0}, index=days)
    bench_w = pd.DataFrame({"SPY": 1.0, CASH: 0.0}, index=days)
    runs = {"benchmark": _run(spy, bench_w), "trend_faber": _run(strat, w)}
    market = SimpleNamespace(prices=pd.DataFrame({"SPY": spy}))
    cfg = {**settings, "report": {**settings["report"],
                                  "downturns": {"gfc": ["2007-06-01", "2009-12-31"], "bear_2022": ["2022-01-01", "2022-12-31"]}}}
    t = downturn_table(runs, market, cfg, {"trend_faber": {"cash": "BIL", "risk_asset": "SPY"}})
    row = t[(t.downturn == "gfc") & (t.strategy == "trend_faber")].iloc[0]
    assert row["spy_drawdown"] < -0.5
    assert abs(row["drawdown"]) < 0.5 * abs(row["spy_drawdown"])
    assert row["exits"] == 1 and row["reentries"] == 1
    assert row["label"] in ("Avoided", "Avoided, but whipsawed")
    assert "BIL" in row["holdings_during"]
    assert t[t.downturn == "bear_2022"].iloc[0]["status"] == "held out (locked)"
    sentence = describe_downturn_row(row, display_name)
    assert sentence.startswith("Trend filter (Faber) fell") and "avoided" in sentence


def _inputs(sharpe_low=0.1, dsr=0.97, pbo=0.3, smaller=4, windows=5, cagr=0.08, bench_cagr=0.085,
            at=0.06, bench_at=0.065):
    summaries = pd.DataFrame([
        {"strategy": "benchmark", "cagr_pretax": bench_cagr, "benchmark_cagr_pretax": bench_cagr,
         "cagr_aftertax_liquidated": bench_at, "benchmark_cagr_aftertax_liquidated": bench_at},
        {"strategy": "gem", "cagr_pretax": cagr, "benchmark_cagr_pretax": bench_cagr,
         "cagr_aftertax_liquidated": at, "benchmark_cagr_aftertax_liquidated": bench_at},
    ])
    dsr_t = pd.DataFrame([{"strategy": "gem", "dsr": dsr}])
    boot = pd.DataFrame([{"strategy": "gem", "sharpe_diff_ci_low": sharpe_low}])
    cons = pd.DataFrame([{"strategy": "gem", "windows": windows, "smaller_drawdown": smaller}])
    return summaries, dsr_t, boot, cons, pbo


CRIT = {"sharpe_ci_low_above": 0.0, "min_dsr": 0.95, "max_pbo_main": 0.5, "drawdown_majority": True,
        "max_cagr_shortfall": 0.01}


def test_verdict_all_pass_is_preferable():
    s, d, b, c, p = _inputs()
    row = evaluate(s, d, b, c, p, CRIT).iloc[0]
    assert row["verdict_ira"] == PREFERABLE and row["verdict_taxable"] == PREFERABLE
    assert row["failed_checks"] == ""


@pytest.mark.parametrize("kw,expected", [
    ({"sharpe_low": -0.05}, LOWER_DD),             # bootstrap range includes zero
    ({"dsr": 0.9}, LOWER_DD),
    ({"pbo": 0.6}, LOWER_DD),
    ({"smaller": 2}, HOLD_SPY),                     # not smaller drawdowns in most windows
    ({"cagr": 0.06}, HOLD_SPY),                     # trails by 2.5 points: fails return, passes stats
])
def test_verdict_failures(kw, expected):
    s, d, b, c, p = _inputs(**kw)
    assert evaluate(s, d, b, c, p, CRIT).iloc[0]["verdict_ira"] == expected


def test_taxable_verdict_uses_after_tax_return():
    s, d, b, c, p = _inputs(at=0.04, bench_at=0.065)
    row = evaluate(s, d, b, c, p, CRIT).iloc[0]
    assert row["verdict_ira"] == PREFERABLE and row["verdict_taxable"] == HOLD_SPY


def test_verdict_text_defaults_to_index_fund():
    s, d, b, c, p = _inputs(dsr=0.5, smaller=1)
    text = verdict_text(evaluate(s, d, b, c, p, CRIT), display_name)
    assert "buying and holding an S&P 500 index fund" in text[0]
    assert any(t.startswith("GEM:") and "Deflated Sharpe" in t for t in text)
    assert any("IRA" in t for t in text)


def test_verdict_reads_criteria_from_config(settings):
    crit = settings["report"]["verdict_criteria"]
    assert crit == CRIT


def test_lower_drawdown_wording_depends_on_return():
    s, d, b, c, p = _inputs(dsr=0.5, cagr=0.084)        # close return, fails DSR
    text = " ".join(verdict_text(evaluate(s, d, b, c, p, CRIT), display_name))
    assert "with a similar return" in text and "may suit" in text
    s, d, b, c, p = _inputs(dsr=0.5, cagr=0.03)         # far behind SPY
    text = " ".join(verdict_text(evaluate(s, d, b, c, p, CRIT), display_name))
    assert "clearly lower return" in text and "may suit" not in text
