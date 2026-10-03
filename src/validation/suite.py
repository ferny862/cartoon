"""Run every validation check on a set of strategy runs.

Inputs are the in-sample ``StrategyRun`` objects from the runner (pre-tax
results; taxes do not change the signals). Outputs are tables plus
plain-English explanations, saved as CSV/Markdown for the step 5 report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import MarketData
from src.config import heldout_start
from src.validation.bootstrap import bootstrap_difference, explain_bootstrap
from src.validation.common import monthly_excess, monthly_returns, monthly_risk_free
from src.validation.dsr import deflated_sharpe_ratio, explain_dsr, sharpe_per_period
from src.validation.pbo import PBOResult, explain_pbo, pbo_cscv
from src.validation.trials import TrialLog
from src.validation.walkforward import consistency, expanding_windows, fixed_windows, regime_table

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    windows: pd.DataFrame
    expanding: pd.DataFrame
    regimes: pd.DataFrame
    consistency: pd.DataFrame
    dsr: pd.DataFrame
    pbo: dict[str, PBOResult]
    bootstrap: pd.DataFrame
    n_trials: int
    explanations: list[str] = field(default_factory=list)

    def pbo_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"candidate_set": k, **v.summary()} for k, v in self.pbo.items()])

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("windows", "expanding", "regimes", "consistency", "dsr", "bootstrap"):
            getattr(self, name).to_csv(out / f"{name}.csv", index=False)
        self.pbo_table().to_csv(out / "pbo.csv", index=False)
        lines = ["# Validation summary (in-sample)", "",
                 f"Configurations tried (ETF universe, from the trial log): **{self.n_trials}**", ""]
        lines += [f"- {e}" for e in self.explanations]
        (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out


def run_validation(runs: dict, market: MarketData, settings: dict, trial_log: TrialLog | None = None) -> ValidationReport:
    v = settings["validation"]
    rf = market.risk_free
    windows = fixed_windows(runs, [tuple(w) for w in v["walk_forward"]["windows"]], rf)
    expanding = expanding_windows(runs, int(v["walk_forward"]["expanding_step_years"]), rf)
    regimes = regime_table(runs, {k: tuple(x) for k, x in settings["regimes"].items()}, rf,
                           heldout_start(settings), bool(settings["dates"].get("heldout_unlocked", False)))
    explanations: list[str] = []

    # --- Deflated Sharpe Ratio -------------------------------------------------
    excess = {n: monthly_excess(r.pretax.equity, rf).iloc[1:] for n, r in runs.items()}   # drop partial first month
    srs = np.array([sharpe_per_period(e) for e in excess.values()])
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    logged = trial_log.n_trials(universe=v.get("dsr", {}).get("trial_universe", "etf")) if trial_log else 0
    n_trials = max(logged, len(runs))
    dsr_rows = []
    for n, e in excess.items():
        res = deflated_sharpe_ratio(e, n_trials, var_sr)
        dsr_rows.append({"strategy": n, **res.as_dict()})
        explanations.append(explain_dsr(n, res))
    dsr = pd.DataFrame(dsr_rows)

    # --- Probability of Backtest Overfitting -----------------------------------
    pbo: dict[str, PBOResult] = {}
    for label, spec in v["pbo"]["candidate_sets"].items():
        names = [n for n in spec["strategies"] if n in runs]
        if len(names) < 2:
            continue
        matrix = pd.DataFrame({n: excess[n] for n in names})
        try:
            res = pbo_cscv(matrix, int(v["pbo"]["n_partitions"]))
        except ValueError as exc:
            log.warning("PBO %s skipped: %s", label, exc)
            explanations.append(f"PBO ({label}) could not be computed: {exc}")
            continue
        pbo[label] = res
        explanations.append(explain_pbo(f"PBO ({label}: {spec.get('note', '')})", res))

    # --- Block bootstrap vs SPY ------------------------------------------------
    bcfg = v["bootstrap"]
    boot_rows = []
    for n, r in runs.items():
        if n == "benchmark":
            continue
        s = monthly_returns(r.pretax.equity).iloc[1:]
        b = monthly_returns(r.benchmark.equity).iloc[1:]
        mrf = monthly_risk_free(rf)
        try:
            res = bootstrap_difference(s, b, mrf, block=int(bcfg["block_months"]),
                                       n_samples=int(bcfg["n_samples"]), seed=int(bcfg["seed"]))
        except ValueError as exc:
            explanations.append(f"Bootstrap for {n} skipped: {exc}")
            continue
        boot_rows.append({"strategy": n, **res.as_dict()})
        explanations.append(explain_bootstrap(n, res))
    bootstrap = pd.DataFrame(boot_rows)

    return ValidationReport(windows, expanding, regimes, consistency(windows), dsr, pbo, bootstrap,
                            n_trials, explanations)
