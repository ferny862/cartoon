"""The full validation suite on synthetic strategy runs."""

import pandas as pd
import pytest
import yaml

from src.backtest.market import build_market_data
from src.backtest.runner import required_symbols, run_strategy
from src.strategies import strategy_names
from src.validation import __main__ as vcli
from src.validation.suite import run_validation
from src.validation.trials import TrialLog
from tests.conftest import build_synthetic_store


@pytest.fixture(scope="module")
def report_and_settings(tmp_path_factory):
    from src.config import load_settings
    store, settings = build_synthetic_store(tmp_path_factory.mktemp("val"), load_settings())
    settings["validation"]["bootstrap"]["n_samples"] = 500
    names = strategy_names(settings)
    market = build_market_data(store, required_symbols(names, settings))
    trials = TrialLog(settings["validation"]["trial_log"])
    runs = {n: run_strategy(n, settings, market, taxable=False, trial_log=trials) for n in names}
    return run_validation(runs, market, settings, trials), settings, store


def test_dsr_uses_logged_trials(report_and_settings):
    rep, _, _ = report_and_settings
    assert rep.n_trials == 7
    assert set(rep.dsr["strategy"]) == {"benchmark", "trend_faber", "gem", "gtaa5", "factor_blend",
                                        "sector_mom_sector_filter", "sector_mom_market_filter"}
    assert rep.dsr["dsr"].between(0, 1).all()
    assert (rep.dsr["dsr"] <= rep.dsr["psr_vs_zero"] + 1e-12).all()   # deflation never helps


def test_pbo_candidate_sets(report_and_settings):
    rep, _, _ = report_and_settings
    main, secondary = rep.pbo["main"], rep.pbo["secondary"]
    assert main.n_strategies == 6 and secondary.n_strategies == 7
    assert main.start == pd.Timestamp("2007-01-31")        # GTAA trades from 1 Dec 2006; first full month Jan
    assert secondary.start == pd.Timestamp("2013-09-30")   # factor blend trades from 1 Aug 2013; first full month Sep
    assert main.n_obs > secondary.n_obs
    assert 0 <= main.pbo <= 1


def test_windows_regimes_and_bootstrap(report_and_settings):
    rep, _, _ = report_and_settings
    bear = rep.regimes[(rep.regimes.window == "bear_2000_2002")].set_index("strategy")["status"]
    assert bear["sector_mom_sector_filter"] == "partial" and bear["gem"] == "not live"
    row = rep.regimes[(rep.regimes.window == "bear_2000_2002") & (rep.regimes.strategy == "sector_mom_sector_filter")]
    assert row["live_from"].iloc[0] == pd.Timestamp("2000-02-01").date()
    assert (rep.regimes[rep.regimes.window == "bear_2022"]["status"] == "held out (locked)").all()
    assert set(rep.windows["window"]) == {"2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2021"}
    assert "benchmark" not in set(rep.bootstrap["strategy"])
    assert (rep.bootstrap["return_diff_ci_low"] <= rep.bootstrap["return_diff_ci_high"]).all()
    assert any("Sharpe" in e for e in rep.explanations) and any("PBO" in e for e in rep.explanations)


def test_save_writes_tables(report_and_settings, tmp_path):
    rep, _, _ = report_and_settings
    out = rep.save(tmp_path / "v")
    for f in ("windows", "expanding", "regimes", "consistency", "dsr", "bootstrap", "pbo"):
        assert (out / f"{f}.csv").exists()
    assert "Configurations tried" in (out / "summary.md").read_text()


def test_cli(report_and_settings, tmp_path, monkeypatch, capsys):
    _, settings, _ = report_and_settings
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(vcli, "load_dotenv_if_present", lambda: None)
    assert vcli.main(["--settings", str(path), "--no-tax"]) == 0
    assert "PBO" in capsys.readouterr().out
