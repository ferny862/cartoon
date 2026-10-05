"""All strategies end to end on a synthetic cached store."""

import pandas as pd
import pytest
import yaml

from src.backtest import __main__ as cli
from src.backtest.market import build_french_market_data, build_market_data
from src.backtest.runner import required_symbols, run_french, run_strategy
from src.strategies import strategy_names
from src.validation.trials import TrialLog
from tests.conftest import build_synthetic_store


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from src.config import load_settings
    return build_synthetic_store(tmp_path_factory.mktemp("synth"), load_settings())


@pytest.fixture(scope="module")
def runs(synth):
    store, settings = synth
    names = strategy_names(settings)
    market = build_market_data(store, required_symbols(names, settings))
    log = TrialLog(settings["validation"]["trial_log"])
    return {n: run_strategy(n, settings, market, trial_log=log) for n in names}, log, market


def test_start_dates_respect_inception_and_lookbacks(runs):
    r, _, _ = runs
    first = {n: run.weights.index[0] for n, run in r.items()}
    assert first["benchmark"] == pd.Timestamp("1998-01-30")
    assert first["trend_faber"] == pd.Timestamp("1998-10-30")          # 10 month-ends of SPY (data from Jan 1998)
    assert first["sector_mom_sector_filter"] == pd.Timestamp("2000-01-31")  # 14 month-ends from Dec 1998
    assert first["gem"] == pd.Timestamp("2003-09-30")                  # EFA has 12m by 2002; AGG (Sep 2003) only needs a price
    assert first["gtaa5"] == pd.Timestamp("2006-11-30")                # DBC (Feb 2006) needs 10 month-ends
    assert first["factor_blend"] == pd.Timestamp("2013-07-31")         # QUAL starts mid-July 2013


def test_no_results_in_heldout_period(runs):
    r, _, market = runs
    assert market.prices.index.max() < pd.Timestamp("2021-10-01")
    for run in r.values():
        assert run.pretax.end < pd.Timestamp("2021-10-01")


def test_benchmark_covers_same_period(runs):
    r, _, _ = runs
    for run in r.values():
        assert run.benchmark.start == run.pretax.start and run.benchmark.end == run.pretax.end


def test_summaries_are_complete(runs):
    r, _, _ = runs
    for name, run in r.items():
        s = run.summary
        assert s["strategy"] == name and s["cost_bps"] == 5
        for key in ("cagr_pretax", "cagr_aftertax_holding", "cagr_aftertax_liquidated", "sharpe",
                    "max_drawdown", "annual_turnover", "cost_drag", "beta"):
            assert key in s
        assert s["cagr_aftertax_liquidated"] <= s["cagr_pretax"] + 1e-12
    bench = r["benchmark"].summary
    assert bench["beta"] == pytest.approx(1.0) and bench["annual_turnover"] == 0


def test_sector_strategies_trade_more_than_trend(runs):
    r, _, _ = runs
    assert r["sector_mom_sector_filter"].summary["annual_turnover"] > r["trend_faber"].summary["annual_turnover"]


def test_trial_log_counts_distinct_configs(runs, synth):
    _, log, market = runs
    store, settings = synth
    assert log.n_trials() == 7
    run_strategy("gem", settings, market, cost_bps=10, taxable=False, trial_log=log)
    assert log.n_trials() == 7                     # cost sensitivity is not a new trial
    assert len(log.entries()) == 8
    assert log.n_trials(["gem"]) == 1


def test_french_sanity_runs(synth):
    store, settings = synth
    market = build_french_market_data(store)
    run = run_french("sector_mom_market_filter", settings, market)
    assert run.universe == "french_gross" and run.taxable is None and run.cost_bps == 0
    assert run.pretax.start.year == 1927
    assert run.benchmark.name.startswith("MKT")


def test_cli_run_and_french(synth, capsys, monkeypatch, tmp_path):
    store, settings = synth
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(cli, "load_dotenv_if_present", lambda: None)
    assert cli.main(["--settings", str(path), "run", "--strategies", "trend_faber", "--cost-bps", "2", "10", "--no-tax"]) == 0
    out = capsys.readouterr().out
    assert "trend_faber" in out
    assert cli.main(["--settings", str(path), "french"]) == 0
    assert "NON-INVESTABLE" in capsys.readouterr().out
    logs = list((tmp_path.parent).rglob("backtest_*.csv")) + list(__import__("pathlib").Path(settings["project"]["log_dir"]).glob("backtest_*.csv"))
    assert logs
