"""The one-time held-out evaluation (synthetic data, temporary held-out log)."""

import pandas as pd
import pytest

from src.backtest.market import build_market_data
from src.data.store import HeldOutLockedError
from src.validation.heldout import check_heldout_allowed, downturns_2022, heldout_weights, render, run_heldout
from tests.conftest import build_synthetic_store

NAMES = ["trend_faber", "sector_mom_market_filter"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from tests.conftest import locked_settings
    tmp = tmp_path_factory.mktemp("heldout")
    store, s = build_synthetic_store(tmp, locked_settings(), start="1998-01-02", end="2024-12-31")
    s["dates"] = {**s["dates"], "heldout_unlocked": True}
    store.settings = s
    s["validation"]["bootstrap"]["n_samples"] = 300
    log = tmp / "heldout_log.md"
    log.write_text("| 2026 | me | abc | `trend_faber` and `sector_mom_market_filter` |\n")
    market = build_market_data(store, ["SPY", "BIL", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"])
    return s, market, log


def test_refuses_while_locked(settings, tmp_path):
    log = tmp_path / "log.md"
    log.write_text("`trend_faber`")
    with pytest.raises(HeldOutLockedError, match="locked"):
        check_heldout_allowed(settings, ["trend_faber"], log)


def test_refuses_without_log_entry(env, tmp_path):
    s, _, _ = env
    empty = tmp_path / "log.md"
    empty.write_text("nothing recorded")
    with pytest.raises(HeldOutLockedError, match="does not record"):
        check_heldout_allowed(s, ["trend_faber"], empty)
    with pytest.raises(HeldOutLockedError):
        check_heldout_allowed(s, ["gem"], env[2])          # gem was not approved for the held-out test


def test_heldout_weights_start_from_last_prior_signal():
    idx = pd.to_datetime(["2021-07-30", "2021-08-31", "2021-09-30", "2021-10-29"])
    w = pd.DataFrame({"A": [1, 0, 1, 0], "B": [0, 1, 0, 1]}, index=idx, dtype=float)
    out = heldout_weights(w, pd.Timestamp("2021-10-01"))
    assert list(out.index) == [pd.Timestamp("2021-09-30"), pd.Timestamp("2021-10-29")]


def test_run_heldout_starts_on_first_heldout_day(env):
    s, market, log = env
    runs = run_heldout(NAMES, s, market, None, log_path=log)
    for name, r in runs.items():
        assert r.pretax.start == pd.Timestamp("2021-10-01")
        assert r.benchmark.start == r.pretax.start and r.benchmark.end == r.pretax.end
        assert r.pretax.equity.iloc[0] < 100_000                 # only the initial trading cost
        for key in ("cagr_pretax", "benchmark_cagr_pretax", "cagr_aftertax_liquidated",
                    "benchmark_cagr_aftertax_liquidated", "max_drawdown", "benchmark_max_drawdown"):
            assert key in r.summary
        assert r.bootstrap.n_obs >= 24
    dt = downturns_2022(runs, market, s)
    assert set(dt["downturn"]) == {"bear_2022"}
    page, md = render(runs, dt, s)
    assert "Held-out test" in page and "Out-of-sample" in page and "<script" not in page
    assert "| Maximum drawdown |" in md and "Trend filter (Faber)" in md
