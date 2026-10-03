import base64
import pickle

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import MarketData, run_backtest
from src.reporting import charts as c
from src.reporting.names import SLOT, display_name


@pytest.fixture(scope="module")
def runs():
    from types import SimpleNamespace
    days = pd.bdate_range("2005-01-03", "2012-12-31")
    rng = np.random.default_rng(0)
    px = pd.DataFrame({"SPY": 100 * np.cumprod(1 + rng.normal(3e-4, 0.01, len(days))),
                       "BIL": 100 * np.cumprod(np.full(len(days), 1.0001))}, index=days)
    md = MarketData(px, pd.Series(0.0, index=days))
    me = px.resample("BME").last().index
    me = me[me.isin(days)]
    bench = run_backtest(pd.DataFrame({"SPY": [1.0]}, index=[me[0]]), md)
    w = pd.DataFrame({"SPY": np.where(np.arange(len(me)) % 3, 1.0, 0.0)}, index=me)
    w["BIL"] = 1 - w["SPY"]
    strat = run_backtest(w, md)
    run = SimpleNamespace(pretax=strat, benchmark=bench)
    return {"benchmark": SimpleNamespace(pretax=bench, benchmark=bench), "trend_faber": run, "gem": run}


@pytest.mark.parametrize("fn", ["equity_chart", "drawdown_chart", "rolling_excess_chart", "annual_returns_chart",
                                "allocation_chart"])
def test_charts_render_png_in_both_themes(runs, fn):
    pair = c.both_themes(getattr(c, fn), runs)
    assert set(pair) == {"light", "dark"}
    for b64 in pair.values():
        assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"
    assert pair["light"] != pair["dark"]


def test_french_chart(runs):
    pair = c.both_themes(c.french_chart, {"sector_mom_market_filter__french": runs["gem"]})
    assert base64.b64decode(pair["dark"])[:4] == b"\x89PNG"


def test_color_follows_entity_not_rank():
    light = c.THEMES["light"]
    assert c.color_for("gem", light) == light["slots"][SLOT["gem"]]
    assert c.color_for("gem__french", light) == c.color_for("gem", light)
    assert c.color_for("benchmark", light) == light["secondary"]
    assert len(set(SLOT.values())) == len(SLOT) <= len(light["slots"])


def test_display_names():
    assert display_name("gem") == "GEM"
    assert display_name("sector_mom_market_filter__french").endswith("(French industries)")
