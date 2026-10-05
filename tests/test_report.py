"""The report pipeline end to end on synthetic data."""

import re

import pandas as pd
import pytest
import yaml

from src.reporting import __main__ as rcli
from src.reporting.build import build_report, fmt, html_table, render_html, render_markdown, write_report
from tests.conftest import build_synthetic_store


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from src.config import load_settings
    store, settings = build_synthetic_store(tmp_path_factory.mktemp("rep"), load_settings())
    settings["validation"]["bootstrap"]["n_samples"] = 300
    return build_report(settings, store), settings


def test_formatting_helpers():
    assert fmt(0.1234, "pct") == "12.3%" and fmt(-0.05, "pct_signed") == "-5.0%"
    assert fmt(float("nan"), "num") == "–" and fmt(None, "pct") == "–"
    assert fmt(True, "bool") == "pass" and fmt("gem", "name") == "GEM"
    t = html_table(pd.DataFrame([{"strategy": "<b>x</b>", "v": 0.5}]), [("strategy", "S", "text"), ("v", "V", "pct")])
    assert "&lt;b&gt;" in t and 'class="n"' in t


def test_report_sections_and_content(built):
    data, settings = built
    page = render_html(data)
    for anchor in ("headline", "charts", "accounts", "costs", "periods", "downturns", "overfitting",
                   "french", "verdict", "limits"):
        assert f'id="{anchor}"' in page
    assert "In-sample results only" in page
    assert page.count('media="(prefers-color-scheme: dark)"') == 6          # 5 charts + French
    assert "2022 bear market" in page and "Held out (locked)" in page
    assert "GROSS, NON-INVESTABLE" in page.upper()
    assert not re.search(r"\bnan\b", page)
    assert not re.search(r"\d\.0 days", page)
    assert "written down before" in page
    assert "<script" not in page                                              # fully static


def test_report_tables_cover_requested_metrics(built):
    data, _ = built
    s = data.summaries
    for col in ("cagr_pretax", "volatility", "sharpe", "sortino", "max_drawdown", "max_drawdown_days", "calmar",
                "beta", "tracking_error", "information_ratio", "annual_turnover", "cost_drag",
                "cagr_aftertax_liquidated", "best_year", "worst_year", "pct_5y_windows_beating_benchmark",
                "dsr", "pbo_main", "benchmark_cagr_aftertax_liquidated"):
        assert col in s.columns, col
    assert sorted(set(data.sensitivity["cost_bps"])) == [2, 5, 10]
    assert data.french_summaries is not None and len(data.french_summaries) == 2


def test_downturns_and_verdict(built):
    data, _ = built
    dt = data.downturns
    assert set(dt["downturn"]) == {"dotcom_2000_2002", "gfc_2008_2009", "covid_2020", "bear_2022"}
    sector_2000 = dt[(dt.downturn == "dotcom_2000_2002") & (dt.strategy == "sector_mom_sector_filter")].iloc[0]
    assert sector_2000["status"] in ("ok", "partial") and sector_2000["label"] in (
        "Avoided", "Avoided, but whipsawed", "Whipsawed", "Neither")
    assert set(data.verdict_table["strategy"]) == {"trend_faber", "gem", "gtaa5", "factor_blend",
                                                   "sector_mom_sector_filter", "sector_mom_market_filter"}
    assert data.verdict_paragraphs


def test_markdown_and_write(built, tmp_path):
    data, _ = built
    md = render_markdown(data)
    assert md.startswith("# ETF strategy research") and "## Verdict" in md and "IN-SAMPLE ONLY" in md
    html_path, md_path = write_report(data, tmp_path)
    assert html_path.exists() and md_path.exists() and html_path.stat().st_size > 100_000


def test_cli_writes_report(built, tmp_path, monkeypatch, capsys):
    _, settings = built
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(rcli, "load_dotenv_if_present", lambda: None)
    data, _ = built
    monkeypatch.setattr(rcli, "build_report", lambda s, store: data)   # the pipeline is covered above
    assert rcli.main(["--settings", str(path), "--out", str(tmp_path / "out")]) == 0
    assert "Report:" in capsys.readouterr().out
    assert list((tmp_path / "out").glob("report_*.html"))
