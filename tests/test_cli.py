"""End-to-end run of `python -m src.data validate` on synthetic cached data."""

import pandas as pd
import yaml

from src.data import __main__ as cli
from src.data.cache import Cache
from src.data.store import NS_TIINGO, NS_YAHOO
from tests.conftest import make_prices


def test_validate_writes_reports(tmp_path, settings, monkeypatch):
    s = {**settings}
    s["data"] = {**settings["data"], "cache_dir": str(tmp_path / "cache")}
    s["project"] = {**settings["project"], "log_dir": str(tmp_path / "logs")}
    s["instruments"] = {k: settings["instruments"][k] for k in ("SPY", "XLK")}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(s))

    cal = pd.bdate_range("1999-01-04", "2000-12-29")
    cache = Cache(tmp_path / "cache")
    spy = make_prices(pd.bdate_range("1998-01-02", "2000-12-29"), seed=1)
    cache.write(NS_TIINGO, "SPY", spy)
    cache.write(NS_YAHOO, "SPY", spy)
    xlk = make_prices(cal, seed=2)
    xlk.iloc[100:, xlk.columns.get_loc("adj_close")] *= 1.25   # extreme move
    cache.write(NS_TIINGO, "XLK", xlk)

    monkeypatch.setattr(cli, "load_dotenv_if_present", lambda: None)
    code = cli.main(["--settings", str(path), "validate"])
    assert code == 0  # warnings only, no errors

    out = pd.read_csv(next((tmp_path / "logs").glob("data_validation_*.csv")))
    xlk_checks = set(out.loc[out.symbol == "XLK", "check"])
    assert "extreme_move" in xlk_checks and "stale_data" in xlk_checks
    assert "crosscheck" in xlk_checks          # noted as not cross-checked
    assert (tmp_path / "logs" / "data.log").exists()
    assert any((tmp_path / "logs").glob("data_validation_*.md"))
