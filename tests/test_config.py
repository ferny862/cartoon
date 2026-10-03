import logging

import pandas as pd
import pytest

from src.config import MissingSecretError, get_secret, heldout_start, inception_dates, instruments, redact
from src.logging_setup import RedactSecretsFilter


def test_settings_reflect_approved_decisions(settings):
    assert settings["project"]["starting_capital"] == 100000
    assert settings["benchmark"] == "SPY"
    assert settings["cash"]["etf"] == "BIL"
    assert heldout_start(settings) == pd.Timestamp("2021-10-01")
    assert settings["dates"]["heldout_unlocked"] is False
    assert settings["paper"]["dry_run"] is True
    assert settings["paper"]["base_url"] == "https://paper-api.alpaca.markets"
    assert settings["costs"]["spread_slippage_bps"] == 5


def test_sector_strategy_defaults(settings):
    for name in ("sector_mom_sector_filter", "sector_mom_market_filter"):
        s = settings["strategies"][name]
        assert s["lookback_months"] == 12 and s["skip_months"] == 1 and s["top_n"] == 3
        assert s["sectors"] == ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def test_disabled_instruments_excluded_by_default(settings):
    enabled = instruments(settings)
    assert "XLRE" not in enabled and "XLC" not in enabled and "VEU" not in enabled
    assert "XLRE" in instruments(settings, include_disabled=True)
    assert inception_dates(settings)["SPY"] == pd.Timestamp("1993-01-22")


def test_every_strategy_symbol_has_an_instrument(settings):
    known = set(settings["instruments"])
    for strat in settings["strategies"].values():
        syms = []
        for key in ("symbol", "risk_asset", "us", "intl", "bonds", "cash", "market"):
            if key in strat:
                syms.append(strat[key])
        syms += strat.get("assets", []) + strat.get("sectors", [])
        assert set(syms) <= known


def test_missing_secret_raises_without_echoing(monkeypatch):
    with pytest.raises(MissingSecretError, match="TIINGO_API_KEY"):
        get_secret("TIINGO_API_KEY")
    assert get_secret("TIINGO_API_KEY", required=False) is None


def test_redact_masks_secret_values(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "supersecretkey123")
    assert "supersecretkey123" not in redact("url?token=supersecretkey123")


def test_logging_filter_masks_secrets(monkeypatch, caplog):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "abcd-secret-9999")
    logger = logging.getLogger("test.redact")
    handler = logging.Handler()
    records = []
    handler.emit = records.append
    handler.addFilter(RedactSecretsFilter())
    logger.addHandler(handler)
    try:
        logger.warning("key is %s", "abcd-secret-9999")
    finally:
        logger.removeHandler(handler)
    assert "abcd-secret-9999" not in records[0].getMessage()
    assert "REDACTED" in records[0].getMessage()
