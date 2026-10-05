"""Paper-trading safety guards. No network: the trading client is always faked."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.paper.client import PaperBroker
from src.paper.safety import (
    PAPER_BASE_URL, PaperSafetyError, check_can_submit, check_paper_settings, check_paper_url,
)

SRC = Path(__file__).resolve().parent.parent / "src"
GOOD_ENV = {"PAPER_TRADING": "true"}


def fake_factory(base_url=PAPER_BASE_URL):
    def make(key, secret):
        return SimpleNamespace(_base_url=base_url, get_account=lambda: SimpleNamespace(
            equity="1000", cash="10", buying_power="20", status="ACTIVE", trading_blocked=False))
    return make


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "PKTESTKEY123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret-abcdef")


@pytest.mark.parametrize("url", [
    "https://api.alpaca.markets",                    # live endpoint
    "https://api.alpaca.markets/v2",
    "http://paper-api.alpaca.markets",               # not https
    "https://paper-api.alpaca.markets.evil.example",
    "https://broker-api.alpaca.markets",
    "",
])
def test_refuses_non_paper_url(url):
    with pytest.raises(PaperSafetyError):
        check_paper_url(url)


def test_accepts_paper_url_variants():
    assert check_paper_url("https://paper-api.alpaca.markets/") == PAPER_BASE_URL
    assert check_paper_url(" HTTPS://PAPER-API.ALPACA.MARKETS ") == PAPER_BASE_URL


@pytest.mark.parametrize("env", [{}, {"PAPER_TRADING": "false"}, {"PAPER_TRADING": "1"}, {"PAPER_TRADING": ""}])
def test_refuses_without_paper_flag(settings, env):
    with pytest.raises(PaperSafetyError, match="PAPER_TRADING"):
        check_paper_settings(settings, env)


def test_settings_with_live_url_refused(settings):
    bad = {**settings, "paper": {**settings["paper"], "base_url": "https://api.alpaca.markets"}}
    with pytest.raises(PaperSafetyError, match="not the Alpaca paper endpoint"):
        check_paper_settings(bad, GOOD_ENV)


def test_broker_refuses_live_url_before_touching_keys(settings, monkeypatch):
    called = []
    bad = {**settings, "paper": {**settings["paper"], "base_url": "https://api.alpaca.markets"}}
    with pytest.raises(PaperSafetyError):
        PaperBroker(bad, GOOD_ENV, client_factory=lambda k, s: called.append(1))
    assert called == []                                   # no client was ever created


def test_broker_rechecks_client_url(settings, keys):
    with pytest.raises(PaperSafetyError, match="not the paper endpoint"):
        PaperBroker(settings, GOOD_ENV, client_factory=fake_factory("https://api.alpaca.markets"))


def test_broker_starts_on_paper(settings, keys):
    broker = PaperBroker(settings, GOOD_ENV, client_factory=fake_factory())
    acct = broker.account()
    assert acct.equity == 1000.0 and acct.cash == 10.0
    assert "secret" not in repr(broker) and "PKTEST" not in repr(broker)


def test_broker_needs_keys(settings, monkeypatch):
    from src.config import MissingSecretError
    with pytest.raises(MissingSecretError, match="ALPACA_API_KEY"):
        PaperBroker(settings, GOOD_ENV, client_factory=fake_factory())


def test_real_client_factory_is_paper(settings, keys):
    """The real alpaca-py client is built for the paper endpoint (no network call is made)."""
    pytest.importorskip("alpaca")
    broker = PaperBroker(settings, GOOD_ENV)
    assert broker._client._base_url == PAPER_BASE_URL


def test_submission_needs_config_and_flag(settings):
    with pytest.raises(PaperSafetyError, match="dry_run"):
        check_can_submit(settings, execute_flag=True)        # default config is dry run
    live_cfg = {**settings, "paper": {**settings["paper"], "dry_run": False}}
    with pytest.raises(PaperSafetyError, match="--execute"):
        check_can_submit(live_cfg, execute_flag=False)
    check_can_submit(live_cfg, execute_flag=True)             # both given: allowed


def test_default_config_is_safe(settings):
    p = settings["paper"]
    assert p["dry_run"] is True and p["base_url"] == PAPER_BASE_URL and p["strategy"] is None


def _source_text():
    return {path: path.read_text(encoding="utf-8") for path in SRC.rglob("*.py")}


def test_no_code_path_to_live_endpoint():
    for path, text in _source_text().items():
        for m in re.finditer(r"([\w-]*)api\.alpaca\.markets", text):
            assert m.group(1) == "paper-", f"{path} references a non-paper Alpaca host"
        assert "TRADING_LIVE" not in text, path
        assert not re.search(r"paper\s*=\s*False", text), path


def test_no_robinhood_anywhere_in_code():
    for path, text in _source_text().items():
        assert "robinhood" not in text.lower(), path
