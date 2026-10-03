"""Shared fixtures. All network access is blocked during tests."""

from __future__ import annotations

import socket

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests; mock the call instead")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture(autouse=True)
def _no_real_secrets(monkeypatch):
    for name in ("TIINGO_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "STOOQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings():
    return load_settings()


def make_prices(
    dates: pd.DatetimeIndex,
    start: float = 100.0,
    daily_return: float = 0.0005,
    seed: int | None = None,
) -> pd.DataFrame:
    """Synthetic frame in the common price schema."""
    n = len(dates)
    if seed is None:
        rets = np.full(n, daily_return)
    else:
        rets = np.random.default_rng(seed).normal(daily_return, 0.01, n)
    rets[0] = 0.0
    adj = start * np.cumprod(1 + rets)
    return pd.DataFrame(
        {"close": adj, "adj_close": adj, "div_cash": 0.0, "split_factor": 1.0, "volume": 1e6},
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def bdays():
    return pd.bdate_range("2020-01-01", "2020-12-31")


def build_synthetic_store(tmp_path, settings, start="1998-01-02", end="2022-12-30", seed=11):
    """A DataStore whose cache holds random-walk prices for every enabled
    instrument (from its inception date), a flat 3% T-bill rate, and synthetic
    French daily data. Returns (store, settings) with paths pointed at tmp_path."""
    from src.config import instruments
    from src.data.cache import Cache
    from src.data.store import NS_FRED, NS_FRENCH, NS_TIINGO, DataStore

    cache = Cache(tmp_path / "cache")
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    for k, (sym, inst) in enumerate(instruments(settings).items()):
        d = days[days >= inst.inception]
        rets = rng.normal(2.5e-4, 0.011, len(d))
        rets[0] = 0.0
        px = 50 * np.cumprod(1 + rets)
        df = pd.DataFrame({"close": px, "adj_close": px, "div_cash": 0.0, "split_factor": 1.0, "volume": 1e6},
                          index=pd.DatetimeIndex(d, name="date"))
        qe = df.resample("BQE").last().index
        df.loc[df.index.isin(qe), "div_cash"] = df.loc[df.index.isin(qe), "close"] * 0.004
        cache.write(NS_TIINGO, sym, df)
    cache.write(NS_FRED, "DTB3", pd.Series(3.0, index=days, name="DTB3").to_frame())
    fdays = pd.bdate_range("1926-07-01", "1950-12-29")
    cols = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
    ind = pd.DataFrame(rng.normal(3e-4, 0.012, (len(fdays), 10)), index=fdays, columns=cols)
    fac = pd.DataFrame({"Mkt-RF": ind.mean(axis=1) - 1e-4, "SMB": 0.0, "HML": 0.0, "RF": 1e-4}, index=fdays)
    cache.write(NS_FRENCH, "industries_10_daily", ind)
    cache.write(NS_FRENCH, "factors_daily", fac)
    s = {**settings,
         "data": {**settings["data"], "cache_dir": str(tmp_path / "cache")},
         "project": {**settings["project"], "log_dir": str(tmp_path / "logs")},
         "validation": {**settings["validation"], "trial_log": str(tmp_path / "logs" / "trials.jsonl")}}
    return DataStore(s, cache=cache, tiingo=object(), yahoo=object()), s
