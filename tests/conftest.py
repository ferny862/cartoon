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
