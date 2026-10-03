"""FRED 3-month Treasury bill rate (series DTB3) and conversion to cash returns.

DTB3 is quoted as an annualized *discount* rate in percent on a 360-day
basis for a 91-day bill. For a quote ``d`` the bill price is
``P = 1 - d * 91 / 360`` and the 91-day holding return is ``1/P - 1``. We
convert that to a per-calendar-day growth factor ``g = (1/P) ** (1/91)`` and
accrue it over the calendar days between trading dates. The return earned
from t0 to t1 uses the rate known at t0, so there is no look-ahead.

The public fredgraph CSV endpoint needs no API key.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BILL_DAYS = 91


def parse_fred_csv(text: str, series: str = "DTB3") -> pd.Series:
    """Parse a fredgraph CSV. Missing values ('.' or blank) become NaN."""
    df = pd.read_csv(io.StringIO(text), na_values=["."], keep_default_na=True)
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    if series not in df.columns:
        raise ValueError(f"Series {series} not found in FRED CSV columns {list(df.columns)}")
    out = pd.Series(
        pd.to_numeric(df[series], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date"),
        name=series,
    )
    return out.sort_index()


def fetch_fred_series(
    series: str = "DTB3",
    url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv",
    session: Any | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    session = session or requests.Session()
    resp = session.get(url, params={"id": series}, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"FRED returned HTTP {resp.status_code} for {series}")
    s = parse_fred_csv(resp.text, series)
    log.info("FRED: downloaded %d observations of %s", s.notna().sum(), series)
    return s


def discount_to_daily_growth(discount_rate_pct: pd.Series | float) -> pd.Series | float:
    """Per-calendar-day growth factor implied by a 91-day bill discount rate in percent."""
    d = np.asarray(discount_rate_pct, dtype=float) / 100.0
    price = 1.0 - d * BILL_DAYS / 360.0
    growth = (1.0 / price) ** (1.0 / BILL_DAYS)
    if isinstance(discount_rate_pct, pd.Series):
        return pd.Series(growth, index=discount_rate_pct.index, name=discount_rate_pct.name)
    return float(growth)


def tbill_returns(discount_rate_pct: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Cash return earned from each trading date to the next.

    The value at ``dates[i]`` is the return from ``dates[i-1]`` to ``dates[i]``
    using the rate observed on or before ``dates[i-1]``. The first value is NaN.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    rates = discount_rate_pct.dropna().sort_index()
    # Rate known as of each date (latest observation on or before it).
    known = rates.reindex(rates.index.union(dates)).ffill().reindex(dates)
    growth = discount_to_daily_growth(known)
    prior_growth = growth.shift(1)
    days = pd.Series(dates, index=dates).diff().dt.days
    out = prior_growth ** days - 1.0
    out.name = "tbill_return"
    return out
