"""Kenneth French Data Library: industry portfolios and factor returns.

These are GROSS, NON-INVESTABLE paper portfolios (no fees, costs or taxes,
and they could not have been bought as funds for most of their history).
They are used only as long-history sanity checks back to 1926.

The library ships zipped CSV files containing several stacked tables
(value-weighted monthly, equal-weighted monthly, annual, ...). ``parse_french_csv``
splits them into named tables of monthly or annual rows.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

LABEL = "GROSS, NON-INVESTABLE (Kenneth French Data Library)"
_DAILY = re.compile(r"^\d{8}$")
_MONTHLY = re.compile(r"^\d{6}$")
_ANNUAL = re.compile(r"^\d{4}$")
_MISSING = (-99.99, -999.0)


def parse_french_csv(text: str) -> dict[str, pd.DataFrame]:
    """Split a French-library CSV into tables keyed by their title line.

    Values are converted from percent to decimal. Missing markers (-99.99,
    -999) become NaN. Daily tables are indexed by trading date, monthly
    tables by month-end dates, annual tables by year-end dates. Each frame
    carries ``attrs['label']``.
    """
    lines = text.splitlines()
    tables: dict[str, pd.DataFrame] = {}
    title = "untitled"
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(",") and i + 1 < len(lines):
            header = [h.strip() for h in line.split(",")[1:]]
            rows: list[tuple[str, list[float]]] = []
            j = i + 1
            while j < len(lines):
                parts = [p.strip() for p in lines[j].split(",")]
                if not parts or not (_DAILY.match(parts[0]) or _MONTHLY.match(parts[0]) or _ANNUAL.match(parts[0])):
                    break
                vals = [float(p) if p else np.nan for p in parts[1 : 1 + len(header)]]
                rows.append((parts[0], vals))
                j += 1
            if rows:
                key = title
                n = 2
                while key in tables:
                    key = f"{title} ({n})"
                    n += 1
                tables[key] = _to_frame(rows, header)
            i = j
            continue
        if line:
            title = line
        i += 1
    return tables


def _to_frame(rows: list[tuple[str, list[float]]], header: list[str]) -> pd.DataFrame:
    stamps = [r[0] for r in rows]
    if len(stamps[0]) == 8:
        index = pd.to_datetime(stamps, format="%Y%m%d")
    elif len(stamps[0]) == 6:
        index = pd.to_datetime(stamps, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        index = pd.to_datetime(stamps, format="%Y") + pd.offsets.YearEnd(0)
    df = pd.DataFrame([r[1] for r in rows], index=pd.DatetimeIndex(index, name="date"), columns=header)
    df = df.replace(list(_MISSING), np.nan) / 100.0
    df.attrs["label"] = LABEL
    return df


def select_monthly_table(tables: dict[str, pd.DataFrame], contains: str | None = None) -> pd.DataFrame:
    """Pick the first monthly table, optionally one whose title contains ``contains``."""
    for title, df in tables.items():
        is_monthly = len(df) > 1 and 20 < (df.index[1] - df.index[0]).days < 40
        if is_monthly and (contains is None or contains.lower() in title.lower()):
            out = df.copy()
            out.attrs["label"] = LABEL
            out.attrs["table"] = title
            return out
    raise KeyError(f"No monthly table matching {contains!r}; tables: {list(tables)}")


def select_daily_table(tables: dict[str, pd.DataFrame], contains: str | None = None) -> pd.DataFrame:
    """Pick the first daily table, optionally one whose title contains ``contains``."""
    for title, df in tables.items():
        is_daily = len(df) > 1 and (df.index[1] - df.index[0]).days <= 5
        if is_daily and (contains is None or contains.lower() in title.lower()):
            out = df.copy()
            out.attrs["label"] = LABEL
            out.attrs["table"] = title
            return out
    raise KeyError(f"No daily table matching {contains!r}; tables: {list(tables)}")


def industry_returns_daily(text: str) -> pd.DataFrame:
    """Daily value-weighted industry returns from a daily industry-portfolio CSV."""
    return select_daily_table(parse_french_csv(text), "Value Weighted Returns -- Daily")


def factor_returns_daily(text: str) -> pd.DataFrame:
    """Daily Fama-French factors (Mkt-RF, SMB, HML, RF)."""
    return select_daily_table(parse_french_csv(text))


def industry_returns(text: str) -> pd.DataFrame:
    """Monthly value-weighted industry returns from an industry-portfolio CSV."""
    return select_monthly_table(parse_french_csv(text), "Value Weighted Returns -- Monthly")


def factor_returns(text: str) -> pd.DataFrame:
    """Monthly Fama-French factors (Mkt-RF, SMB, HML, RF) from the factors CSV."""
    return select_monthly_table(parse_french_csv(text))


def read_zipped_csv(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(name).decode("latin-1")


def fetch_french_dataset(
    name: str,
    base_url: str = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp",
    session: Any | None = None,
    timeout: float = 60.0,
) -> str:
    """Download ``<name>_CSV.zip`` and return the CSV text."""
    session = session or requests.Session()
    url = f"{base_url.rstrip('/')}/{name}_CSV.zip"
    resp = session.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"French library returned HTTP {resp.status_code} for {name}")
    log.info("French library: downloaded %s", name)
    return read_zipped_csv(resp.content)
