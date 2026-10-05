"""Common daily price-frame schema shared by all price sources.

Every source returns a DataFrame indexed by a tz-naive, normalized
``DatetimeIndex`` named ``date`` with these float columns:

close         raw (unadjusted) close
adj_close     close adjusted for splits and dividends (total return)
div_cash      cash dividend per share with ex-date on this day (0 if none)
split_factor  split ratio effective on this day (1 if none)
volume        raw share volume
"""

from __future__ import annotations

import pandas as pd

PRICE_COLUMNS = ["close", "adj_close", "div_cash", "split_factor", "volume"]


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame to the common schema: sorted, de-duplicated, float columns."""
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        # Drop the timezone but keep the exchange-local wall-clock date.
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            out[col] = {"div_cash": 0.0, "split_factor": 1.0}.get(col, float("nan"))
    out = out[PRICE_COLUMNS].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
