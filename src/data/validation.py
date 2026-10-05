"""Data validation: missing days, bad prices, extreme moves, stale data,
inception mismatches and cross-source disagreements.

Checks never modify data. Each returns a list of ``Issue`` records which the
CLI writes to a log so a human can review them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Issue:
    symbol: str
    check: str
    severity: str          # "error" | "warning" | "info"
    message: str
    date: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def check_missing_days(symbol: str, prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> list[Issue]:
    """Reference-calendar trading days missing between the symbol's first and last row."""
    if prices.empty:
        return [Issue(symbol, "missing_days", "error", "no data")]
    window = calendar[(calendar >= prices.index.min()) & (calendar <= prices.index.max())]
    missing = window.difference(prices.index)
    if len(missing) == 0:
        return []
    sample = ", ".join(str(d.date()) for d in missing[:10])
    more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
    return [Issue(symbol, "missing_days", "warning", f"{len(missing)} trading days missing: {sample}{more}", str(missing[0].date()))]


def check_calendar_gaps(symbol: str, index: pd.DatetimeIndex, max_gap_business_days: int) -> list[Issue]:
    """Gaps between consecutive rows longer than ``max_gap_business_days`` business days."""
    issues: list[Issue] = []
    if len(index) < 2:
        return issues
    starts = index[:-1].values.astype("datetime64[D]")
    ends = index[1:].values.astype("datetime64[D]")
    gaps = np.busday_count(starts, ends) - 1  # business days strictly between rows
    for k in np.nonzero(gaps > max_gap_business_days)[0]:
        issues.append(Issue(symbol, "calendar_gap", "warning",
                            f"{int(gaps[k])} business days without data after {index[k].date()}",
                            str(index[k].date())))
    return issues


def check_bad_prices(symbol: str, prices: pd.DataFrame) -> list[Issue]:
    """Zero, negative or NaN closes."""
    issues: list[Issue] = []
    for col in ("close", "adj_close"):
        if col not in prices:
            continue
        s = prices[col]
        bad = s[(s <= 0)]
        if len(bad):
            issues.append(Issue(symbol, "nonpositive_price", "error",
                                f"{len(bad)} rows with {col} <= 0", str(bad.index[0].date())))
        nan = s[s.isna()]
        if len(nan):
            issues.append(Issue(symbol, "nan_price", "error",
                                f"{len(nan)} rows with missing {col}", str(nan.index[0].date())))
    return issues


def check_extreme_moves(symbol: str, prices: pd.DataFrame, threshold: float) -> list[Issue]:
    """Daily adjusted returns whose absolute value exceeds ``threshold``."""
    rets = prices["adj_close"].pct_change(fill_method=None)
    hits = rets[rets.abs() > threshold]
    return [Issue(symbol, "extreme_move", "warning", f"daily return {r:+.2%}", str(d.date()))
            for d, r in hits.items()]


def check_stale_runs(symbol: str, prices: pd.DataFrame, run_days: int) -> list[Issue]:
    """Runs of at least ``run_days`` consecutive identical raw closes."""
    s = prices["close"]
    changed = s.ne(s.shift()).cumsum()
    run_lengths = s.groupby(changed).transform("size")
    issues = []
    for _, grp in s[run_lengths >= run_days].groupby(changed[run_lengths >= run_days]):
        issues.append(Issue(symbol, "stale_run", "warning",
                            f"close unchanged at {grp.iloc[0]:.4f} for {len(grp)} consecutive days",
                            str(grp.index[0].date())))
    return issues


def check_staleness(symbol: str, prices: pd.DataFrame, as_of: pd.Timestamp, max_business_days: int) -> list[Issue]:
    """Last row older than ``max_business_days`` business days before ``as_of``."""
    last = prices.index.max()
    lag = int(np.busday_count(np.datetime64(last.date()), np.datetime64(pd.Timestamp(as_of).date())))
    if lag > max_business_days:
        return [Issue(symbol, "stale_data", "warning",
                      f"last observation {last.date()} is {lag} business days old", str(last.date()))]
    return []


def check_inception(symbol: str, prices: pd.DataFrame, inception: pd.Timestamp, tolerance_days: int) -> list[Issue]:
    """First observation should be close to the recorded inception date."""
    first = prices.index.min()
    diff = (first - pd.Timestamp(inception)).days
    if diff < -tolerance_days:
        return [Issue(symbol, "pre_inception_data", "error",
                      f"data starts {first.date()}, {-diff} days before recorded inception {inception.date()}; "
                      f"pre-inception rows will be dropped", str(first.date()))]
    if diff > tolerance_days:
        return [Issue(symbol, "late_first_observation", "warning",
                      f"data starts {first.date()}, {diff} days after recorded inception {inception.date()}",
                      str(first.date()))]
    return []


def compare_sources(
    symbol: str,
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    daily_tolerance: float,
    monthly_tolerance: float,
) -> tuple[list[Issue], pd.DataFrame]:
    """Compare adjusted returns from two sources over their common dates.

    Daily differences are expected around dividend ex-dates because providers
    adjust slightly differently; monthly differences should be small.
    Returns issues plus a frame of all daily differences above tolerance.
    """
    common = primary.index.intersection(secondary.index)
    if len(common) < 2:
        return [Issue(symbol, "crosscheck", "warning", "fewer than 2 overlapping dates")], pd.DataFrame()
    a = primary.loc[common, "adj_close"]
    b = secondary.loc[common, "adj_close"]
    ra, rb = a.pct_change(fill_method=None), b.pct_change(fill_method=None)
    diff = (ra - rb).abs()
    daily_bad = pd.DataFrame({"primary": ra, "secondary": rb, "abs_diff": diff})[diff > daily_tolerance]

    ma = a.resample("ME").last().pct_change(fill_method=None)
    mb = b.resample("ME").last().pct_change(fill_method=None)
    mdiff = (ma - mb).abs().dropna()
    monthly_bad = mdiff[mdiff > monthly_tolerance]

    issues: list[Issue] = []
    if len(daily_bad):
        issues.append(Issue(symbol, "crosscheck_daily", "info",
                            f"{len(daily_bad)} of {len(common) - 1} days differ by more than {daily_tolerance:.2%}; "
                            f"max {diff.max():.2%}", str(daily_bad.index[0].date())))
    for d, v in monthly_bad.items():
        issues.append(Issue(symbol, "crosscheck_monthly", "warning",
                            f"monthly return differs by {v:.2%} between sources", str(d.date())))
    if not issues:
        log.info("Cross-check %s: sources agree on %d overlapping days", symbol, len(common))
    return issues, daily_bad


def validate_prices(
    symbol: str,
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    inception: pd.Timestamp | None,
    cfg: dict,
    as_of: pd.Timestamp | None = None,
) -> list[Issue]:
    """Run every single-source check for one symbol."""
    if prices.empty:
        return [Issue(symbol, "empty", "error", "no rows")]
    as_of = pd.Timestamp.today().normalize() if as_of is None else as_of
    issues: list[Issue] = []
    issues += check_bad_prices(symbol, prices)
    issues += check_missing_days(symbol, prices, calendar)
    issues += check_calendar_gaps(symbol, prices.index, cfg["max_gap_business_days"])
    issues += check_extreme_moves(symbol, prices, cfg["extreme_move_threshold"])
    issues += check_stale_runs(symbol, prices, cfg["stale_run_days"])
    issues += check_staleness(symbol, prices, as_of, cfg["max_staleness_business_days"])
    if inception is not None:
        issues += check_inception(symbol, prices, inception, cfg["inception_tolerance_days"])
    return issues


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame([i.as_dict() for i in issues], columns=["symbol", "check", "severity", "message", "date"])
