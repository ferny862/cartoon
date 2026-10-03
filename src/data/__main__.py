"""Command-line entry point for the data layer.

    python -m src.data download [--symbols SPY XLK] [--refresh] [--no-crosscheck]
    python -m src.data validate

``validate`` writes a CSV and a Markdown summary to the log directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_dotenv_if_present, load_settings
from src.data import validation as v
from src.data.store import NS_TIINGO, NS_YAHOO, DataStore
from src.logging_setup import setup_logging

log = logging.getLogger("src.data")


def cmd_download(store: DataStore, args: argparse.Namespace) -> int:
    symbols = args.symbols or store.enabled_symbols()
    ref = store.settings["data"]["reference_calendar_symbol"]
    if ref not in symbols:
        symbols = [ref] + symbols
    store.download_prices(symbols, refresh=args.refresh)
    store.download_fred(refresh=args.refresh)
    store.download_french(refresh=args.refresh)
    if not args.no_crosscheck:
        try:
            store.download_crosscheck(symbols, refresh=args.refresh)
        except ImportError:
            log.warning("yfinance is not installed; skipping cross-check download")
    log.info("Download complete")
    return 0


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([str(i)] + [str(x) for x in row]) + " |" for i, row in zip(df.index, df.to_numpy())]
    return "\n".join(lines)


def cmd_validate(store: DataStore, args: argparse.Namespace) -> int:
    cfg = store.settings["data"]["validation"]
    calendar = store.calendar()
    issues: list[v.Issue] = []
    crosscheck_frames = []
    for sym in store.enabled_symbols():
        if not store.cache.exists(NS_TIINGO, sym):
            issues.append(v.Issue(sym, "not_downloaded", "error", "no cached Tiingo data"))
            continue
        prices = store.raw_prices(sym)
        issues += v.validate_prices(sym, prices, calendar, store.inception.get(sym), cfg)
        if store.cache.exists(NS_YAHOO, sym):
            cc, daily = v.compare_sources(sym, prices, store.raw_prices(sym, NS_YAHOO),
                                          cfg["crosscheck_daily_tolerance"], cfg["crosscheck_monthly_tolerance"])
            issues += cc
            if len(daily):
                crosscheck_frames.append(daily.assign(symbol=sym))
        else:
            issues.append(v.Issue(sym, "crosscheck", "info", "no yfinance data cached; not cross-checked"))

    log_dir = PROJECT_ROOT / store.settings["project"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    frame = v.issues_frame(issues)
    frame.to_csv(log_dir / f"data_validation_{stamp}.csv", index=False)
    if crosscheck_frames:
        pd.concat(crosscheck_frames).to_csv(log_dir / f"crosscheck_daily_{stamp}.csv")
    summary = frame.groupby(["symbol", "severity"]).size().unstack(fill_value=0) if len(frame) else frame
    md = [f"# Data validation {stamp}", "", f"{len(frame)} issues found.", ""]
    if len(frame):
        md += ["## Counts by symbol and severity", "", _markdown_table(summary), ""]
        for sev in ("error", "warning", "info"):
            md += ["", f"## {sev.capitalize()}s", ""]
            md += [f"- **{r.symbol}** `{r.check}` {r.date or ''}: {r.message}"
                   for r in frame.itertuples() if r.severity == sev] or ["None."]
    (log_dir / f"data_validation_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    n_err = int((frame["severity"] == "error").sum()) if len(frame) else 0
    log.info("Validation finished: %d issues (%d errors); see %s", len(frame), n_err, log_dir)
    return 1 if n_err else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.data")
    parser.add_argument("--settings", default=None, help="path to settings.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download prices, T-bill rates and French data")
    dl.add_argument("--symbols", nargs="*")
    dl.add_argument("--refresh", action="store_true", help="ignore the cache")
    dl.add_argument("--no-crosscheck", action="store_true", help="skip the yfinance cross-check download")
    sub.add_parser("validate", help="run data-quality checks on cached data")
    args = parser.parse_args(argv)

    load_dotenv_if_present()
    settings = load_settings(args.settings)
    setup_logging(Path(settings["project"]["log_dir"]) / "data.log")
    store = DataStore(settings)
    return {"download": cmd_download, "validate": cmd_validate}[args.command](store, args)


if __name__ == "__main__":
    sys.exit(main())
