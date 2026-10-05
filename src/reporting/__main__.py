"""Build the HTML report and Markdown summary.

    python -m src.reporting

Runs every strategy (in-sample unless the held-out period is unlocked), the
validation suite, the downturn analysis and the French appendix, then writes
reports/output/report_<date>.html and summary_<date>.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, load_dotenv_if_present, load_settings
from src.data.store import DataStore
from src.logging_setup import setup_logging
from src.reporting.build import build_report, write_report

log = logging.getLogger("src.reporting")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.reporting")
    parser.add_argument("--settings", default=None)
    parser.add_argument("--out", default=None, help="output directory (default: report.output_dir)")
    args = parser.parse_args(argv)
    load_dotenv_if_present()
    settings = load_settings(args.settings)
    setup_logging(Path(settings["project"]["log_dir"]) / "report.log")
    if settings["dates"].get("heldout_unlocked"):
        log.warning("HELD-OUT PERIOD IS UNLOCKED: the report includes data from %s onward",
                    settings["dates"]["heldout_start"])
    data = build_report(settings, DataStore(settings))
    out = Path(args.out) if args.out else PROJECT_ROOT / settings["report"]["output_dir"]
    html_path, md_path = write_report(data, out)
    print("\n".join(data.verdict_paragraphs))
    print(f"\nReport: {html_path}\nSummary: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
