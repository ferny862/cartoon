"""Hard safety rules for the paper-trading phase.

1. The client refuses to start unless PAPER_TRADING is "true" and the base
   URL is exactly the Alpaca paper endpoint.
2. There is no code path to a live endpoint: the only URL this package
   knows is the paper URL, and the broker client is always constructed with
   ``paper=True`` and that URL, then checked again after construction.
3. Order submission additionally needs ``paper.dry_run: false`` in the
   settings AND an explicit ``--execute`` flag.

Any future live module must live elsewhere, behind its own explicitly named
environment flag and an interactive confirmation, disabled by default.
"""

from __future__ import annotations

import os

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
PAPER_FLAG = "PAPER_TRADING"


class PaperSafetyError(RuntimeError):
    """Raised whenever a paper-trading safety rule is not satisfied."""


def _normalize(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def check_paper_url(url: str) -> str:
    if _normalize(url) != PAPER_BASE_URL:
        raise PaperSafetyError(
            f"Refusing to start: base URL {url!r} is not the Alpaca paper endpoint {PAPER_BASE_URL}. "
            "Live trading is not available in this project."
        )
    return PAPER_BASE_URL


def check_paper_flag(environ: dict | None = None) -> None:
    env = os.environ if environ is None else environ
    if str(env.get(PAPER_FLAG, "")).strip().lower() != "true":
        raise PaperSafetyError(f"Refusing to start: set the environment variable {PAPER_FLAG}=true for paper trading.")


def check_paper_settings(settings: dict, environ: dict | None = None) -> str:
    """Validate every start-up rule; returns the paper URL to use."""
    paper = settings.get("paper", {})
    if paper.get("broker") != "alpaca":
        raise PaperSafetyError("Only the Alpaca paper broker is supported.")
    check_paper_flag(environ)
    return check_paper_url(paper.get("base_url", ""))


def check_can_submit(settings: dict, execute_flag: bool) -> None:
    """Orders go out only when the settings AND the command line both ask for it."""
    if settings.get("paper", {}).get("dry_run", True):
        raise PaperSafetyError("Dry run: paper.dry_run is true in config/settings.yaml, so no orders are submitted.")
    if not execute_flag:
        raise PaperSafetyError("Dry run: pass --execute to submit paper orders.")
