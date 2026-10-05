"""Settings loading and secret handling.

Settings come from ``config/settings.yaml``. Secrets come only from
environment variables (optionally loaded from a local ``.env`` file) and are
never written to logs or returned in error messages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Every environment variable that holds a secret. Values of these are masked
# by the logging redaction filter.
SECRET_ENV_VARS = ("TIINGO_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "STOOQ_API_KEY")


class MissingSecretError(RuntimeError):
    """Raised when a required secret environment variable is not set."""


def load_settings(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML settings file into a plain dict."""
    path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    with open(path, encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    if not isinstance(settings, dict):
        raise ValueError(f"Settings file {path} did not parse to a mapping")
    return settings


def load_dotenv_if_present(path: str | os.PathLike | None = None) -> None:
    """Load a local .env file into the environment without overriding existing values."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=path or PROJECT_ROOT / ".env", override=False)


def get_secret(name: str, required: bool = True) -> str | None:
    """Return a secret from the environment.

    The error message names the variable but never echoes any value.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        if required:
            raise MissingSecretError(
                f"Environment variable {name} is not set. Copy .env.example to .env "
                f"and fill it in, or export it in your shell."
            )
        return None
    return value


def redact(text: str, extra_secrets: tuple[str, ...] = ()) -> str:
    """Replace any known secret value appearing in ``text`` with a mask."""
    secrets = [os.environ.get(v, "") for v in SECRET_ENV_VARS] + list(extra_secrets)
    for s in secrets:
        s = (s or "").strip()
        if len(s) >= 4:
            text = text.replace(s, "***REDACTED***")
    return text


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    inception: pd.Timestamp
    asset_class: str
    tax_character: str
    enabled: bool = True


def instruments(settings: dict[str, Any], include_disabled: bool = False) -> dict[str, Instrument]:
    """Parse the instruments table into Instrument records keyed by symbol."""
    out: dict[str, Instrument] = {}
    for symbol, spec in settings["instruments"].items():
        inst = Instrument(
            symbol=symbol,
            name=spec["name"],
            inception=pd.Timestamp(spec["inception"]),
            asset_class=spec["asset_class"],
            tax_character=spec["tax_character"],
            enabled=bool(spec.get("enabled", True)),
        )
        if inst.enabled or include_disabled:
            out[symbol] = inst
    return out


def inception_dates(settings: dict[str, Any]) -> dict[str, pd.Timestamp]:
    """Inception date for every instrument, enabled or not."""
    return {s: i.inception for s, i in instruments(settings, include_disabled=True).items()}


def heldout_start(settings: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(settings["dates"]["heldout_start"])
