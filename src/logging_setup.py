"""Logging configuration with secret redaction."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import PROJECT_ROOT, redact


class RedactSecretsFilter(logging.Filter):
    """Masks the value of any secret environment variable in log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = None
        return True


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger for console and, optionally, a log file."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_project_handler", False):
            root.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
        handler.addFilter(RedactSecretsFilter())
        handler._project_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    return root
