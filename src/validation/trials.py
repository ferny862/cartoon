"""Trial log: every strategy configuration ever run, appended to a JSONL file.

The Deflated Sharpe Ratio and the Probability of Backtest Overfitting need
the true number of configurations tried. A *trial* is a distinct set of
strategy parameters (signal logic). Re-running the same parameters with a
different cost assumption, account type or data universe is recorded too,
but does not count as a new trial because the signals are identical.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


def config_hash(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TrialLog:
    def __init__(self, path: str | Path):
        path = Path(path)
        self.path = path if path.is_absolute() else PROJECT_ROOT / path

    def record(self, strategy: str, params: dict[str, Any], context: dict[str, Any] | None = None) -> dict:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strategy": strategy,
            "config_hash": config_hash(params),
            "params": params,
            "context": context or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def n_trials(self, strategies: list[str] | None = None, universe: str | None = None) -> int:
        """Distinct parameter sets run, optionally limited to some strategies or one data universe."""
        hashes = {
            e["config_hash"] for e in self.entries()
            if (strategies is None or e["strategy"] in strategies)
            and (universe is None or e.get("context", {}).get("universe") == universe)
        }
        return len(hashes)
