"""Local Parquet cache with a JSON metadata sidecar per file."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT


class Cache:
    """Stores DataFrames as ``<root>/<namespace>/<key>.parquet``."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.root = root if root.is_absolute() else PROJECT_ROOT / root

    def _path(self, namespace: str, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self.root / namespace / f"{safe}.parquet"

    def _meta_path(self, namespace: str, key: str) -> Path:
        return self._path(namespace, key).with_suffix(".meta.json")

    def exists(self, namespace: str, key: str) -> bool:
        return self._path(namespace, key).exists()

    def write(self, namespace: str, key: str, df: pd.DataFrame, meta: dict[str, Any] | None = None) -> Path:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        info = {
            "downloaded_at": time.time(),
            "rows": int(len(df)),
            "first": str(df.index.min()) if len(df) else None,
            "last": str(df.index.max()) if len(df) else None,
        }
        info.update(meta or {})
        self._meta_path(namespace, key).write_text(json.dumps(info, indent=2, default=str))
        return path

    def read(self, namespace: str, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(namespace, key))

    def metadata(self, namespace: str, key: str) -> dict[str, Any]:
        path = self._meta_path(namespace, key)
        return json.loads(path.read_text()) if path.exists() else {}

    def is_fresh(self, namespace: str, key: str, max_age_days: float, now: float | None = None) -> bool:
        if not self.exists(namespace, key):
            return False
        downloaded = self.metadata(namespace, key).get("downloaded_at")
        if downloaded is None:
            return False
        now = time.time() if now is None else now
        return (now - float(downloaded)) < max_age_days * 86400

    def keys(self, namespace: str) -> list[str]:
        folder = self.root / namespace
        if not folder.exists():
            return []
        return sorted(p.name[: -len(".parquet")] for p in folder.glob("*.parquet"))
