"""Request budgeting for free data APIs.

``RateLimiter`` enforces sliding-window request limits (for Tiingo's free
tier: 50 per hour and 1,000 per day) and persists request timestamps to a
JSON file so separate runs share one budget. ``SymbolBudget`` tracks unique
symbols requested per calendar month (Tiingo free tier: about 500).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path


class RateLimitExceeded(RuntimeError):
    """Raised when a request would exceed a limit and waiting is not allowed."""

    def __init__(self, message: str, retry_after_seconds: float):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class RateLimiter:
    """Sliding-window limiter over one or more (max_requests, window_seconds) pairs."""

    def __init__(
        self,
        limits: list[tuple[int, float]],
        state_path: str | Path | None = None,
        max_wait_seconds: float = 120.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not limits:
            raise ValueError("At least one limit is required")
        self.limits = sorted(limits, key=lambda x: x[1])
        self.state_path = Path(state_path) if state_path else None
        self.max_wait_seconds = max_wait_seconds
        self._clock = clock
        self._sleep = sleep
        self._timestamps: list[float] = self._load()

    def _load(self) -> list[float]:
        if self.state_path and self.state_path.exists():
            try:
                return sorted(float(t) for t in json.loads(self.state_path.read_text()))
            except (ValueError, TypeError):
                return []
        return []

    def _save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._timestamps))

    def _prune(self, now: float) -> None:
        longest = self.limits[-1][1]
        self._timestamps = [t for t in self._timestamps if now - t < longest]

    def wait_time(self) -> float:
        """Seconds until one more request is allowed under every limit."""
        now = self._clock()
        self._prune(now)
        wait = 0.0
        for max_requests, window in self.limits:
            recent = [t for t in self._timestamps if now - t < window]
            if len(recent) >= max_requests:
                # The oldest request that must expire before we are under the limit.
                oldest_needed = recent[len(recent) - max_requests]
                wait = max(wait, oldest_needed + window - now)
        return wait

    def acquire(self) -> None:
        """Record one request, waiting (up to max_wait_seconds) if needed."""
        wait = self.wait_time()
        if wait > 0:
            if wait > self.max_wait_seconds:
                raise RateLimitExceeded(
                    f"Request budget exhausted; next request allowed in {wait:,.0f} seconds",
                    retry_after_seconds=wait,
                )
            self._sleep(wait)
        self._timestamps.append(self._clock())
        self._save()

    @property
    def used(self) -> int:
        self._prune(self._clock())
        return len(self._timestamps)


class SymbolBudget:
    """Tracks unique symbols requested per calendar month."""

    def __init__(
        self,
        max_per_month: int,
        state_path: str | Path | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.max_per_month = max_per_month
        self.state_path = Path(state_path) if state_path else None
        self._clock = clock
        self._state: dict[str, list[str]] = {}
        if self.state_path and self.state_path.exists():
            try:
                self._state = json.loads(self.state_path.read_text())
            except ValueError:
                self._state = {}

    def _month(self) -> str:
        return time.strftime("%Y-%m", time.gmtime(self._clock()))

    def check_and_add(self, symbol: str) -> None:
        month = self._month()
        seen = set(self._state.get(month, []))
        if symbol in seen:
            return
        if len(seen) >= self.max_per_month:
            raise RateLimitExceeded(
                f"Monthly unique-symbol budget of {self.max_per_month} reached for {month}",
                retry_after_seconds=float("inf"),
            )
        seen.add(symbol)
        self._state = {month: sorted(seen)}  # older months are irrelevant
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state))

    def count(self) -> int:
        return len(self._state.get(self._month(), []))
