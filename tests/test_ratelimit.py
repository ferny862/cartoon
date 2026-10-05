import pytest

from src.data.ratelimit import RateLimiter, RateLimitExceeded, SymbolBudget


class FakeClock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_hourly_limit_waits_when_short():
    clock = FakeClock()
    rl = RateLimiter([(3, 60)], clock=clock, sleep=clock.sleep, max_wait_seconds=120)
    for _ in range(3):
        rl.acquire()
    assert clock.slept == []
    rl.acquire()  # must wait for the first request to leave the 60s window
    assert clock.slept == [pytest.approx(60)]


def test_free_tier_limits_raise_when_wait_too_long():
    clock = FakeClock()
    rl = RateLimiter([(50, 3600), (1000, 86400)], clock=clock, sleep=clock.sleep, max_wait_seconds=120)
    for _ in range(50):
        rl.acquire()
    with pytest.raises(RateLimitExceeded) as exc:
        rl.acquire()
    assert exc.value.retry_after_seconds == pytest.approx(3600)


def test_daily_limit_applies_across_hours():
    clock = FakeClock()
    rl = RateLimiter([(50, 3600), (1000, 86400)], clock=clock, sleep=clock.sleep, max_wait_seconds=1e9)
    for _ in range(1000):
        rl.acquire()
    assert rl.wait_time() > 3600  # the daily window, not the hourly one, binds


def test_state_persists_between_instances(tmp_path):
    clock = FakeClock()
    path = tmp_path / "state.json"
    rl = RateLimiter([(2, 3600)], state_path=path, clock=clock, sleep=clock.sleep, max_wait_seconds=0)
    rl.acquire()
    rl.acquire()
    rl2 = RateLimiter([(2, 3600)], state_path=path, clock=clock, sleep=clock.sleep, max_wait_seconds=0)
    with pytest.raises(RateLimitExceeded):
        rl2.acquire()


def test_symbol_budget(tmp_path):
    clock = FakeClock()
    b = SymbolBudget(2, state_path=tmp_path / "s.json", clock=clock)
    b.check_and_add("SPY")
    b.check_and_add("SPY")  # repeat symbols are free
    b.check_and_add("EFA")
    with pytest.raises(RateLimitExceeded):
        b.check_and_add("AGG")
    assert SymbolBudget(2, state_path=tmp_path / "s.json", clock=clock).count() == 2
    clock.t += 40 * 86400  # next month resets
    b.check_and_add("AGG")
