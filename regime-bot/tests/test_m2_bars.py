from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from regimebot.clock import FixedClock
from regimebot.data.bars import (
    BarIssue,
    Candle,
    candles_to_frame,
    find_gaps,
    frame_to_candles,
    load_cache,
    save_cache,
    validate,
)

H = timedelta(hours=1)
T0 = datetime(2025, 1, 6, 14, 30, tzinfo=UTC)


def c(i: int, close: float = 100.0, **kw: float) -> Candle:
    base = dict(open=close, high=close + 1, low=close - 1, close=close, volume=1000.0)
    base.update(kw)
    return Candle(opened_at=T0 + i * H, closed_at=T0 + (i + 1) * H, **base)


def test_ok_candle() -> None:
    clock = FixedClock(T0 + 1 * H + timedelta(seconds=5))
    assert validate(c(0), clock, bar=H) is None


def test_future_candle_rejected() -> None:
    clock = FixedClock(T0 + timedelta(minutes=30))  # bar still open
    assert validate(c(0), clock, bar=H) is BarIssue.FUTURE


def test_stale_candle() -> None:
    clock = FixedClock(T0 + 1 * H + 2 * H + timedelta(seconds=1))
    assert validate(c(0), clock, bar=H) is BarIssue.STALE


@pytest.mark.parametrize(
    "kw",
    [
        {"close": float("nan")},
        {"volume": -1.0},
        {"low": 0.0},
        {"high": 50.0},  # high below close
    ],
)
def test_bad_values(kw: dict[str, float]) -> None:
    clock = FixedClock(T0 + H)
    bad = c(0)
    bad = Candle(**{**bad.__dict__, **kw})
    assert validate(bad, clock, bar=H) is BarIssue.BAD_VALUES


def test_naive_timestamps_rejected() -> None:
    with pytest.raises(ValueError):
        Candle(
            opened_at=datetime(2025, 1, 1),
            closed_at=datetime(2025, 1, 1, 1),
            open=1, high=1, low=1, close=1, volume=1,
        )


def test_find_gaps_intraday_only() -> None:
    bars = [c(0), c(1), c(3), c(4)]  # bar 2 missing inside the session
    overnight = Candle(**{**c(0).__dict__,
                          "opened_at": T0 + 20 * H, "closed_at": T0 + 21 * H})
    assert find_gaps(bars, bar=H) == [2]
    assert find_gaps([c(4), overnight], bar=H) == []  # session break is not a gap


def test_cache_round_trip(tmp_path: Path) -> None:
    bars = [c(i, close=100 + i) for i in range(5)]
    p = tmp_path / "spy.parquet"
    save_cache(candles_to_frame(bars), p)
    assert frame_to_candles(load_cache(p)) == bars
