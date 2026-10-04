"""Candle type, validation and local cache.

Contract: a candle may be used for a decision only once ``closed_at <= clock.now()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import pandas as pd

from regimebot.clock import Clock

MAX_INTRADAY_GAP = timedelta(hours=6)  # larger jumps are session breaks, not missing bars


@dataclass(frozen=True)
class Candle:
    opened_at: datetime
    closed_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.opened_at.tzinfo is None or self.closed_at.tzinfo is None:
            raise ValueError("candle timestamps must be timezone-aware")
        if self.closed_at <= self.opened_at:
            raise ValueError("closed_at must be after opened_at")


class BarIssue(Enum):
    FUTURE = "future"
    STALE = "stale"
    BAD_VALUES = "bad_values"


def validate(
    candle: Candle, clock: Clock, bar: timedelta, max_lag_bars: int = 2
) -> BarIssue | None:
    now = clock.now()
    if candle.closed_at > now:
        return BarIssue.FUTURE
    if now - candle.closed_at > max_lag_bars * bar:
        return BarIssue.STALE
    prices = (candle.open, candle.high, candle.low, candle.close)
    if not all(math.isfinite(x) and x > 0 for x in prices):
        return BarIssue.BAD_VALUES
    if not (math.isfinite(candle.volume) and candle.volume >= 0):
        return BarIssue.BAD_VALUES
    if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close):
        return BarIssue.BAD_VALUES
    return None


def find_gaps(candles: list[Candle], bar: timedelta) -> list[int]:
    """Indices i where a bar is missing between candles[i-1] and candles[i] within a session."""
    out = []
    for i in range(1, len(candles)):
        d = candles[i].opened_at - candles[i - 1].opened_at
        if bar < d <= MAX_INTRADAY_GAP:
            out.append(i)
    return out


COLUMNS = ["opened_at", "closed_at", "open", "high", "low", "close", "volume"]


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame([[getattr(c, k) for k in COLUMNS] for c in candles], columns=COLUMNS)


def frame_to_candles(df: pd.DataFrame) -> list[Candle]:
    return [
        Candle(
            opened_at=r.opened_at.to_pydatetime(),
            closed_at=r.closed_at.to_pydatetime(),
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=float(r.volume),
        )
        for r in df.itertuples(index=False)
    ]


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_cache(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
