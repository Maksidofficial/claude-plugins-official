"""Configuration dataclasses, loadable from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class LevelConfig:
    # A swing high/low is a pivot with `pivot_left` lower highs before it and
    # `pivot_right` lower highs after it. It only becomes a known liquidity
    # pool once the right-hand bars have closed (no lookahead).
    pivot_left: int = 5
    pivot_right: int = 5
    # Pivots within this many ATRs of an existing pool merge into it
    # ("equal highs / equal lows" = stacked stops = more liquidity).
    equal_tolerance_atr: float = 0.15
    max_age_bars: int = 500
    # Minimum number of merged touches before a pool is tradable.
    min_touches: int = 2


@dataclass
class SweepConfig:
    # The wick must run at least this far through the pool...
    min_penetration_atr: float = 0.05
    # ...but not so far that it looks like a genuine breakout.
    max_penetration_atr: float = 2.0
    # Require the candle to close back on the original side of the level.
    require_close_back: bool = True


@dataclass
class CascadeConfig:
    # "liquidations": require real forced-order data
    # "proxy":        use volume spike + range expansion instead
    # "auto":         liquidations when enough history exists, else proxy
    # "off":          sweeps alone are enough
    mode: str = "auto"
    lookback: int = 96
    min_history: int = 30
    # Liquidations are summed over the sweep bar plus the previous
    # `window_bars - 1` bars, then compared with their rolling distribution.
    window_bars: int = 2
    liq_zscore: float = 3.0
    min_liq_notional: float = 0.0
    volume_zscore: float = 2.0
    range_atr: float = 1.0


@dataclass
class TradeConfig:
    min_rr: float = 2.5
    # Used when no opposing pool offers `min_rr`. Set to 0 to skip the trade.
    fallback_rr: float = 2.5
    stop_buffer_atr: float = 0.75
    max_bars_in_trade: int = 48
    cooldown_bars: int = 3
    # Close and reverse when an opposite sweep + cascade fires mid-trade.
    flip_on_opposite: bool = True
    allow_long: bool = True
    allow_short: bool = True


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_leverage: float = 5.0
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0
    # Fraction of start-of-day equity; once lost, no new entries that UTC day.
    max_daily_loss: float = 0.03


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    atr_period: int = 14
    levels: LevelConfig = field(default_factory=LevelConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    defaults = cls()
    kwargs = {}
    for name, value in data.items():
        sub = getattr(defaults, name)
        if is_dataclass(sub) and isinstance(value, dict):
            kwargs[name] = _build(type(sub), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def timeframe_ms(tf: str) -> int:
    try:
        return TIMEFRAME_MS[tf]
    except KeyError:
        raise ValueError(f"Unsupported timeframe {tf!r}; use one of {list(TIMEFRAME_MS)}") from None
