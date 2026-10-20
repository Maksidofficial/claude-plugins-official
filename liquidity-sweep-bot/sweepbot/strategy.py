"""Sweep-to-sweep reversal strategy.

The idea: stops and liquidation levels cluster beyond obvious swing highs and
lows. When price runs one of those pools, forced orders fire in a cascade and,
once they are exhausted, price often snaps back and travels to the liquidity
resting on the *other* side of the range. The bot therefore:

1. maps unswept swing-high / swing-low pools (``levels.LiquidityMap``);
2. waits for a bar that wicks through a pool and closes back inside it;
3. requires a liquidation cascade on the side that just got flushed
   (``cascade.CascadeDetector``);
4. enters the opposite way, stops beyond the sweep extreme, and targets the
   nearest opposing pool that pays at least ``min_rr``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cascade import CascadeDetector, CascadeReading
from .config import Config
from .levels import LiquidityMap, Pool


@dataclass
class Bar:
    time: Any
    open: float
    high: float
    low: float
    close: float
    volume: float
    liq_long: float | None = None  # notional of longs liquidated (forced SELLs)
    liq_short: float | None = None  # notional of shorts liquidated (forced BUYs)


@dataclass
class Signal:
    time: Any
    index: int
    side: str  # "long" | "short"
    ref_price: float  # signal bar close
    stop: float
    target: float
    target_is_pool: bool
    swept_level: float
    liquidity_touches: int
    penetration_atr: float
    cascade: CascadeReading

    @property
    def rr(self) -> float:
        risk = abs(self.ref_price - self.stop)
        return abs(self.target - self.ref_price) / risk if risk else 0.0

    def describe(self) -> str:
        tgt = "pool" if self.target_is_pool else "fallback"
        return (
            f"{self.side.upper()} @ {self.ref_price:.2f} | swept {self.swept_level:.2f} "
            f"({self.liquidity_touches} touch, {self.penetration_atr:.2f} ATR deep) | "
            f"SL {self.stop:.2f} TP {self.target:.2f} ({tgt}, {self.rr:.2f}R) | "
            f"cascade[{self.cascade.source}] {self.cascade.detail}"
        )


class ATR:
    """Wilder's ATR, updated one bar at a time."""

    def __init__(self, period: int):
        self.period = period
        self.value: float | None = None
        self._prev_close: float | None = None
        self._seed: list[float] = []

    def update(self, high: float, low: float, close: float) -> float | None:
        tr = high - low if self._prev_close is None else max(
            high - low, abs(high - self._prev_close), abs(low - self._prev_close)
        )
        self._prev_close = close
        if self.value is None:
            self._seed.append(tr)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
        else:
            self.value = (self.value * (self.period - 1) + tr) / self.period
        return self.value


class SweepStrategy:
    """Stateful, bar-by-bar signal engine shared by the backtester and live loop."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.atr = ATR(cfg.atr_period)
        self.levels = LiquidityMap(cfg.levels)
        self.cascade = CascadeDetector(cfg.cascade)
        self.index = -1

    def on_bar(self, bar: Bar) -> Signal | None:
        """Process a *closed* bar; return a signal to act on at the next bar."""
        self.index += 1
        atr = self.atr.value  # ATR as of the previous close: no lookahead into this bar
        reading = self.cascade.update(bar.volume, bar.high - bar.low, atr, bar.liq_long, bar.liq_short)

        swept_highs, swept_lows = self.levels.take_swept(bar.high, bar.low)
        signal = None
        if atr:
            candidates = []
            if self.cfg.trade.allow_short and reading.short_cascade:
                candidates.append(self._evaluate("short", bar, swept_highs, atr, reading))
            if self.cfg.trade.allow_long and reading.long_cascade:
                candidates.append(self._evaluate("long", bar, swept_lows, atr, reading))
            candidates = [c for c in candidates if c is not None]
            # An outside bar sweeping both sides: trust the side it closed away from.
            if len(candidates) == 2:
                mid = (bar.high + bar.low) / 2
                candidates = [c for c in candidates if (c.side == "long") == (bar.close > mid)]
            signal = candidates[0] if candidates else None

        self.atr.update(bar.high, bar.low, bar.close)
        self.levels.update(self.index, bar.high, bar.low, self.atr.value)
        return signal

    def _evaluate(self, side: str, bar: Bar, swept: list[Pool], atr: float, reading: CascadeReading) -> Signal | None:
        if not swept:
            return None
        sw, tc = self.cfg.sweep, self.cfg.trade
        if side == "short":
            level = max(p.price for p in swept)
            extreme = bar.high
            penetration = bar.high - level
            closed_back = bar.close < level
        else:
            level = min(p.price for p in swept)
            extreme = bar.low
            penetration = level - bar.low
            closed_back = bar.close > level
        touches = sum(p.touches for p in swept)

        if touches < self.cfg.levels.min_touches:
            return None
        if sw.require_close_back and not closed_back:
            return None
        if not (sw.min_penetration_atr * atr <= penetration <= sw.max_penetration_atr * atr):
            return None

        entry = bar.close
        buffer = tc.stop_buffer_atr * atr
        stop = extreme + buffer if side == "short" else extreme - buffer
        risk = abs(entry - stop)
        if risk <= 0:
            return None

        target, from_pool = None, False
        for pool in self.levels.targets(side, entry):
            if abs(pool.price - entry) / risk >= tc.min_rr:
                target, from_pool = pool.price, True
                break
        if target is None:
            if tc.fallback_rr <= 0:
                return None
            target = entry - tc.fallback_rr * risk if side == "short" else entry + tc.fallback_rr * risk

        return Signal(
            time=bar.time,
            index=self.index,
            side=side,
            ref_price=entry,
            stop=stop,
            target=target,
            target_is_pool=from_pool,
            swept_level=level,
            liquidity_touches=touches,
            penetration_atr=penetration / atr,
            cascade=reading,
        )
