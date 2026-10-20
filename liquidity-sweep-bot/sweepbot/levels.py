"""Liquidity pools: resting stop clusters above swing highs and below swing lows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import LevelConfig


@dataclass
class Pool:
    # "high": buy-side liquidity (short stops / short liquidations above a high)
    # "low":  sell-side liquidity (long stops / long liquidations below a low)
    side: str
    price: float
    created: int  # bar index of the (latest) pivot that formed it
    confirmed: int  # bar index at which it became known
    touches: int = 1


class LiquidityMap:
    """Incrementally tracks unswept swing-high and swing-low pools."""

    def __init__(self, cfg: LevelConfig):
        self.cfg = cfg
        self.highs: list[Pool] = []
        self.lows: list[Pool] = []
        self._buf: deque[tuple[int, float, float]] = deque(maxlen=cfg.pivot_left + cfg.pivot_right + 1)

    def update(self, i: int, high: float, low: float, atr: float | None) -> None:
        """Feed a closed bar. Call *after* sweep checks for the same bar."""
        self._buf.append((i, high, low))
        if len(self._buf) == self._buf.maxlen:
            mid = self.cfg.pivot_left
            idx, h, lo = self._buf[mid]
            left = list(self._buf)[:mid]
            right = list(self._buf)[mid + 1 :]
            if h > max(b[1] for b in left) and h >= max(b[1] for b in right):
                self._add("high", h, idx, i, atr)
            if lo < min(b[2] for b in left) and lo <= min(b[2] for b in right):
                self._add("low", lo, idx, i, atr)
        cutoff = i - self.cfg.max_age_bars
        self.highs = [p for p in self.highs if p.created >= cutoff]
        self.lows = [p for p in self.lows if p.created >= cutoff]

    def _add(self, side: str, price: float, created: int, confirmed: int, atr: float | None) -> None:
        pools = self.highs if side == "high" else self.lows
        tol = (atr or 0.0) * self.cfg.equal_tolerance_atr
        for p in pools:
            if abs(p.price - price) <= tol:
                p.touches += 1
                p.price = max(p.price, price) if side == "high" else min(p.price, price)
                p.created = created
                return
        pools.append(Pool(side, price, created, confirmed))

    def take_swept(self, high: float, low: float) -> tuple[list[Pool], list[Pool]]:
        """Remove and return every pool the bar traded through (highs, lows)."""
        swept_h = [p for p in self.highs if high > p.price]
        swept_l = [p for p in self.lows if low < p.price]
        if swept_h:
            self.highs = [p for p in self.highs if high <= p.price]
        if swept_l:
            self.lows = [p for p in self.lows if low >= p.price]
        return swept_h, swept_l

    def targets(self, side: str, entry: float) -> list[Pool]:
        """Opposing pools ordered nearest-first for a trade in `side` direction."""
        if side == "long":
            return sorted((p for p in self.highs if p.price > entry), key=lambda p: p.price)
        return sorted((p for p in self.lows if p.price < entry), key=lambda p: -p.price)
