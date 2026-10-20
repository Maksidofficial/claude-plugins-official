"""Liquidation cascade detection.

Forced-order side convention (Binance ``forceOrder``): a SELL forced order is a
*long* being liquidated; a BUY forced order is a *short* being liquidated.

* Long cascade  -> price was flushed down  -> confirms a bullish (low) sweep.
* Short cascade -> price was squeezed up   -> confirms a bearish (high) sweep.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .config import CascadeConfig


@dataclass
class CascadeReading:
    long_cascade: bool
    short_cascade: bool
    source: str  # "liquidations" | "proxy" | "off" | "warmup"
    detail: str = ""


def _zscore(value: float, hist: deque[float]) -> float:
    n = len(hist)
    if n < 2:
        return 0.0
    mean = sum(hist) / n
    var = sum((x - mean) ** 2 for x in hist) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return math.inf if value > mean else 0.0
    return (value - mean) / std


class CascadeDetector:
    def __init__(self, cfg: CascadeConfig):
        if cfg.mode not in ("auto", "liquidations", "proxy", "off"):
            raise ValueError(f"cascade.mode must be auto|liquidations|proxy|off, got {cfg.mode!r}")
        self.cfg = cfg
        self._vol_hist: deque[float] = deque(maxlen=cfg.lookback)
        self._liq_long_hist: deque[float] = deque(maxlen=cfg.lookback)
        self._liq_short_hist: deque[float] = deque(maxlen=cfg.lookback)
        # Previous `window_bars - 1` per-bar values; the current bar is added on top.
        self._liq_long_win: deque[float] = deque(maxlen=max(0, cfg.window_bars - 1))
        self._liq_short_win: deque[float] = deque(maxlen=max(0, cfg.window_bars - 1))

    def update(
        self,
        volume: float,
        bar_range: float,
        atr: float | None,
        liq_long: float | None,
        liq_short: float | None,
    ) -> CascadeReading:
        have_liq = liq_long is not None and liq_short is not None
        win_l = sum(self._liq_long_win) + liq_long if have_liq else 0.0
        win_s = sum(self._liq_short_win) + liq_short if have_liq else 0.0
        reading = self._evaluate(volume, bar_range, atr, have_liq, win_l, win_s)

        # Update history after evaluating so the current bar is never compared with itself.
        self._vol_hist.append(volume)
        if have_liq:
            self._liq_long_hist.append(win_l)
            self._liq_short_hist.append(win_s)
            self._liq_long_win.append(liq_long)
            self._liq_short_win.append(liq_short)
        else:
            # Missing data breaks the rolling window; restart it.
            self._liq_long_win.clear()
            self._liq_short_win.clear()
        return reading

    def _evaluate(self, volume, bar_range, atr, have_liq, win_l, win_s) -> CascadeReading:
        cfg = self.cfg
        if cfg.mode == "off":
            return CascadeReading(True, True, "off")

        liq_ready = have_liq and len(self._liq_long_hist) >= cfg.min_history
        if cfg.mode == "liquidations" or (cfg.mode == "auto" and liq_ready):
            if not liq_ready:
                return CascadeReading(False, False, "warmup")
            zl = _zscore(win_l, self._liq_long_hist)
            zs = _zscore(win_s, self._liq_short_hist)
            long_c = zl >= cfg.liq_zscore and win_l >= cfg.min_liq_notional
            short_c = zs >= cfg.liq_zscore and win_s >= cfg.min_liq_notional
            return CascadeReading(
                long_c,
                short_c,
                "liquidations",
                f"long_liq={win_l:,.0f} (z={zl:.1f}) short_liq={win_s:,.0f} (z={zs:.1f})",
            )

        if len(self._vol_hist) < cfg.min_history or not atr:
            return CascadeReading(False, False, "warmup")
        zv = _zscore(volume, self._vol_hist)
        expansion = bar_range / atr
        hit = zv >= cfg.volume_zscore and expansion >= cfg.range_atr
        return CascadeReading(hit, hit, "proxy", f"volume z={zv:.1f} range={expansion:.1f}xATR")
