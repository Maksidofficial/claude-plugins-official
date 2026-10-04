"""Deterministic, past-only features.

Row t uses candles <= t only. ``compute_features`` (batch, used for fitting) and
``FeatureEngine`` (incremental, used by live and backtest) must agree; a test enforces it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from regimebot.data.bars import Candle
from regimebot.types import Vec

FEATURES = ("ret10", "rvol", "range5", "volratio5", "trend")
RET_BARS = 10  # multi-bar return: one-bar returns are noise the HMM would split states on
SMOOTH = 5  # bars averaged for range and volume
VOL_WINDOW = 20
EMA_SPAN = 50
EMA_ALPHA = 2 / (EMA_SPAN + 1)
WARMUP = 50  # rows before this are NaN / None



def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    c = bars["close"].astype(float)
    logret = np.log(c / c.shift(1))
    ret = np.log(c / c.shift(RET_BARS))
    rvol = logret.rolling(VOL_WINDOW).std(ddof=1)
    rng = ((bars["high"] - bars["low"]) / c.shift(1)).rolling(SMOOTH).mean()
    v = bars["volume"].astype(float)
    volratio = v.rolling(SMOOTH).mean() / v.shift(SMOOTH).rolling(VOL_WINDOW).mean()
    ema = np.log(c).ewm(alpha=EMA_ALPHA, adjust=False).mean()
    trend = (ema - ema.shift(1)) / rvol
    out = pd.DataFrame(
        {"ret10": ret, "rvol": rvol, "range5": rng, "volratio5": volratio, "trend": trend}
    )
    out.iloc[:WARMUP] = np.nan
    return out


class FeatureEngine:
    """Incremental twin of ``compute_features``. Serializable so restarts resume exactly."""

    def __init__(self) -> None:
        self.n = 0
        self.ema = math.nan
        self.closes: deque[float] = deque(maxlen=RET_BARS + 1)
        self.rets: deque[float] = deque(maxlen=VOL_WINDOW)
        self.ranges: deque[float] = deque(maxlen=SMOOTH)
        self.vols: deque[float] = deque(maxlen=SMOOTH + VOL_WINDOW)

    def update(self, c: Candle) -> Vec | None:
        i, self.n = self.n, self.n + 1
        lp = math.log(c.close)
        prev = self.closes[-1] if self.closes else math.nan
        self.closes.append(c.close)
        self.vols.append(c.volume)
        if i == 0:
            self.ema = lp
            return None
        self.rets.append(math.log(c.close / prev))
        self.ranges.append((c.high - c.low) / prev)
        ema_prev = self.ema
        self.ema = EMA_ALPHA * lp + (1 - EMA_ALPHA) * self.ema
        if i < WARMUP:
            return None
        ret = math.log(c.close / self.closes[0])
        rvol = float(np.std(np.fromiter(self.rets, float), ddof=1))
        rng = sum(self.ranges) / SMOOTH
        vols = list(self.vols)
        volratio = (sum(vols[-SMOOTH:]) / SMOOTH) / (sum(vols[:VOL_WINDOW]) / VOL_WINDOW)
        trend = (self.ema - ema_prev) / rvol if rvol > 0 else math.nan
        return np.array([ret, rvol, rng, volratio, trend], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "ema": self.ema,
            "closes": list(self.closes),
            "rets": list(self.rets),
            "ranges": list(self.ranges),
            "vols": list(self.vols),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureEngine:
        e = cls()
        e.n, e.ema = int(d["n"]), float(d["ema"])
        e.closes.extend(d["closes"])
        e.rets.extend(d["rets"])
        e.ranges.extend(d["ranges"])
        e.vols.extend(d["vols"])
        return e


@dataclass(frozen=True)
class Standardizer:
    """Fitted on the training window only and stored with the model it was fitted for."""

    mean: Vec
    std: Vec

    @classmethod
    def fit(cls, X: Vec) -> Standardizer:
        std = X.std(axis=0, ddof=1)
        if not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("feature with zero or undefined variance")
        return cls(mean=X.mean(axis=0), std=std)

    def transform(self, X: Vec) -> Vec:
        out: Vec = (X - self.mean) / self.std
        return out

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> Standardizer:
        return cls(mean=np.array(d["mean"]), std=np.array(d["std"]))
