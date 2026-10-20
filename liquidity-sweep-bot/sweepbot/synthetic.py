"""Synthetic OHLCV with planted range sweeps, for demos and tests (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sweep_series(n_ranges: int = 12, bars_per_range: int = 120, seed: int = 7, start_price: float = 30_000.0) -> pd.DataFrame:
    """Each segment oscillates inside a range, sweeps one side with a long wick,
    a volume spike and a burst of liquidations, then trends to the other side.
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = start_price
    for _ in range(n_ranges):
        width = price * rng.uniform(0.02, 0.04)
        lo, hi = price - width / 2, price + width / 2
        sweep_low = rng.random() < 0.5
        swing_bars = bars_per_range - 30
        # Oscillate inside the range so clear swing highs/lows form.
        phase = rng.uniform(0, 2 * np.pi)
        for k in range(swing_bars):
            mid = (lo + hi) / 2 + (width * 0.42) * np.sin(phase + k * 2 * np.pi / 24)
            rows.append(_bar(rng, price, mid, width * 0.04, base_vol=100))
            price = rows[-1]["close"]
        # The sweep bar: wick through the extreme, close back inside, heavy volume + liquidations.
        depth = width * rng.uniform(0.06, 0.12)
        if sweep_low:
            low = lo - depth
            close = lo + width * 0.05
            rows.append(dict(open=price, high=max(price, close) + width * 0.01, low=low, close=close,
                             volume=100 * rng.uniform(6, 9), liq_long=5e6 * rng.uniform(1, 2), liq_short=1e4))
        else:
            high = hi + depth
            close = hi - width * 0.05
            rows.append(dict(open=price, high=high, low=min(price, close) - width * 0.01, close=close,
                             volume=100 * rng.uniform(6, 9), liq_long=1e4, liq_short=5e6 * rng.uniform(1, 2)))
        price = close
        # Expansion to the opposite side of the range and a little beyond.
        target = (hi + width * 0.15) if sweep_low else (lo - width * 0.15)
        for k in range(29):
            mid = price + (target - price) * 0.15
            rows.append(_bar(rng, price, mid, width * 0.03, base_vol=140))
            price = rows[-1]["close"]
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, index=idx)


def _bar(rng, open_: float, close_target: float, noise: float, base_vol: float) -> dict:
    close = close_target + rng.normal(0, noise * 0.3)
    high = max(open_, close) + abs(rng.normal(0, noise * 0.5))
    low = min(open_, close) - abs(rng.normal(0, noise * 0.5))
    return dict(
        open=open_, high=high, low=low, close=close,
        volume=base_vol * rng.uniform(0.7, 1.3),
        liq_long=abs(rng.normal(2e4, 1e4)), liq_short=abs(rng.normal(2e4, 1e4)),
    )
