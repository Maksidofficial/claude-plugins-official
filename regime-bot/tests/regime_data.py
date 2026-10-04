"""Synthetic market with known regimes, for engine and backtest tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

REGIMES = {  # mean log return per bar, vol per bar, volume scale
    "CALM_UP": (0.0006, 0.003, 1.0),
    "CHOP": (0.0, 0.006, 1.2),
    "STRESS": (-0.0006, 0.012, 1.8),
    "CRASH": (-0.004, 0.03, 3.0),
}
NAMES = list(REGIMES)
A = np.array(
    [
        [0.985, 0.010, 0.004, 0.001],
        [0.012, 0.980, 0.007, 0.001],
        [0.010, 0.020, 0.960, 0.010],
        [0.000, 0.020, 0.080, 0.900],
    ]
)


def regime_bars(n: int, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    s, states = 0, []
    for _ in range(n):
        states.append(NAMES[s])
        s = int(rng.choice(4, p=A[s]))
    r = np.array([rng.normal(REGIMES[k][0], REGIMES[k][1]) for k in states])
    close = 400 * np.exp(np.cumsum(r))
    open_ = np.r_[400.0, close[:-1]]
    wig = np.array([REGIMES[k][1] for k in states]) * rng.uniform(0.2, 0.8, n)
    high = np.maximum(open_, close) * (1 + wig)
    low = np.minimum(open_, close) * (1 - wig)
    vol = np.array([REGIMES[k][2] for k in states]) * rng.uniform(8e5, 1.2e6, n)
    t0 = datetime(2022, 1, 3, 14, 30, tzinfo=UTC)
    opened = [t0 + timedelta(hours=i) for i in range(n)]
    df = pd.DataFrame(
        {
            "opened_at": opened,
            "closed_at": [t + timedelta(hours=1) for t in opened],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )
    return df, states
