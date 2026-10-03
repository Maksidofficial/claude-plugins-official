from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

T0 = datetime(2023, 1, 3, 14, 30, tzinfo=UTC)
H = timedelta(hours=1)


def make_bars(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0002, 0.004, n)
    close = 400 * np.exp(np.cumsum(r))
    open_ = np.r_[400.0, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.uniform(5e5, 1.5e6, n)
    opened = [T0 + i * H for i in range(n)]
    return pd.DataFrame(
        {
            "opened_at": opened,
            "closed_at": [t + H for t in opened],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars(400)
