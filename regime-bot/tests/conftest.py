import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

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


@pytest.fixture(scope="module")
def model_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small model fitted on synthetic regimes, saved like a production artifact."""
    from regime_data import regime_bars
    from regimebot.data.features import compute_features
    from regimebot.hmm.fit import fit_regime_model
    from regimebot.hmm.labels import auto_label

    bars, _ = regime_bars(1400, seed=4)
    raw = compute_features(bars.iloc[:900]).dropna().to_numpy()
    m, _ = fit_regime_model(raw, ks=(4,), restarts=2, seed=0)
    m = m.with_labels(auto_label(m))
    p = tmp_path_factory.mktemp("models") / "current.json"
    p.write_text(json.dumps(m.to_dict()))
    return p
