import math

import numpy as np
import pandas as pd
import pytest

from regimebot.data.bars import frame_to_candles
from regimebot.data.features import (
    FEATURES,
    WARMUP,
    FeatureEngine,
    Standardizer,
    compute_features,
)


def test_feature_names() -> None:
    assert FEATURES == ("logret", "rvol", "range", "volratio", "trend")


def test_hand_computed_values(bars: pd.DataFrame) -> None:
    f = compute_features(bars)
    i = 120
    c, h, lo, v = (bars[k].to_numpy() for k in ("close", "high", "low", "volume"))
    lr = np.log(c[1:] / c[:-1])  # lr[j] is the return into bar j+1
    assert f["logret"].iloc[i] == pytest.approx(math.log(c[i] / c[i - 1]))
    assert f["rvol"].iloc[i] == pytest.approx(np.std(lr[i - 20 : i], ddof=1))
    assert f["range"].iloc[i] == pytest.approx((h[i] - lo[i]) / c[i - 1])
    assert f["volratio"].iloc[i] == pytest.approx(v[i] / v[i - 20 : i].mean())
    lp = np.log(c)
    a = 2 / 51
    ema = [lp[0]]
    for x in lp[1:]:
        ema.append(a * x + (1 - a) * ema[-1])
    assert f["trend"].iloc[i] == pytest.approx((ema[i] - ema[i - 1]) / f["rvol"].iloc[i])


def test_warmup_rows_are_nan(bars: pd.DataFrame) -> None:
    f = compute_features(bars)
    assert f.iloc[:WARMUP].isna().all(axis=None)
    assert f.iloc[WARMUP:].notna().all(axis=None)


def test_truncation_invariance(bars: pd.DataFrame) -> None:
    full = compute_features(bars)
    for k in (WARMUP, 150, 399):
        part = compute_features(bars.iloc[: k + 1])
        pd.testing.assert_frame_equal(part, full.iloc[: k + 1])


def test_incremental_matches_batch(bars: pd.DataFrame) -> None:
    batch = compute_features(bars).to_numpy()
    eng = FeatureEngine()
    for i, candle in enumerate(frame_to_candles(bars)):
        x = eng.update(candle)
        if i < WARMUP:
            assert x is None
        else:
            assert x is not None
            np.testing.assert_allclose(x, batch[i], rtol=1e-9, atol=1e-12)


def test_engine_state_round_trip(bars: pd.DataFrame) -> None:
    candles = frame_to_candles(bars)
    a = FeatureEngine()
    for cd in candles[:200]:
        a.update(cd)
    b = FeatureEngine.from_dict(a.to_dict())
    for cd in candles[200:]:
        xa, xb = a.update(cd), b.update(cd)
        assert xa is not None and xb is not None
        np.testing.assert_array_equal(xa, xb)


def test_standardizer_uses_only_fit_data(bars: pd.DataFrame) -> None:
    X = compute_features(bars).dropna().to_numpy()
    s = Standardizer.fit(X[:100])
    np.testing.assert_allclose(s.mean, X[:100].mean(axis=0))
    np.testing.assert_allclose(s.std, X[:100].std(axis=0, ddof=1))
    z = s.transform(X[150])
    np.testing.assert_allclose(z, (X[150] - s.mean) / s.std)
    s2 = Standardizer.from_dict(s.to_dict())
    np.testing.assert_array_equal(s2.transform(X[150]), z)


def test_standardizer_rejects_constant_column() -> None:
    X = np.ones((50, 5))
    with pytest.raises(ValueError):
        Standardizer.fit(X)
