import ast
from pathlib import Path

import numpy as np
import pytest
from conftest import make_bars
from hmmlearn.hmm import GaussianHMM
from hypothesis import given, settings
from hypothesis import strategies as st

from regimebot.data.bars import frame_to_candles
from regimebot.data.features import FeatureEngine, Standardizer, compute_features
from regimebot.hmm.filter import FilterError, RegimeFilter
from regimebot.hmm.fit import RegimeModel

PKG = Path(__file__).resolve().parents[1] / "regimebot"


def toy_model() -> RegimeModel:
    rng = np.random.default_rng(3)
    k, d = 3, 5
    A = np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.05, 0.15, 0.8]])
    means = rng.normal(0, 1, (k, d))
    covars = np.stack([np.eye(d) * s for s in (0.5, 1.0, 2.0)])
    sc = Standardizer(mean=np.zeros(d), std=np.ones(d))
    return RegimeModel(np.array([0.5, 0.3, 0.2]), A, means, covars, sc, 0.0)


def as_hmmlearn(m: RegimeModel) -> GaussianHMM:
    h = GaussianHMM(n_components=m.k, covariance_type="full")
    h.startprob_, h.transmat_, h.means_, h.covars_ = m.startprob, m.transmat, m.means, m.covars
    h.n_features = m.means.shape[1]
    return h


def run(f: RegimeFilter, Z: np.ndarray) -> list:
    out, alpha = [], None
    for z in Z:
        o = f.step(alpha, z)
        alpha = o.probs
        out.append(o)
    return out


def test_total_loglik_matches_hmmlearn() -> None:
    m = toy_model()
    Z = np.random.default_rng(0).normal(0, 1.2, (300, 5))
    outs = run(RegimeFilter(m), Z)
    assert sum(o.loglik for o in outs) == pytest.approx(as_hmmlearn(m).score(Z), abs=1e-8)


def test_filtered_equals_last_posterior_of_prefix() -> None:
    m = toy_model()
    Z = np.random.default_rng(1).normal(0, 1.2, (120, 5))
    outs = run(RegimeFilter(m), Z)
    h = as_hmmlearn(m)
    for t in (0, 10, 57, 119):
        np.testing.assert_allclose(outs[t].probs, h.predict_proba(Z[: t + 1])[-1], atol=1e-10)


def test_next_state_is_alpha_times_A() -> None:
    m = toy_model()
    o = RegimeFilter(m).step(None, np.zeros(5))
    np.testing.assert_allclose(o.next_probs, o.probs @ m.transmat)
    assert o.probs.sum() == pytest.approx(1.0)
    assert o.next_probs.sum() == pytest.approx(1.0)


def test_bad_input_raises() -> None:
    f = RegimeFilter(toy_model())
    with pytest.raises(FilterError):
        f.step(None, np.array([np.nan, 0, 0, 0, 0]))
    with pytest.raises(FilterError):
        f.step(None, np.zeros(4))


def test_extreme_input_does_not_underflow() -> None:
    o = RegimeFilter(toy_model()).step(None, np.full(5, 40.0))
    assert np.all(np.isfinite(o.probs)) and o.probs.sum() == pytest.approx(1.0)


BARS = make_bars(260, seed=5)
CANDLES = frame_to_candles(BARS)
RAW = compute_features(BARS).dropna().to_numpy()
FITTED = toy_model()
MODEL = RegimeModel(
    FITTED.startprob, FITTED.transmat, FITTED.means, FITTED.covars, Standardizer.fit(RAW), 0.0
)


def pipeline(candles: list) -> list[tuple]:
    fe, filt, alpha, out = FeatureEngine(), RegimeFilter(MODEL), None, []
    for c in candles:
        x = fe.update(c)
        if x is None:
            out.append(None)
            continue
        o = filt.step(alpha, MODEL.scaler.transform(x))
        alpha = o.probs
        out.append((o.probs.tobytes(), o.next_probs.tobytes()))
    return out


BASELINE = pipeline(CANDLES)


@settings(max_examples=40, deadline=None)
@given(k=st.integers(min_value=0, max_value=len(CANDLES) - 2), seed=st.integers(0, 10_000))
def test_no_decision_touches_a_future_candle(k: int, seed: int) -> None:
    noise = make_bars(len(CANDLES), seed=seed)
    noisy = noise.copy()
    shift = CANDLES[k + 1].opened_at - noise["opened_at"].iloc[k + 1]
    for col in ("opened_at", "closed_at"):
        noisy[col] = noise[col] + shift
    future = frame_to_candles(noisy.iloc[k + 1 :])
    tampered = CANDLES[: k + 1] + future
    got = pipeline(tampered)
    assert got[: k + 1] == BASELINE[: k + 1]
    if k + 1 >= 50:  # past warmup the tampering must be visible, or this test proves nothing
        assert got[k + 1 :] != BASELINE[k + 1 :]


BANNED_CALLS = {"predict", "predict_proba", "decode", "score_samples"}


def test_smoothing_and_viterbi_are_banned_outside_fit() -> None:
    for py in PKG.rglob("*.py"):
        rel = py.relative_to(PKG).as_posix()
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in BANNED_CALLS, f"{rel}: .{node.func.attr}()"
            if rel != "hmm/fit.py":
                if isinstance(node, ast.Import):
                    assert all(not a.name.startswith("hmmlearn") for a in node.names), rel
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("hmmlearn"), rel


def test_unreachable_state_does_not_break_filter() -> None:
    m = toy_model()
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    zero = RegimeModel(np.array([1.0, 0.0, 0.0]), A, m.means, m.covars, m.scaler, 0.0)
    f = RegimeFilter(zero)
    o = f.step(None, m.means[0])
    far = m.means[2] + 30 * (m.means[2] - m.means[0])  # only state 2 explains this at all
    o = f.step(o.probs, far)
    assert np.all(np.isfinite(o.probs)) and o.probs.sum() == pytest.approx(1.0)


def test_fit_floors_transition_probabilities() -> None:
    from regimebot.hmm.fit import TRANSMAT_FLOOR, floor_transmat

    A = floor_transmat(np.array([[1.0, 0.0], [0.3, 0.7]]))
    assert A.min() >= TRANSMAT_FLOOR * 0.99
    np.testing.assert_allclose(A.sum(axis=1), 1.0)
