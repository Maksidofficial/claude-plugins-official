import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from regimebot.hmm.fit import (
    KScore,
    RegimeModel,
    choose_k,
    expected_durations,
    fit_regime_model,
)

TRUE_MEANS = np.array(
    [
        [0.5, -1.0, -0.8, -0.3, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.5, 2.0, 2.0, 1.5, -1.5],
    ]
)


def simulate(n: int = 3000, seed: int = 7) -> np.ndarray:
    m = GaussianHMM(n_components=3, covariance_type="diag", random_state=seed)
    m.startprob_ = np.array([0.6, 0.3, 0.1])
    m.transmat_ = np.array([[0.95, 0.04, 0.01], [0.04, 0.93, 0.03], [0.05, 0.10, 0.85]])
    m.means_ = TRUE_MEANS
    m.covars_ = np.full((3, 5), 0.3)
    X, _ = m.sample(n, random_state=seed)
    return np.asarray(X)


def test_expected_duration() -> None:
    A = np.array([[0.9, 0.1], [0.5, 0.5]])
    np.testing.assert_allclose(expected_durations(A), [10.0, 2.0])


def test_choose_k_prefers_simpler_on_close_scores() -> None:
    scores = [
        KScore(k=2, oos_ll=-998.0, bic=5000.0),
        KScore(k=3, oos_ll=-995.0, bic=4990.0),  # within 1% of k=4 on both
        KScore(k=4, oos_ll=-990.0, bic=4985.0),
    ]
    assert choose_k(scores, tol=0.01) == 2


def test_choose_k_takes_clear_winner() -> None:
    scores = [
        KScore(k=2, oos_ll=-1500.0, bic=7000.0),
        KScore(k=3, oos_ll=-1000.0, bic=5000.0),
        KScore(k=4, oos_ll=-998.0, bic=5200.0),
    ]
    assert choose_k(scores, tol=0.01) == 3


def test_choose_k_oos_dominates_when_bic_disagrees() -> None:
    scores = [
        KScore(k=2, oos_ll=-1500.0, bic=4000.0),  # best BIC, bad OOS
        KScore(k=3, oos_ll=-1000.0, bic=5000.0),
    ]
    assert choose_k(scores, tol=0.01) == 3


@pytest.fixture(scope="module")
def fitted() -> tuple[RegimeModel, object]:
    return fit_regime_model(simulate(), ks=(2, 3, 4), restarts=4, seed=1)


def test_recovers_three_states(fitted: tuple[RegimeModel, object]) -> None:
    model, report = fitted
    assert model.k == 3
    raw_means = model.means * model.scaler.std + model.scaler.mean
    for true in TRUE_MEANS:
        assert np.min(np.abs(raw_means - true).max(axis=1)) < 0.15


def test_model_is_valid(fitted: tuple[RegimeModel, object]) -> None:
    model, _ = fitted
    np.testing.assert_allclose(model.transmat.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.startprob.sum(), 1.0)
    assert model.covars.shape == (3, 5, 5)


def test_same_seed_same_model() -> None:
    X = simulate(1200)
    a, _ = fit_regime_model(X, ks=(2, 3), restarts=3, seed=11)
    b, _ = fit_regime_model(X, ks=(2, 3), restarts=3, seed=11)
    assert a.k == b.k
    # Multithreaded BLAS reorders float sums; identical up to round-off.
    for name in ("startprob", "transmat", "means", "covars"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=0, atol=1e-9)


def test_json_round_trip(fitted: tuple[RegimeModel, object]) -> None:
    model, _ = fitted
    again = RegimeModel.from_dict(model.to_dict())
    assert again.to_dict() == model.to_dict()
