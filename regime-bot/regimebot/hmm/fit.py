"""Fit the Gaussian HMM and choose K. Fitting only: decisions never call hmmlearn.

Selection: fit each K on the first part of the data, score the held-out tail.
  A = Ks within ``tol`` of the best out-of-sample log-likelihood
  B = Ks within ``tol`` of the best (lowest) BIC
  K* = smallest K in A∩B, else smallest K in A.
Close scores therefore resolve to the simpler model, and OOS evidence always wins.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from hmmlearn.hmm import GaussianHMM

from regimebot.data.features import Standardizer
from regimebot.types import Vec


@dataclass(frozen=True)
class KScore:
    k: int
    oos_ll: float
    bic: float


@dataclass(frozen=True)
class SelectionReport:
    scores: list[KScore]
    chosen_k: int
    n_train: int
    n_holdout: int


@dataclass(frozen=True)
class RegimeModel:
    """Parameters in standardized feature space plus the scaler that defines that space."""

    startprob: Vec
    transmat: Vec
    means: Vec
    covars: Vec  # (k, d, d)
    scaler: Standardizer
    loglik: float
    labels: list[str] = field(default_factory=list)

    @property
    def k(self) -> int:
        return int(self.transmat.shape[0])

    def with_labels(self, labels: list[str]) -> RegimeModel:
        return RegimeModel(
            self.startprob, self.transmat, self.means, self.covars, self.scaler, self.loglik, labels
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "startprob": self.startprob.tolist(),
            "transmat": self.transmat.tolist(),
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
            "scaler": self.scaler.to_dict(),
            "loglik": self.loglik,
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegimeModel:
        return cls(
            startprob=np.array(d["startprob"]),
            transmat=np.array(d["transmat"]),
            means=np.array(d["means"]),
            covars=np.array(d["covars"]),
            scaler=Standardizer.from_dict(d["scaler"]),
            loglik=float(d["loglik"]),
            labels=list(d.get("labels", [])),
        )


TRANSMAT_FLOOR = 1e-6  # no regime change is ever "impossible" for the filter


def floor_transmat(A: Vec) -> Vec:
    out: Vec = np.maximum(A, TRANSMAT_FLOOR)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def expected_durations(transmat: Vec) -> Vec:
    out: Vec = 1.0 / (1.0 - np.diag(transmat))
    return out


def _fit_best(Z: Vec, k: int, restarts: int, seed: int) -> GaussianHMM:
    best: GaussianHMM | None = None
    best_ll = -np.inf
    for r in range(restarts):
        m = GaussianHMM(
            n_components=k, covariance_type="full", n_iter=300, tol=1e-5, random_state=seed + r
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Z)
            ll = float(m.score(Z))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError(f"no restart converged for k={k}")
    return best


def choose_k(scores: list[KScore], tol: float) -> int:
    best_ll = max(s.oos_ll for s in scores)
    best_bic = min(s.bic for s in scores)
    a = {s.k for s in scores if best_ll - s.oos_ll <= tol * abs(best_ll)}
    b = {s.k for s in scores if s.bic - best_bic <= tol * abs(best_bic)}
    return min(a & b) if a & b else min(a)


def fit_regime_model(
    raw: Vec,
    ks: tuple[int, ...] = (2, 3, 4, 5),
    restarts: int = 50,
    seed: int = 42,
    holdout_frac: float = 0.2,
    tol: float = 0.01,
) -> tuple[RegimeModel, SelectionReport]:
    """``raw``: unstandardized feature rows, oldest first, all strictly before the fit time."""
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw features contain NaN/inf")
    n_train = int(len(raw) * (1 - holdout_frac))
    train, hold = raw[:n_train], raw[n_train:]
    sc_train = Standardizer.fit(train)
    Zt, Zh = sc_train.transform(train), sc_train.transform(hold)

    scores = []
    for k in ks:
        m = _fit_best(Zt, k, restarts, seed)
        scores.append(KScore(k=k, oos_ll=float(m.score(Zh)), bic=float(m.bic(Zt))))
    k_star = choose_k(scores, tol)

    scaler = Standardizer.fit(raw)
    Z = scaler.transform(raw)
    final = _fit_best(Z, k_star, restarts, seed)
    model = RegimeModel(
        startprob=np.asarray(final.startprob_, dtype=np.float64),
        transmat=floor_transmat(np.asarray(final.transmat_, dtype=np.float64)),
        means=np.asarray(final.means_, dtype=np.float64),
        covars=np.asarray(final.covars_, dtype=np.float64),
        scaler=scaler,
        loglik=float(final.score(Z)),
    )
    return model, SelectionReport(scores, k_star, n_train, len(hold))


WARM_START_TOL = 0.01  # a random restart replaces the warm start only if >1% better LL


def _warm_fit(Z: Vec, init: RegimeModel, scaler: Standardizer, seed: int) -> GaussianHMM | None:
    """Start EM from the current model, re-expressed in the new scaler's units."""
    raw_mu = init.means * init.scaler.std + init.scaler.mean
    raw_cov = init.covars * np.outer(init.scaler.std, init.scaler.std)
    m = GaussianHMM(
        n_components=init.k, covariance_type="full", n_iter=300, tol=1e-5,
        random_state=seed, init_params="",
    )
    m.startprob_ = init.startprob
    m.transmat_ = init.transmat
    m.means_ = (raw_mu - scaler.mean) / scaler.std
    m.covars_ = raw_cov / np.outer(scaler.std, scaler.std)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Z)
        return m if np.isfinite(m.score(Z)) else None
    except (ValueError, np.linalg.LinAlgError):
        return None


def fit_k(
    raw: Vec, k: int, restarts: int, seed: int, init: RegimeModel | None = None
) -> RegimeModel:
    """Refit with K fixed. Used by walk-forward refits; K itself changes only via review.

    With ``init``, EM starts from the current model so states keep their identity across
    refits; random restarts can only replace it with a clearly better fit.
    """
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw features contain NaN/inf")
    scaler = Standardizer.fit(raw)
    Z = scaler.transform(raw)
    m = _fit_best(Z, k, restarts, seed)
    if init is not None and init.k == k:
        warm = _warm_fit(Z, init, scaler, seed)
        if warm is not None:
            ll_w, ll_r = float(warm.score(Z)), float(m.score(Z))
            if ll_r - ll_w <= WARM_START_TOL * abs(ll_w):
                m = warm
    return RegimeModel(
        startprob=np.asarray(m.startprob_, dtype=np.float64),
        transmat=floor_transmat(np.asarray(m.transmat_, dtype=np.float64)),
        means=np.asarray(m.means_, dtype=np.float64),
        covars=np.asarray(m.covars_, dtype=np.float64),
        scaler=scaler,
        loglik=float(m.score(Z)),
    )
