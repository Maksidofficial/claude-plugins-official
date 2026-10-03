"""Forward filter: P(state_t | x_1..x_t). The only source of state probabilities for decisions.

There is deliberately no batch API here. Smoothed posteriors and Viterbi paths use future
observations, so they are banned from the package outside fitting (enforced by a test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from regimebot.hmm.fit import RegimeModel
from regimebot.types import Vec


class FilterError(RuntimeError):
    pass


@dataclass(frozen=True)
class FilterOutput:
    probs: Vec  # P(state_t | x_<=t)
    next_probs: Vec  # P(state_t+1 | x_<=t) = probs @ A
    loglik: float  # log p(x_t | x_<t), for live drift monitoring


class RegimeFilter:
    def __init__(self, model: RegimeModel) -> None:
        self.model = model
        self.d = model.means.shape[1]
        self._chol = []
        self._logdet = []
        for cov in model.covars:
            L = np.linalg.cholesky(cov)
            self._chol.append(L)
            self._logdet.append(2.0 * float(np.log(np.diag(L)).sum()))

    def _log_emission(self, z: Vec) -> Vec:
        out = np.empty(self.model.k)
        params = zip(self.model.means, self._chol, self._logdet, strict=True)
        for i, (mu, L, ld) in enumerate(params):
            y = solve_triangular(L, z - mu, lower=True)
            out[i] = -0.5 * (self.d * math.log(2 * math.pi) + ld + float(y @ y))
        return out

    def step(self, prev: Vec | None, z: Vec) -> FilterOutput:
        if z.shape != (self.d,) or not np.all(np.isfinite(z)):
            raise FilterError(f"bad observation: shape={z.shape}")
        prior = self.model.startprob if prev is None else prev @ self.model.transmat
        logb = self._log_emission(z)
        m = float(logb.max())
        w = prior * np.exp(logb - m)
        s = float(w.sum())
        if not (s > 0 and math.isfinite(s)):
            raise FilterError("filter normalizer is zero or non-finite")
        probs: Vec = w / s
        nxt: Vec = probs @ self.model.transmat
        return FilterOutput(probs=probs, next_probs=nxt, loglik=m + math.log(s))
