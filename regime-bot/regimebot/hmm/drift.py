"""Detect when a refit or live data no longer matches the model we trade on.

Any alarm freezes new entries and alerts a human. Nothing here unfreezes automatically.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from regimebot.hmm.fit import RegimeModel
from regimebot.hmm.labels import auto_label, match_states, raw_covars, raw_means


@dataclass(frozen=True)
class DriftThresholds:
    mean_shift_sigma: float = 2.0  # Mahalanobis shift of a matched state's mean, old covariance
    transmat_l1: float = 0.2  # max row L1 distance between aligned transition matrices


@dataclass(frozen=True)
class DriftReport:
    max_mean_shift_sigma: float
    transmat_l1: float
    alarms: list[str] = field(default_factory=list)


def compare_models(old: RegimeModel, new: RegimeModel, th: DriftThresholds) -> DriftReport:
    alarms: list[str] = []
    if old.k != new.k:
        alarms.append("k_changed")
    mapping = match_states(old, new)

    mo, mn, co = raw_means(old), raw_means(new), raw_covars(old)
    shift = 0.0
    for n_i, o_i in mapping.items():
        d = mn[n_i] - mo[o_i]
        shift = max(shift, float(np.sqrt(d @ np.linalg.solve(co[o_i], d))))
    if shift > th.mean_shift_sigma:
        alarms.append("mean_shift")

    l1 = float("inf")
    if old.k == new.k:
        perm = [mapping[i] for i in range(new.k)]  # new i -> old perm[i]
        aligned = np.empty_like(old.transmat)
        for i in range(new.k):
            for j in range(new.k):
                aligned[perm[i], perm[j]] = new.transmat[i, j]
        l1 = float(np.abs(aligned - old.transmat).sum(axis=1).max())
    if l1 > th.transmat_l1:
        alarms.append("transmat")

    if new.labels and new.labels != auto_label(new):
        alarms.append("label_conflict")
    return DriftReport(shift, l1, alarms)


class LiveLLMonitor:
    """Alarm when the rolling mean of live per-bar log-likelihood falls below the
    ``pct`` percentile of the same rolling statistic in-sample."""

    def __init__(self, threshold: float, window: int) -> None:
        self.threshold = threshold
        self.recent: deque[float] = deque(maxlen=window)

    @classmethod
    def from_in_sample(cls, ll: np.ndarray, window: int = 35, pct: float = 5.0) -> LiveLLMonitor:
        roll = np.convolve(ll, np.ones(window) / window, mode="valid")
        return cls(float(np.percentile(roll, pct)), window)

    def add(self, ll: float) -> None:
        self.recent.append(ll)

    def alarm(self) -> bool:
        full = len(self.recent) == self.recent.maxlen
        return full and float(np.mean(self.recent)) < self.threshold
