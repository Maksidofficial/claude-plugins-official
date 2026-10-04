"""Are the HMM's probabilities honest? Checked on observable outcomes only.

True states are never observed, so we score the model's predictive distribution of the next
bar's log return: a mixture of the per-state return distributions weighted by the
next-state probabilities. If the state probabilities are calibrated, the probability
integral transform (PIT) of realized returns is uniform. Reliability = max deviation of the
PIT's empirical CDF from the diagonal at deciles. Brier score on P(next return > 0) is
reported next to the base-rate (climatology) Brier.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from regimebot.types import Vec

LEVELS = np.linspace(0.1, 0.9, 9)


def pit(r: float, next_probs: Vec, mus: Vec, sigmas: Vec) -> float:
    return float(np.sum(next_probs * norm.cdf((r - mus) / sigmas)))


def brier(p: Vec, outcome: Vec) -> float:
    return float(np.mean((p - outcome) ** 2))


def reliability_max_dev(pits: Vec, levels: Vec = LEVELS) -> float:
    pits = np.asarray(pits)
    return float(max(abs(np.mean(pits <= q) - q) for q in levels))


def reliability_curve(pits: Vec, levels: Vec = LEVELS) -> list[tuple[float, float]]:
    pits = np.asarray(pits)
    return [(float(q), float(np.mean(pits <= q))) for q in levels]


@dataclass(frozen=True)
class CalibrationReport:
    per_label: dict[str, dict[str, float]]
    calibrated: dict[str, bool]
    curves: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


def calibration_report(
    rows: list[tuple[str, float, float, int]], min_n: int = 200, tol: float = 0.10
) -> CalibrationReport:
    """rows: (label active at forecast time, PIT, predicted P(up), realized up 0/1)."""
    groups: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for label, pit_v, p_up, up in rows:
        groups[label].append((pit_v, p_up, up))
    per, ok, curves = {}, {}, {}
    for label, g in groups.items():
        pits: Vec = np.array([x[0] for x in g], dtype=np.float64)
        probs: Vec = np.array([x[1] for x in g], dtype=np.float64)
        y: Vec = np.array([x[2] for x in g], dtype=np.float64)
        dev = reliability_max_dev(pits)
        per[label] = {
            "n": float(len(g)),
            "max_dev": dev,
            "brier": brier(probs, y),
            "brier_climatology": brier(np.full_like(y, y.mean()), y),
        }
        curves[label] = reliability_curve(pits)
        ok[label] = len(g) >= min_n and dev <= tol
    return CalibrationReport(per, ok, curves)
