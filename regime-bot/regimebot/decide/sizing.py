"""Confidence-scaled sizing (SPEC §6). Output is a fraction of equity, long-only notional.

size = min(hard cap, state leverage cap, playbook max, ¼·Kelly)
       × P(active) × (1 − H/H_max) × switcher multiplier
and zero in CRASH, with no active regime, when uncertain, or before calibration passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from regimebot.limits import LIMITS

KELLY_FRACTION = 0.25


@dataclass(frozen=True)
class SizeInputs:
    active: str | None
    probs: dict[str, float]  # filtered probability per label
    kelly: float  # full-Kelly fraction estimated for the active playbook
    playbook_max: float
    size_mult: float  # from the switcher: 1, early-cut 0.5, or 0
    calibrated: bool


def entropy_factor(probs: dict[str, float]) -> float:
    n = len(probs)
    if n <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return max(0.0, 1.0 - h / math.log(n))


def target_fraction(x: SizeInputs) -> float:
    if x.active is None or x.active == "CRASH" or not x.calibrated:
        return 0.0
    if not (math.isfinite(x.kelly) and x.kelly > 0) or x.size_mult <= 0:
        return 0.0
    cap = min(
        LIMITS.max_position_pct,
        LIMITS.leverage_for(x.active),
        max(0.0, x.playbook_max),
        KELLY_FRACTION * x.kelly,
    )
    p_active = x.probs.get(x.active, 0.0)
    return max(0.0, cap * p_active * entropy_factor(x.probs) * x.size_mult)
