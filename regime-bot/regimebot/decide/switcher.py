"""Deterministic playbook switching with hysteresis. Models advise; this decides.

Rules (SPEC §5):
  1. a new label must exceed ``takeover`` filtered probability,
  2. for ``hold`` consecutive candles,
  3. and no switch happens within ``cooldown`` candles of the previous one;
  4. P(next in STRESS/CRASH) > ``early_cut_p`` multiplies size by ``early_cut``;
  5. top-two gap < ``gap`` means uncertain: size 0;
  6. model error or stale data: flat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

RISK_LABELS = ("STRESS", "CRASH")


@dataclass(frozen=True)
class SwitchParams:
    takeover: float = 0.70
    hold: int = 3
    cooldown: int = 6
    early_cut_p: float = 0.25
    early_cut: float = 0.5
    gap: float = 0.15


@dataclass
class SwitchState:
    active: str | None = None
    challenger: str | None = None
    count: int = 0
    cooldown_left: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SwitchState:
        return cls(**d)


@dataclass(frozen=True)
class SwitchDecision:
    active: str | None
    switched: bool
    uncertain: bool
    flat: bool
    size_mult: float
    reason: str
    probs: dict[str, float] = field(default_factory=dict)
    next_probs: dict[str, float] = field(default_factory=dict)


class Switcher:
    def __init__(self, params: SwitchParams, state: SwitchState | None = None) -> None:
        self.p = params
        self.state = state or SwitchState()

    def fail(self, kind: str) -> SwitchDecision:
        s = self.state
        s.challenger, s.count = None, 0
        return SwitchDecision(s.active, False, False, True, 0.0, f"flat: {kind}")

    def step(self, probs: dict[str, float], next_probs: dict[str, float]) -> SwitchDecision:
        s, p = self.state, self.p
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        (lead, p1), p2 = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
        reasons: list[str] = []

        if lead != s.active and p1 > p.takeover:
            s.count = s.count + 1 if s.challenger == lead else 1
            s.challenger = lead
        else:
            s.challenger, s.count = None, 0

        switched = False
        if s.challenger is not None and s.count >= p.hold and s.cooldown_left == 0:
            reasons.append(
                f"switch {s.active}->{lead}: P={p1:.3f}>{p.takeover} for {s.count} candles"
            )
            s.active, switched = lead, True
            s.challenger, s.count, s.cooldown_left = None, 0, p.cooldown
        elif s.cooldown_left > 0:
            s.cooldown_left -= 1

        uncertain = p1 - p2 < p.gap
        mult = 0.0 if s.active is None else 1.0
        if s.active is None:
            reasons.append("no active regime")
        if uncertain:
            mult = 0.0
            reasons.append(f"uncertain: top-2 gap {p1 - p2:.3f}<{p.gap}")
        p_risk = sum(next_probs.get(k, 0.0) for k in RISK_LABELS)
        if p_risk > p.early_cut_p:
            mult *= p.early_cut
            reasons.append(f"early_cut: P(next stress/crash)={p_risk:.3f}>{p.early_cut_p}")
        return SwitchDecision(
            active=s.active,
            switched=switched,
            uncertain=uncertain,
            flat=mult == 0.0,
            size_mult=mult,
            reason="; ".join(reasons) or "hold",
            probs=dict(probs),
            next_probs=dict(next_probs),
        )
