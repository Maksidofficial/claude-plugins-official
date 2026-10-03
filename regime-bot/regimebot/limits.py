"""HARD LIMITS. Code constants only.

Nothing here is loaded from config, env or a model. Changing a value means a reviewed
code commit. Every order passes these checks in decide/risk.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

_STATE_LEVERAGE: Final = MappingProxyType(
    {"CALM_UP": 1.0, "CHOP": 0.5, "STRESS": 0.25, "CRASH": 0.0}
)


@dataclass(frozen=True)
class HardLimits:
    max_position_pct: float = 0.20  # notional / equity
    daily_loss_pct: float = 0.02  # flat + no entries for the rest of the session
    max_drawdown_pct: float = 0.10  # from peak equity -> kill switch
    manual_approval_usd: float = 5_000.0  # orders above this need a human
    max_overnight_pct: float = 0.10  # notional / equity held through the close
    state_leverage: MappingProxyType[str, float] = field(default_factory=lambda: _STATE_LEVERAGE)

    def leverage_for(self, state: str) -> float:
        return self.state_leverage.get(state, 0.0)


LIMITS: Final = HardLimits()
