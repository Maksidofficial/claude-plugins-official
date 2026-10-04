from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Applied on every fill, each side: price moves against us by slippage, plus commission."""

    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    def fill_price(self, ref: float, qty: float) -> float:
        slip = self.slippage_bps / 1e4
        return ref * (1 + slip) if qty > 0 else ref * (1 - slip)

    def commission(self, price: float, qty: float) -> float:
        return abs(qty) * price * self.commission_bps / 1e4
