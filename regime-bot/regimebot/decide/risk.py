"""Pre-trade veto chain and kill switch. Every order passes here; no model can bypass it.

Orders that only shrink a position always pass, so exits and flattening are never blocked.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from regimebot.limits import LIMITS


@dataclass(frozen=True)
class Account:
    equity: float
    peak_equity: float
    day_start_equity: float
    position_qty: float  # shares, long-only: >= 0


@dataclass(frozen=True)
class Order:
    qty: float  # signed shares: + buy, - sell
    price: float  # reference price for notional checks


@dataclass(frozen=True)
class Veto:
    rule: str
    detail: str


class KillSwitch:
    """A file on disk, so it survives restarts. Only an explicit human reset clears it."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def active(self) -> bool:
        return self.path.exists()

    def reason(self) -> str:
        try:
            return str(json.loads(self.path.read_text())["reason"])
        except (OSError, ValueError, KeyError):
            return "killed (reason unreadable)" if self.active() else ""

    def fire(self, reason: str) -> None:
        if self.active():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"reason": reason, "at": datetime.now(UTC).isoformat()}))
        os.replace(tmp, self.path)

    def reset(self, confirm: str) -> None:
        if not confirm.strip():
            raise ValueError("kill switch reset needs a written confirmation")
        self.path.unlink(missing_ok=True)


def is_reducing(position: float, qty: float) -> bool:
    new = position + qty
    return new * position >= 0 and abs(new) < abs(position)


class RiskGate:
    def __init__(self, kill: KillSwitch) -> None:
        self.kill = kill

    def guard(self, acct: Account) -> bool:
        """Run every candle, order or not. Returns True when the kill switch is active."""
        dd = 1 - acct.equity / acct.peak_equity
        if dd >= LIMITS.max_drawdown_pct:
            self.kill.fire(f"max_drawdown {dd:.2%} >= {LIMITS.max_drawdown_pct:.0%}")
        return self.kill.active()

    def check(
        self,
        order: Order,
        acct: Account,
        state: str | None,
        *,
        approved: bool = False,
        frozen: bool = False,
        last_bar: bool = False,
    ) -> Veto | None:
        if not (math.isfinite(order.price) and order.price > 0 and math.isfinite(order.qty)):
            return Veto("bad_order", f"qty={order.qty} price={order.price}")
        if order.qty == 0:
            return Veto("bad_order", "zero quantity")
        if is_reducing(acct.position_qty, order.qty):
            return None

        if self.kill.active():
            return Veto("kill_switch", self.kill.reason())
        dd = 1 - acct.equity / acct.peak_equity
        if dd >= LIMITS.max_drawdown_pct:
            self.kill.fire(f"max_drawdown {dd:.2%} >= {LIMITS.max_drawdown_pct:.0%}")
            return Veto("kill_switch", self.kill.reason())
        day = 1 - acct.equity / acct.day_start_equity
        if day >= LIMITS.daily_loss_pct:
            return Veto("daily_loss", f"{day:.2%} >= {LIMITS.daily_loss_pct:.0%}")
        if frozen:
            return Veto("frozen", "new entries frozen pending human review")

        new_qty = acct.position_qty + order.qty
        if new_qty < 0:
            return Veto("long_only", f"resulting position {new_qty}")
        frac = new_qty * order.price / acct.equity
        lev = LIMITS.leverage_for(state or "")
        if frac > lev:
            return Veto("state_leverage", f"{frac:.2%} > {lev:.0%} in {state}")
        if frac > LIMITS.max_position_pct:
            return Veto("max_position", f"{frac:.2%} > {LIMITS.max_position_pct:.0%}")
        if last_bar and frac > LIMITS.max_overnight_pct:
            return Veto("overnight", f"{frac:.2%} > {LIMITS.max_overnight_pct:.0%}")
        notional = abs(order.qty) * order.price
        if notional > LIMITS.manual_approval_usd and not approved:
            return Veto("needs_approval", f"${notional:,.0f} > ${LIMITS.manual_approval_usd:,.0f}")
        return None
