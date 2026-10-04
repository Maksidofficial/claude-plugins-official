"""Brokers. SimBroker mirrors what the live broker does with bracket orders, so a backtest
and paper trading can be compared line by line."""

from __future__ import annotations

from dataclasses import dataclass, field

from regimebot.backtest.costs import CostModel
from regimebot.data.bars import Candle
from regimebot.engine import OrderAction


@dataclass(frozen=True)
class Fill:
    ts: str
    qty: float
    price: float
    commission: float
    reason: str
    state: str | None = None


@dataclass
class SimBroker:
    """Market orders decided at close(t) fill at open(t+1) with slippage. Bracket stop and
    take-profit are checked intrabar; a gap through the stop fills at the open; if stop and
    take-profit are both touched in one bar the stop is assumed to fill first."""

    cash: float
    costs: CostModel
    qty: float = 0.0
    entry_price: float | None = None
    stop: float | None = None
    take_profit: float | None = None
    fills: list[Fill] = field(default_factory=list)
    _pending: list[tuple[OrderAction, str | None]] = field(default_factory=list)
    _entry_state: str | None = None

    def submit(self, orders: list[OrderAction], ts: str, state: str | None = None) -> None:
        self._pending.extend((o, state) for o in orders)

    def _fill(self, ts: str, qty: float, ref: float, reason: str, state: str | None) -> None:
        px = self.costs.fill_price(ref, qty)
        com = self.costs.commission(px, qty)
        if self.qty == 0 and qty > 0:
            self.entry_price, self._entry_state = px, state
        elif qty > 0 and self.entry_price is not None:
            self.entry_price = (self.entry_price * self.qty + px * qty) / (self.qty + qty)
        self.cash -= qty * px + com
        self.qty += qty
        self.fills.append(Fill(ts, qty, px, com, reason, self._entry_state))
        if self.qty == 0:
            self.entry_price = self.stop = self.take_profit = None
            self._entry_state = None

    def on_bar(self, c: Candle) -> None:
        ts = c.opened_at.isoformat()
        pending, self._pending = self._pending, []
        for o, state in pending:
            if o.qty < 0:
                o_qty = -min(-o.qty, self.qty)
                if o_qty == 0:
                    continue
            else:
                o_qty = o.qty
            self._fill(ts, o_qty, c.open, o.reason, state)
            if o.qty > 0 and o.stop_price is not None:
                self.stop, self.take_profit = o.stop_price, o.take_profit
        if self.qty <= 0:
            return
        if self.stop is not None and c.low <= self.stop:
            self._fill(ts, -self.qty, min(c.open, self.stop), "stop", None)
        elif self.take_profit is not None and c.high >= self.take_profit:
            self._fill(ts, -self.qty, max(c.open, self.take_profit), "take_profit", None)

    def equity(self, mark: float) -> float:
        return self.cash + self.qty * mark
