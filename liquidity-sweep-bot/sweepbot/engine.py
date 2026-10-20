"""Glue between the strategy and a simulated broker: entries, flips, cooldowns
and the daily loss limit. Used by the backtester and by live paper trading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from .broker import PaperBroker, Trade, position_size
from .config import Config
from .strategy import Bar, Signal, SweepStrategy

log = logging.getLogger(__name__)


@dataclass
class StepResult:
    signal: Signal | None = None
    trades: list[Trade] = field(default_factory=list)
    opened: bool = False
    skipped: str = ""


class Engine:
    """Bar-by-bar trading loop.

    With ``fill_on_close=False`` (backtests) a signal is filled at the *next*
    bar's open. With ``fill_on_close=True`` (live paper) it is filled at the
    signal bar's close, which is the best price available at that moment.
    """

    def __init__(self, cfg: Config, equity: float = 10_000.0, fill_on_close: bool = False):
        self.cfg = cfg
        self.strategy = SweepStrategy(cfg)
        self.broker = PaperBroker(equity, cfg.risk)
        self.fill_on_close = fill_on_close
        self.index = -1
        self.pending: Signal | None = None
        self.cooldown_until = -1
        self._day = None
        self._day_start_equity = equity

    def _halted(self, price: float) -> bool:
        limit = self.cfg.risk.max_daily_loss
        return limit > 0 and self.broker.mark(price) <= self._day_start_equity * (1 - limit)

    def _execute(self, sig: Signal, price: float, bar: Bar, res: StepResult) -> None:
        pos = self.broker.position
        if pos is not None and pos.side != sig.side:
            res.trades.append(self.broker.close(price, bar.time, self.index, "flip"))
        if self.broker.position is not None:
            return
        if self._halted(price):
            res.skipped = "daily loss limit"
            return
        valid = sig.stop < price < sig.target if sig.side == "long" else sig.target < price < sig.stop
        if not valid:
            res.skipped = "price already beyond stop/target at fill"
            return
        qty = position_size(self.broker.equity, price, sig.stop, self.cfg.risk)
        if qty <= 0:
            res.skipped = "zero size"
            return
        self.broker.open(sig.side, qty, price, sig.stop, sig.target, bar.time, self.index)
        res.opened = True

    def step(self, bar: Bar) -> StepResult:
        self.index += 1
        res = StepResult()

        day = pd.Timestamp(bar.time).date() if bar.time is not None else None
        if day != self._day:
            self._day = day
            self._day_start_equity = self.broker.mark(bar.open)

        if self.pending is not None:
            sig, self.pending = self.pending, None
            self._execute(sig, bar.open, bar, res)

        trade = self.broker.on_bar(bar, self.index, self.cfg.trade.max_bars_in_trade)
        if trade:
            res.trades.append(trade)
            self.cooldown_until = self.index + self.cfg.trade.cooldown_bars

        sig = self.strategy.on_bar(bar)
        if sig is not None:
            res.signal = sig
            pos = self.broker.position
            if pos is None:
                actionable = self.index >= self.cooldown_until
                if not actionable:
                    res.skipped = "cooldown"
            else:
                actionable = pos.side != sig.side and self.cfg.trade.flip_on_opposite
                if not actionable:
                    res.skipped = "already in a position"
            if actionable:
                if self.fill_on_close:
                    self._execute(sig, bar.close, bar, res)
                else:
                    self.pending = sig
        return res

    def finish(self, bar: Bar) -> Trade | None:
        """Close any open position at the final bar's close."""
        if self.broker.position is None:
            return None
        return self.broker.close(bar.close, bar.time, self.index, "end of data")
