"""Live paper trading: public candles in, the shared engine decides, the paper broker fills.

One ``step(now)`` per wake-up: fetch candles closed since the last one, run each through
``engine.on_candle`` exactly as the backtest does, journal it, and persist all state
atomically. Candles that are already stale when they arrive (first start, catch-up after
an outage) warm the features but are never traded: the engine's stale check sees to that.
"""

from __future__ import annotations

import json
import logging
import time as _time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from regimebot.backtest.costs import CostModel
from regimebot.clock import Clock, SystemClock
from regimebot.data.bars import frame_to_candles
from regimebot.decide.playbook import load_playbooks
from regimebot.decide.risk import KillSwitch, RiskGate
from regimebot.decide.switcher import SwitchParams
from regimebot.engine import Context, EngineState, OrderAction, on_candle
from regimebot.exec.approvals import ApprovalQueue
from regimebot.exec.paper import PaperBroker
from regimebot.hmm.filter import RegimeFilter
from regimebot.hmm.fit import RegimeModel

log = logging.getLogger("regimebot.live")

Feed = Callable[[datetime, datetime], pd.DataFrame]


@dataclass(frozen=True)
class LiveConfig:
    state_dir: Path
    model_path: Path
    playbooks_dir: Path
    symbol: str
    bar: timedelta
    start: datetime  # first candle to ingest on a fresh start (warm-up history)
    costs: CostModel
    qty_step: float
    session_close: time | None
    tz: str
    switch: SwitchParams
    initial_cash: float
    calibrated: Mapping[str, bool]
    kelly: Mapping[str, float]
    auto_approve: bool = False  # tests only; production approvals come from a human
    max_lag_bars: int = 2
    frozen: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class LiveRunner:
    def __init__(self, cfg: LiveConfig, feed: Feed, alert: Callable[[str], None] | None = None):
        self.cfg, self.feed = cfg, feed
        self.alert = alert or (lambda msg: log.warning(msg))
        d = cfg.state_dir
        d.mkdir(parents=True, exist_ok=True)
        self.journal_path = d / "journal.jsonl"
        self.model = RegimeModel.from_dict(json.loads(cfg.model_path.read_text()))
        self.filt = RegimeFilter(self.model)
        self.books, errors = load_playbooks(cfg.playbooks_dir)
        for s, e in errors.items():
            self.alert(f"playbook {s} rejected, that state trades flat: {e}")
        self.risk = RiskGate(KillSwitch(d / "KILLED"))
        self.approvals = ApprovalQueue(d / "approvals.json")
        bp, ep = d / "broker.json", d / "engine.json"
        self.broker = (PaperBroker.load(bp) if bp.exists()
                       else PaperBroker(cash=cfg.initial_cash, costs=cfg.costs))
        self.state = EngineState.from_json(ep.read_text()) if ep.exists() else EngineState.initial()
        self._reconcile()

    # -- persistence -------------------------------------------------------------------
    def _journal(self, rec: dict[str, Any]) -> None:
        with self.journal_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    def _save(self) -> None:
        self.broker.save(self.cfg.state_dir / "broker.json")
        p = self.cfg.state_dir / "engine.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(self.state.to_json())
        tmp.replace(p)

    def _last_position(self) -> float | None:
        if not self.journal_path.exists():
            return None
        last = None
        with self.journal_path.open() as f:
            for line in f:
                rec = json.loads(line)
                if "position_after" in rec:
                    last = float(rec["position_after"])
        return last

    def _reconcile(self) -> None:
        expected = self._last_position()
        if expected is not None and abs(expected - self.broker.qty) > 1e-9:
            reason = f"reconcile mismatch: journal {expected} vs broker {self.broker.qty}"
            self.risk.kill.fire(reason)
            self._journal({"kind": "reconcile", "reason": reason})
            self.alert(f"KILL SWITCH: {reason}")

    # -- loop --------------------------------------------------------------------------
    def step(self, now: datetime) -> None:
        last = (datetime.fromisoformat(self.state.last_closed_at)
                if self.state.last_closed_at else None)
        since = last or self.cfg.start
        try:
            df = self.feed(since, now)
        except Exception as e:  # network, parse: never fatal, staleness handles it
            self._journal({"kind": "fetch_error", "at": now.isoformat(), "error": str(e)[:300]})
            df = None
        if df is not None and len(df):
            for c in frame_to_candles(df):
                if last is None or c.closed_at > last:
                    self._on_candle(c, now)
                    last = c.closed_at
        self._check_stale(now, last)

    def _on_candle(self, c: Any, now: datetime) -> None:
        self.broker.on_bar(c)
        active = self.state.switch.get("active")
        approved = self.cfg.auto_approve or self.approvals.granted(active, now)
        ctx = Context(
            model=self.model, filt=self.filt, playbooks=self.books, risk=self.risk,
            switch=self.cfg.switch, kelly=self.cfg.kelly, calibrated=self.cfg.calibrated,
            frozen=self.cfg.frozen, equity=self.broker.equity(c.close),
            position_qty=self.broker.qty, entry_price=self.broker.entry_price,
            approved=approved, now=now, bar=self.cfg.bar, qty_step=self.cfg.qty_step,
            session_close=self.cfg.session_close, tz=ZoneInfo(self.cfg.tz),
        )
        self.state, res = on_candle(self.state, c, ctx)
        self.broker.submit(res.orders, res.ts, res.decision.get("active"))
        for a in res.approvals:
            rid = self.approvals.request(a["qty"], a["price"], a["state"], now)
            self.alert(f"approval needed {rid}: {a['qty']:+.5f} @ {a['price']:.2f} ({a['state']})")
        if res.decision.get("switched"):
            self.alert(f"state switch -> {res.decision.get('active')}: {res.decision['reason']}")
        self._journal({
            "kind": "candle", "ts": res.ts, "result": json.loads(res.to_json()),
            "position_after": self.broker.qty, "equity": self.broker.equity(c.close),
        })
        self._save()

    def _check_stale(self, now: datetime, last: datetime | None) -> None:
        if last is None or now - last <= self.cfg.max_lag_bars * self.cfg.bar:
            return
        if self.broker.qty <= 0:
            return
        if any(o.qty < 0 for o, _ in self.broker._pending):
            return  # flatten already queued
        o = OrderAction(-self.broker.qty, self.broker.entry_price or 0.0, None, None,
                        "stale data: flatten at next available price")
        self.broker.submit([o], now.isoformat(), None)
        self._journal({"kind": "stale", "at": now.isoformat(), "last_candle": last.isoformat(),
                       "orders": [{"qty": o.qty, "reason": o.reason}],
                       "position_after": self.broker.qty})
        self.alert(f"stale data since {last.isoformat()}: flattening {self.broker.qty}")
        self._save()


def run_forever(runner: LiveRunner, clock: Clock | None = None, grace_s: int = 15) -> None:
    """Wake shortly after each bar closes. systemd restarts the process if it dies."""
    clock = clock or SystemClock()
    bar_s = int(runner.cfg.bar.total_seconds())
    while True:
        now = clock.now()
        runner.step(now)
        heartbeat = runner.cfg.state_dir / "heartbeat"
        heartbeat.write_text(datetime.now(UTC).isoformat())
        wait = bar_s - (int(now.timestamp()) % bar_s) + grace_s
        _time.sleep(min(wait, 60))  # re-check at least every minute for staleness
