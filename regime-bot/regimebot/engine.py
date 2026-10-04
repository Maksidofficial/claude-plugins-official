"""One decision per closed candle. Live and backtest both call ``on_candle``; only the broker
behind them differs.

Order of authority: data checks -> kill switch / drawdown guard -> daily loss -> switcher
(hysteresis) -> playbook rules -> sizing -> risk veto chain. Models only supply probabilities.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np

from regimebot.clock import FixedClock
from regimebot.data.bars import BarIssue, Candle, validate
from regimebot.data.features import FeatureEngine
from regimebot.decide.playbook import FLAT, Playbook
from regimebot.decide.risk import Account, Order, RiskGate
from regimebot.decide.sizing import SizeInputs, target_fraction
from regimebot.decide.switcher import SwitchDecision, Switcher, SwitchParams, SwitchState
from regimebot.hmm.filter import FilterError, FilterOutput
from regimebot.hmm.fit import RegimeModel
from regimebot.limits import LIMITS
from regimebot.types import Vec

NY = ZoneInfo("America/New_York")
STRETCH_WINDOW = 20
TREND, RVOL = 4, 1  # feature columns
REDUCE_BELOW = 0.75  # only cut an open position when the target drops below 75% of it


class Filter(Protocol):
    def step(self, prev: Vec | None, z: Vec) -> FilterOutput: ...


@dataclass(frozen=True)
class Context:
    """Everything outside the engine's own state, supplied fresh for each candle."""

    model: RegimeModel
    filt: Filter
    playbooks: Mapping[str, Playbook]
    risk: RiskGate
    switch: SwitchParams
    kelly: Mapping[str, float]
    calibrated: Mapping[str, bool]  # per label; sizing is zero until a label passes
    frozen: bool
    equity: float
    position_qty: float  # from the broker, the source of truth
    entry_price: float | None
    approved: bool
    now: datetime
    bar: timedelta = timedelta(hours=1)


@dataclass
class EngineState:
    features: dict[str, Any]
    log_closes: list[float]
    alpha: list[float] | None
    switch: dict[str, Any]
    entry_state: str | None
    bars_held: int
    peak_equity: float | None
    day: str | None
    day_start_equity: float | None
    last_closed_at: str | None

    @classmethod
    def initial(cls) -> EngineState:
        return cls(FeatureEngine().to_dict(), [], None, SwitchState().to_dict(),
                   None, 0, None, None, None, None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> EngineState:
        return cls(**json.loads(s))


@dataclass(frozen=True)
class OrderAction:
    qty: float
    price: float
    stop_price: float | None
    take_profit: float | None
    reason: str


@dataclass(frozen=True)
class StepResult:
    ts: str
    decision: dict[str, Any]
    signals: dict[str, float]
    orders: list[OrderAction] = field(default_factory=list)
    vetoes: list[dict[str, str]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


def _by_label(p: Vec, labels: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for lab, v in zip(labels, p, strict=True):
        out[lab] = out.get(lab, 0.0) + float(v)
    return out


def stretch(log_closes: list[float]) -> float:
    if len(log_closes) < STRETCH_WINDOW:
        return math.nan
    a = np.array(log_closes[-STRETCH_WINDOW:])
    sd = float(a.std(ddof=1))
    return float((a[-1] - a.mean()) / sd) if sd > 0 else math.nan


def is_last_bar(c: Candle) -> bool:
    """True for the bar that spans the 16:00 New York close."""
    o, t = c.opened_at.astimezone(NY), c.closed_at.astimezone(NY)
    close = t.replace(hour=16, minute=0, second=0, microsecond=0)
    return o < close <= t


def _exit_reason(pb: Playbook, st: EngineState, dec: SwitchDecision, sig: dict[str, float],
                 close: float, entry: float | None) -> str | None:
    if st.entry_state != dec.active:
        return f"regime changed {st.entry_state}->{dec.active}"
    if dec.probs.get(pb.state, 0.0) < pb.invalidate_below_prob:
        return f"invalidated: P({pb.state})<{pb.invalidate_below_prob}"
    if st.bars_held >= pb.max_hold_bars:
        return f"max hold {pb.max_hold_bars} bars"
    if entry is not None:
        if close <= entry * (1 - pb.stop_pct):
            return "stop"
        if close >= entry * (1 + pb.take_profit_pct):
            return "take profit"
    if pb.style == "trend_following" and sig["trend"] < pb.exit_threshold:
        return "trend exit"
    if pb.style == "mean_reversion" and sig["stretch"] > pb.exit_threshold:
        return "reversion exit"
    if pb.style == "flat":
        return "flat playbook"
    return None


def entry_signal(pb: Playbook, sig: dict[str, float]) -> bool:
    if pb.style == "trend_following":
        return sig["trend"] > pb.entry_threshold
    if pb.style == "mean_reversion":
        return sig["stretch"] < -pb.entry_threshold
    return False


def on_candle(state: EngineState, candle: Candle, ctx: Context) -> tuple[EngineState, StepResult]:
    st = EngineState.from_json(state.to_json())  # never mutate the caller's state
    ts = candle.closed_at.isoformat()
    if st.last_closed_at and candle.closed_at <= datetime.fromisoformat(st.last_closed_at):
        return state, StepResult(ts, {"reason": "duplicate or out-of-order candle"}, {})

    issue = validate(candle, FixedClock(ctx.now), ctx.bar)
    if issue is BarIssue.FUTURE:
        return state, StepResult(ts, {"reason": "candle not closed yet"}, {})
    st.last_closed_at = ts

    # account bookkeeping
    st.peak_equity = max(st.peak_equity or ctx.equity, ctx.equity)
    day = candle.opened_at.astimezone(NY).date().isoformat()
    if day != st.day:
        st.day, st.day_start_equity = day, ctx.equity
    acct = Account(ctx.equity, st.peak_equity, st.day_start_equity or ctx.equity, ctx.position_qty)
    if ctx.position_qty > 0:
        st.bars_held += 1
    else:
        st.entry_state, st.bars_held = None, 0

    sw = Switcher(ctx.switch, SwitchState.from_dict(st.switch))
    labels = ctx.model.labels
    signals: dict[str, float] = {}
    state_next: list[float] = []
    loglik: float | None = None
    dec: SwitchDecision
    if issue is BarIssue.BAD_VALUES:
        dec = sw.fail("bad candle values")
    else:
        fe = FeatureEngine.from_dict(st.features)
        x = fe.update(candle)
        st.features = fe.to_dict()
        st.log_closes = (st.log_closes + [math.log(candle.close)])[-STRETCH_WINDOW:]
        out: FilterOutput | None = None
        err = ""
        if x is not None:
            try:
                z = ctx.model.scaler.transform(x)
                prev = None if st.alpha is None else np.array(st.alpha)
                out = ctx.filt.step(prev, z)
                st.alpha = out.probs.tolist()
                state_next = out.next_probs.tolist()
                loglik = out.loglik
            except (FilterError, ValueError) as e:
                st.alpha, err = None, f"model error: {e}"
            signals = {"trend": float(x[TREND]), "stretch": stretch(st.log_closes)}
        if err:
            dec = sw.fail(err)
        elif out is None:
            dec = sw.fail("warmup")
        elif issue is BarIssue.STALE:
            dec = sw.fail("stale data")
        else:
            dec = sw.step(_by_label(out.probs, labels), _by_label(out.next_probs, labels))
    st.switch = sw.state.to_dict()

    # target position as a fraction of equity
    reasons = [dec.reason]
    killed = ctx.risk.guard(acct)
    day_loss = 1 - acct.equity / acct.day_start_equity
    pos_frac = ctx.position_qty * candle.close / ctx.equity if ctx.equity > 0 else 0.0
    pb = ctx.playbooks.get(dec.active or "") or FLAT["CRASH"]
    stop = tp = None
    if killed:
        target = 0.0
        reasons.append(f"kill switch: {ctx.risk.kill.reason()}")
    elif day_loss >= LIMITS.daily_loss_pct:
        target = 0.0
        reasons.append(f"daily loss {day_loss:.2%}")
    elif dec.flat or not signals:
        target = 0.0
    else:
        sized = target_fraction(SizeInputs(
            active=dec.active, probs=dec.probs, kelly=ctx.kelly.get(dec.active or "", 0.0),
            playbook_max=pb.max_size, size_mult=dec.size_mult,
            calibrated=ctx.calibrated.get(dec.active or "", False),
        ))
        if ctx.position_qty > 0:
            why = _exit_reason(pb, st, dec, signals, candle.close, ctx.entry_price)
            if why:
                target = 0.0
                reasons.append(f"exit: {why}")
            elif sized < pos_frac * REDUCE_BELOW:
                target = sized
                reasons.append(f"reduce to {sized:.3f}")
            else:
                target = pos_frac
        elif entry_signal(pb, signals) and sized > 0:
            target = sized
            stop = candle.close * (1 - pb.stop_pct)
            tp = candle.close * (1 + pb.take_profit_pct)
            reasons.append(f"entry {pb.style} {sized:.3f}")
        else:
            target = 0.0
    last_bar = is_last_bar(candle)
    if last_bar and target > LIMITS.max_overnight_pct:
        target = LIMITS.max_overnight_pct
        reasons.append("overnight cap")

    desired = math.floor(target * ctx.equity / candle.close) if target > 0 else 0.0
    if 0 < pos_frac and target == pos_frac:
        desired = ctx.position_qty
    delta = float(desired - ctx.position_qty)

    orders: list[OrderAction] = []
    vetoes: list[dict[str, str]] = []
    approvals: list[dict[str, Any]] = []
    if delta != 0:
        veto = ctx.risk.check(Order(delta, candle.close), acct, dec.active,
                              approved=ctx.approved, frozen=ctx.frozen, last_bar=last_bar)
        if veto is None:
            orders.append(OrderAction(delta, candle.close, stop, tp, reasons[-1]))
            if ctx.position_qty == 0 and delta > 0:
                st.entry_state, st.bars_held = dec.active, 0
        else:
            vetoes.append({"rule": veto.rule, "detail": veto.detail})
            if veto.rule == "needs_approval":
                approvals.append({"qty": delta, "price": candle.close, "state": dec.active})

    decision = {
        "active": dec.active,
        "switched": dec.switched,
        "uncertain": dec.uncertain,
        "flat": dec.flat,
        "size_mult": dec.size_mult,
        "probs": dec.probs,
        "next_probs": dec.next_probs,
        "target": target,
        "issue": issue.value if issue else None,
        "state_next": state_next,
        "loglik": loglik,
        "reason": "; ".join(r for r in reasons if r),
    }
    return st, StepResult(ts, decision, signals, orders, vetoes, approvals)
