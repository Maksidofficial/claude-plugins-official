"""Walk-forward backtest. Same engine as live; SimBroker in place of the exchange.

Timeline per candle t (all past-only):
  1. pending orders from t-1 fill at open(t); bracket stops checked inside bar t
  2. last forecast is scored against the realized return (calibration evidence)
  3. once per day: per-label Kelly and calibration are recomputed from OOS history so far
  4. engine decides at close(t) with the current model
  5. refit due? fit on rows closed <= t; the new model is used from t+1, if no drift alarm

Simulated humans: order approvals are assumed granted; a drift freeze lifts when a refit is
clean, or when two consecutive refits agree on the drifted model (hmm.drift.simulated_review).
Live, only a human does either.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

from regimebot.backtest.baselines import buy_and_hold, static_playbook
from regimebot.backtest.calibration import CalibrationReport, calibration_report, pit
from regimebot.backtest.costs import CostModel
from regimebot.backtest.metrics import Metrics, Trade, summarize, trades_from_fills
from regimebot.data.bars import frame_to_candles
from regimebot.data.features import compute_features
from regimebot.decide.playbook import load_playbooks
from regimebot.decide.risk import KillSwitch, RiskGate
from regimebot.decide.switcher import SwitchParams
from regimebot.engine import Context, EngineState, StepResult, on_candle
from regimebot.exec.broker import Fill, SimBroker
from regimebot.hmm.drift import DriftThresholds, LiveLLMonitor, compare_models, simulated_review
from regimebot.hmm.filter import RegimeFilter
from regimebot.hmm.fit import RegimeModel, SelectionReport, fit_k, fit_regime_model
from regimebot.hmm.labels import auto_label, relabel_refit, state_stats

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class WFConfig:
    oos_start: datetime
    state_dir: Path
    refit_days: int = 30
    ks: tuple[int, ...] = (2, 3, 4, 5)
    restarts_initial: int = 50
    restarts_refit: int = 10
    seed: int = 42
    costs: CostModel = field(default_factory=CostModel)
    switch: SwitchParams = field(default_factory=SwitchParams)
    drift: DriftThresholds = field(default_factory=DriftThresholds)
    kelly_prior: float = 0.4
    kelly_min_trades: int = 20
    calib_min_n: int = 200
    calib_tol: float = 0.10
    initial_cash: float = 100_000.0
    warmup_bars: int = 200
    bar: timedelta = timedelta(hours=1)
    ll_window: int = 35


@dataclass
class WFResult:
    equity: pd.Series
    fills: list[Fill]
    trades: list[Trade]
    records: list[StepResult]
    swaps: list[dict[str, Any]]
    calibration: CalibrationReport
    metrics: Metrics
    per_state: dict[str, dict[str, float]]
    baselines: dict[str, Metrics]
    time_in_state: dict[str, float]
    final_model: RegimeModel
    selection: SelectionReport
    notes: list[str]


def _ll_monitor(model: RegimeModel, raw: np.ndarray, window: int) -> LiveLLMonitor:
    f, alpha, lls = RegimeFilter(model), None, []
    for x in raw[-2000:]:
        o = f.step(alpha, model.scaler.transform(x))
        alpha = o.probs
        lls.append(o.loglik)
    return LiveLLMonitor.from_in_sample(np.array(lls), window=window)


def _kelly(trades: list[Trade], cfg: WFConfig) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        if t.state:
            by[t.state].append(t.ret)
    out = {}
    for label in ("CALM_UP", "CHOP", "STRESS", "CRASH"):
        r = np.array(by.get(label, []))
        if len(r) >= cfg.kelly_min_trades and r.var(ddof=1) > 0:
            out[label] = float(np.clip(r.mean() / r.var(ddof=1), 0.0, 10.0))
        else:
            out[label] = cfg.kelly_prior
    return out


def run_walkforward(bars: pd.DataFrame, cfg: WFConfig, playbooks_dir: Path) -> WFResult:
    books, errors = load_playbooks(playbooks_dir)
    notes = [f"playbook {k} rejected, trades flat: {v}" for k, v in errors.items()]
    notes.append("manual approvals are simulated as granted in the backtest")
    raw_all = compute_features(bars)
    closed = [t.to_pydatetime() for t in pd.DatetimeIndex(bars["closed_at"])]
    candles = frame_to_candles(bars)
    i_oos = next(i for i, t in enumerate(closed) if t > cfg.oos_start)

    train = raw_all.iloc[:i_oos].dropna().to_numpy()
    model, selection = fit_regime_model(train, ks=cfg.ks, restarts=cfg.restarts_initial,
                                        seed=cfg.seed)
    model = model.with_labels(auto_label(model))
    swaps: list[dict[str, Any]] = [{
        "at": cfg.oos_start, "train_end": closed[i_oos - 1], "effective_from": closed[i_oos],
        "accepted": True, "alarms": [], "k": model.k, "labels": list(model.labels),
    }]
    filt = RegimeFilter(model)
    ll_mon = _ll_monitor(model, train, cfg.ll_window)

    risk = RiskGate(KillSwitch(cfg.state_dir / "KILLED"))
    broker = SimBroker(cfg.initial_cash, cfg.costs)
    state = EngineState.initial()
    kelly = {k: cfg.kelly_prior for k in ("CALM_UP", "CHOP", "STRESS", "CRASH")}
    calibrated: dict[str, bool] = {}
    calib_rows: list[tuple[str, float, float, int]] = []
    pending: tuple[str, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    frozen = ll_frozen = False
    candidate: RegimeModel | None = None
    next_refit = cfg.oos_start + timedelta(days=cfg.refit_days)
    last_day = None
    records: list[StepResult] = []
    eq_t: list[datetime] = []
    eq_v: list[float] = []

    for i in range(max(0, i_oos - cfg.warmup_bars), len(candles)):
        c = candles[i]
        trading = i >= i_oos
        if trading:
            broker.on_bar(c)
            if pending:
                label, nps, mus, sds, prev_close = pending
                r = math.log(c.close / prev_close)
                p_up = 1.0 - float(np.sum(nps * norm.cdf((0.0 - mus) / sds)))
                calib_rows.append((label, pit(r, nps, mus, sds), p_up, int(r > 0)))
            day = c.closed_at.astimezone(NY).date()
            if day != last_day:
                last_day = day
                kelly = _kelly(trades_from_fills(broker.fills), cfg)
                rep_c = calibration_report(calib_rows, cfg.calib_min_n, cfg.calib_tol)
                calibrated = rep_c.calibrated
        pending = None

        ctx = Context(
            model=model, filt=filt, playbooks=books, risk=risk, switch=cfg.switch, kelly=kelly,
            calibrated=calibrated, frozen=frozen or ll_frozen, equity=broker.equity(c.close),
            position_qty=broker.qty, entry_price=broker.entry_price, approved=True,
            now=c.closed_at, bar=cfg.bar,
        )
        state, res = on_candle(state, c, ctx)
        if not trading:
            continue
        records.append(res)
        d = res.decision
        broker.submit(res.orders, res.ts, d.get("active"))
        eq_t.append(c.closed_at)
        eq_v.append(broker.equity(c.close))
        if d.get("state_next") and d.get("active"):
            st = state_stats(model)
            pending = (d["active"], np.array(d["state_next"]),
                       np.array([s.mean_ret for s in st]), np.array([s.vol for s in st]), c.close)
        if d.get("loglik") is not None:
            ll_mon.add(float(d["loglik"]))
            if ll_mon.alarm() and not ll_frozen:
                ll_frozen = True
                notes.append(f"{c.closed_at.isoformat()}: live LL alarm, entries frozen")

        if c.closed_at >= next_refit:
            rows = raw_all.iloc[: i + 1].dropna().to_numpy()
            fitted = fit_k(rows, model.k, cfg.restarts_refit, cfg.seed, init=candidate or model)
            new = relabel_refit(model, fitted)
            rep = compare_models(model, new, cfg.drift)
            accepted, why = simulated_review(model, new, candidate, cfg.drift)
            swaps.append({
                "at": c.closed_at, "train_end": closed[i],
                "effective_from": closed[i + 1] if i + 1 < len(closed) else c.closed_at + cfg.bar,
                "accepted": accepted, "alarms": rep.alarms, "k": new.k, "labels": list(new.labels),
                "mean_shift_sigma": rep.max_mean_shift_sigma, "transmat_l1": rep.transmat_l1,
                "review": why,
            })
            if accepted:
                if rep.alarms:  # structure changed: name states afresh from their statistics
                    new = new.with_labels(auto_label(new))
                model, filt, candidate = new, RegimeFilter(new), None
                state.alpha = None
                frozen = ll_frozen = False
                ll_mon = _ll_monitor(model, rows, cfg.ll_window)
                if rep.alarms:
                    notes.append(f"{c.closed_at.isoformat()}: {why}; accepted (simulated review)")
            else:
                frozen, candidate = True, new
                notes.append(f"{c.closed_at.isoformat()}: {why}, entries frozen")
            while next_refit <= c.closed_at:
                next_refit += timedelta(days=cfg.refit_days)

    if risk.kill.active():
        notes.append(f"kill switch fired during backtest: {risk.kill.reason()}")
    equity = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t))
    trades = trades_from_fills(broker.fills)

    per_state: dict[str, dict[str, float]] = {}
    for label in sorted({t.state for t in trades if t.state}):
        ts = [t for t in trades if t.state == label]
        per_state[label] = {
            "trades": float(len(ts)),
            "win_rate": sum(t.pnl > 0 for t in ts) / len(ts),
            "pnl": float(sum(t.pnl for t in ts)),
            "avg_ret": float(np.mean([t.ret for t in ts])),
            "largest_loss": float(min(t.pnl for t in ts)),
        }
    counts = Counter(r.decision.get("active") or "NONE" for r in records)
    time_in_state = {k: v / len(records) for k, v in counts.items()}

    oos_bars = bars.iloc[i_oos:].reset_index(drop=True)
    _, bh = buy_and_hold(oos_bars, cfg.initial_cash, cfg.costs)
    statics = {
        name: static_playbook(oos_bars, pb, cfg.initial_cash, cfg.costs)[1]
        for name, pb in books.items() if pb.style != "flat"
    }
    baselines = {"buy_and_hold": bh}
    if statics:
        best = max(statics, key=lambda k: statics[k].sharpe)
        baselines[f"static_{best}"] = statics[best]

    return WFResult(
        equity=equity, fills=broker.fills, trades=trades, records=records, swaps=swaps,
        calibration=calibration_report(calib_rows, cfg.calib_min_n, cfg.calib_tol),
        metrics=summarize(equity, trades), per_state=per_state, baselines=baselines,
        time_in_state=time_in_state, final_model=model, selection=selection, notes=notes,
    )
