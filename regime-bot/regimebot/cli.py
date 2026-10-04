"""Command line entry points used by systemd and by the operator.

    python -m regimebot.cli live            # paper-trade forever (systemd: regimebot.service)
    python -m regimebot.cli dashboard       # read-only dashboard on 127.0.0.1:8765
    python -m regimebot.cli watchdog        # once a minute (regimebot-watchdog.timer)
    python -m regimebot.cli nightly         # Opus 5.5 review + gated promotion + daily report
    python -m regimebot.cli report          # print today's report
    python -m regimebot.cli approve <id>    # grant a pending large-order approval
    python -m regimebot.cli kill --reason ...   /   kill-reset --confirm "..."
    python -m regimebot.cli drill-kill      # prove the kill switch flattens and blocks
    python -m regimebot.cli preflight       # go-live gate, non-zero exit on any failure
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state" / "live"


def _alerter(state: Path) -> Any:
    from regimebot.ops.alerts import Alerter

    return Alerter(state / "alerts.jsonl", token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
                   chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None)


def _live_config(state: Path, model: Path) -> Any:
    from regimebot.backtest.costs import CostModel
    from regimebot.decide.switcher import SwitchParams
    from regimebot.live import LiveConfig

    params = {}
    p = state.parent / "live_params.json"
    if p.exists():
        params = json.loads(p.read_text())
    return LiveConfig(
        state_dir=state, model_path=model, playbooks_dir=ROOT / "playbooks",
        symbol=params.get("symbol", "BTCUSDT"), bar=timedelta(hours=1),
        start=datetime.now(UTC) - timedelta(days=10), costs=CostModel(10.0, 2.0),
        qty_step=1e-5, session_close=None, tz="UTC", switch=SwitchParams(),
        initial_cash=float(params.get("initial_cash", 100_000.0)),
        calibrated=params.get("calibrated", {}),  # empty: every label sizes to zero
        kelly=params.get("kelly", {}),
        frozen=bool(params.get("frozen", False)),
    )


def cmd_live(a: argparse.Namespace) -> int:
    from regimebot.data.binance import fetch_klines
    from regimebot.live import LiveRunner, run_forever

    cfg = _live_config(a.state, a.model)
    runner = LiveRunner(cfg, lambda since, now: fetch_klines(cfg.symbol, "1h", since, now),
                        alerter=_alerter(a.state))
    run_forever(runner)
    return 0


def cmd_dashboard(a: argparse.Namespace) -> int:
    from regimebot.decide.playbook import load_playbooks
    from regimebot.hmm.fit import RegimeModel
    from regimebot.ops.dashboard import serve

    model = RegimeModel.from_dict(json.loads(a.model.read_text()))
    books, _ = load_playbooks(ROOT / "playbooks")
    serve(a.state, model, books, host="127.0.0.1", port=a.port).serve_forever()
    return 0


def cmd_watchdog(a: argparse.Namespace) -> int:
    from regimebot.ops.watchdog import check_heartbeat

    al = _alerter(a.state)
    return 0 if check_heartbeat(a.state, datetime.now(UTC), lambda m: al.send("kill", m)) else 1


def cmd_report(a: argparse.Namespace) -> int:
    from regimebot.hmm.fit import RegimeModel
    from regimebot.ops.report import daily_report, render_report
    from regimebot.research.nightly import load_journal

    model = RegimeModel.from_dict(json.loads(a.model.read_text()))
    rep = daily_report(load_journal(a.state / "journal.jsonl"), model, a.day)
    print(render_report(rep))
    return 0


def cmd_nightly(a: argparse.Namespace) -> int:
    from regimebot.backtest.gates import Gates, evaluate
    from regimebot.backtest.report import render
    from regimebot.backtest.walkforward import run_walkforward
    from regimebot.hmm.fit import RegimeModel
    from regimebot.ops.report import daily_report, paper_summary, render_report
    from regimebot.research.model import OpusResearch
    from regimebot.research.nightly import load_journal, run_nightly
    from regimebot.research.validate import validate_and_promote

    al = _alerter(a.state)
    day = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    records = load_journal(a.state / "journal.jsonl")
    model = RegimeModel.from_dict(json.loads(a.model.read_text()))
    al.send("report", render_report(daily_report(records, model, day)))
    (a.state / "paper_summary.json").write_text(json.dumps(paper_summary(records), indent=1))
    try:
        research = OpusResearch()
    except Exception as e:  # SDK missing or no credentials: trading is unaffected
        al.send("error", f"nightly review skipped: {e}")
        return 0
    proposal = run_nightly(records, ROOT, research, day)
    if proposal is None or not proposal.playbook_changes:
        return 0

    from regimebot.backtest.run import config_for, load_bars

    bars = load_bars()

    def backtest(pb_dir: Path) -> tuple[str, Any]:
        res = run_walkforward(bars, config_for(), pb_dir)
        gate = evaluate(res.metrics, res.baselines, Gates())
        return render(res, gate), gate

    out = validate_and_promote(proposal, ROOT, backtest)
    al.send("drift" if not out.promoted else "report", f"nightly proposal: {out.reason}")
    return 0


def cmd_approve(a: argparse.Namespace) -> int:
    from regimebot.exec.approvals import ApprovalQueue

    try:
        ApprovalQueue(a.state / "approvals.json").grant(a.id, datetime.now(UTC))
    except KeyError as e:
        print(e)
        return 2
    print(f"granted {a.id}")
    return 0


def cmd_kill(a: argparse.Namespace) -> int:
    from regimebot.decide.risk import KillSwitch

    KillSwitch(a.state / "KILLED").fire(f"manual: {a.reason}")
    _alerter(a.state).send("kill", f"KILL SWITCH (manual): {a.reason}")
    return 0


def cmd_kill_reset(a: argparse.Namespace) -> int:
    from regimebot.decide.risk import KillSwitch

    try:
        KillSwitch(a.state / "KILLED").reset(a.confirm or "")
    except ValueError as e:
        print(e)
        return 2
    _alerter(a.state).send("kill", f"kill switch reset by operator: {a.confirm}")
    return 0


def cmd_drill_kill(a: argparse.Namespace) -> int:
    """Fire the kill switch against a held position and record what happened."""
    from regimebot.data.bars import Candle
    from regimebot.decide.playbook import load_playbooks
    from regimebot.decide.risk import Account, KillSwitch, Order, RiskGate
    from regimebot.decide.switcher import SwitchParams
    from regimebot.engine import Context, EngineState, on_candle
    from regimebot.hmm.filter import RegimeFilter
    from regimebot.hmm.fit import RegimeModel

    kill = KillSwitch(a.state / "KILLED")
    if kill.active():
        print("kill switch already active; refusing to drill over a real event")
        return 2
    model = RegimeModel.from_dict(json.loads(a.model.read_text()))
    books, _ = load_playbooks(ROOT / "playbooks")
    risk = RiskGate(kill)
    kill.fire("drill")
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    c = Candle(now - timedelta(hours=1), now, 100.0, 101.0, 99.0, 100.0, 1.0)
    ctx = Context(model=model, filt=RegimeFilter(model), playbooks=books, risk=risk,
                  switch=SwitchParams(), kelly={}, calibrated={}, frozen=False,
                  equity=100_000.0, position_qty=1.5, entry_price=100.0, approved=True,
                  now=now)
    _, res = on_candle(EngineState.initial(), c, ctx)
    flattened = [o.qty for o in res.orders] == [-1.5]
    veto = risk.check(Order(1.0, 100.0), Account(100_000.0, 100_000.0, 100_000.0, 0.0),
                      "CALM_UP", approved=True)
    blocked = veto is not None and veto.rule == "kill_switch"
    kill.reset("drill complete")
    rec = {"at": now.isoformat(), "fired": True, "flattened": flattened,
           "entries_blocked": blocked, "reset": not kill.active()}
    (a.state / "drills").mkdir(parents=True, exist_ok=True)
    (a.state / "drills" / "kill.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return 0 if flattened and blocked else 1


def cmd_preflight(a: argparse.Namespace) -> int:
    from regimebot.preflight import run_preflight

    ok, rows = run_preflight(ROOT, a.state)
    for r in rows:
        print(f"[{'PASS' if r.ok else 'FAIL'}] {r.name}: {r.detail}")
    print("GO-LIVE: " + ("all checks clean" if ok else "REFUSED"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="regimebot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Any, model: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--state", type=Path, default=STATE)
        if model:
            p.add_argument("--model", type=Path, default=ROOT / "state" / "models" / "current.json")
        p.set_defaults(fn=fn)
        return p

    add("live", cmd_live, model=True)
    add("dashboard", cmd_dashboard, model=True).add_argument("--port", type=int, default=8765)
    add("watchdog", cmd_watchdog)
    add("nightly", cmd_nightly, model=True)
    add("report", cmd_report, model=True).add_argument("--day", default=None)
    add("approve", cmd_approve).add_argument("id")
    add("kill", cmd_kill).add_argument("--reason", required=True)
    add("kill-reset", cmd_kill_reset).add_argument("--confirm", default=None)
    add("drill-kill", cmd_drill_kill, model=True)
    add("preflight", cmd_preflight)
    a = ap.parse_args(argv)
    a.state.mkdir(parents=True, exist_ok=True)
    rc: int = a.fn(a)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
