import configparser
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from regimebot.cli import main
from regimebot.ops.watchdog import check_heartbeat
from regimebot.preflight import CHECKS, run_preflight

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


def unit(name: str) -> configparser.ConfigParser:
    c = configparser.ConfigParser(strict=False, interpolation=None)
    c.optionxform = str  # type: ignore[assignment,method-assign]
    c.read(DEPLOY / name)
    return c


def test_live_service_restarts_and_is_sandboxed() -> None:
    u = unit("regimebot.service")
    s = u["Service"]
    assert s["Restart"] == "always" and int(s["RestartSec"]) <= 30
    assert s["User"] not in ("", "root")
    assert "regimebot.cli live" in s["ExecStart"]
    assert s["EnvironmentFile"].endswith(".env")
    for k in ("NoNewPrivileges", "ProtectSystem", "PrivateTmp"):
        assert k in s
    assert u["Install"]["WantedBy"] == "multi-user.target"


@pytest.mark.parametrize("name,cmd", [("regimebot-nightly", "nightly"),
                                      ("regimebot-watchdog", "watchdog")])
def test_timers_have_matching_services(name: str, cmd: str) -> None:
    t = unit(f"{name}.timer")
    assert t["Timer"].get("OnCalendar") and t["Timer"].get("Persistent") == "true"
    s = unit(f"{name}.service")
    assert s["Service"]["Type"] == "oneshot" and f"regimebot.cli {cmd}" in s["Service"]["ExecStart"]


def test_watchdog_fires_kill_on_stale_heartbeat(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    alerts: list[str] = []
    (tmp_path / "heartbeat").write_text((now - timedelta(seconds=90)).isoformat())
    assert check_heartbeat(tmp_path, now, alerts.append) is True
    assert not (tmp_path / "KILLED").exists()
    (tmp_path / "heartbeat").write_text((now - timedelta(minutes=4)).isoformat())
    assert check_heartbeat(tmp_path, now, alerts.append) is False
    assert (tmp_path / "KILLED").exists() and alerts
    assert "heartbeat" in json.loads((tmp_path / "KILLED").read_text())["reason"]


def test_watchdog_missing_heartbeat_is_stale(tmp_path: Path) -> None:
    assert check_heartbeat(tmp_path, datetime.now(UTC), lambda m: None) is False


def test_cli_kill_and_reset(tmp_path: Path) -> None:
    assert main(["kill", "--state", str(tmp_path), "--reason", "manual test"]) == 0
    assert (tmp_path / "KILLED").exists()
    assert main(["kill-reset", "--state", str(tmp_path)]) == 2  # needs written confirmation
    assert (tmp_path / "KILLED").exists()
    assert main(["kill-reset", "--state", str(tmp_path),
                 "--confirm", "I reviewed the positions"]) == 0
    assert not (tmp_path / "KILLED").exists()


def test_cli_approve(tmp_path: Path) -> None:
    from regimebot.exec.approvals import ApprovalQueue

    q = ApprovalQueue(tmp_path / "approvals.json")
    rid = q.request(0.1, 90_000.0, "CALM_UP", datetime.now(UTC))
    assert main(["approve", "--state", str(tmp_path), rid]) == 0
    assert q.granted("CALM_UP", datetime.now(UTC))
    assert main(["approve", "--state", str(tmp_path), "nope"]) == 2


def test_kill_drill_records_evidence(tmp_path: Path, model_file: Path) -> None:
    assert main(["drill-kill", "--state", str(tmp_path), "--model", str(model_file)]) == 0
    rec = json.loads((tmp_path / "drills" / "kill.json").read_text())
    assert rec["fired"] and rec["flattened"] and rec["entries_blocked"] and rec["reset"]
    assert not (tmp_path / "KILLED").exists()


# ---------------- preflight ----------------


def good_env(tmp: Path, model_file: Path) -> dict[str, Any]:
    (tmp / "strategy.md").write_text("# strategy.md\n\n## Status: ACCEPTED\n")
    (tmp / "drills").mkdir(parents=True, exist_ok=True)
    (tmp / "drills" / "kill.json").write_text(json.dumps(
        {"fired": True, "flattened": True, "entries_blocked": True, "reset": True}))
    (tmp / "backtest_summary.json").write_text(json.dumps({
        "metrics": {"sharpe": 1.8, "hit_rate": 0.58},
        "swaps": [{"accepted": True, "alarms": []}] * 10,
    }))
    (tmp / "paper_summary.json").write_text(json.dumps(
        {"days": 45, "trades": 40, "sharpe": 1.6, "hit_rate": 0.56}))
    return {"root": tmp, "state": tmp, "run_tests": lambda selector: True}


def test_preflight_all_clean(tmp_path: Path, model_file: Path) -> None:
    ok, rows = run_preflight(**good_env(tmp_path, model_file))
    assert ok, rows
    assert {r.name for r in rows} == set(CHECKS)


@pytest.mark.parametrize(
    "breakit,check",
    [
        (lambda t: (t / "strategy.md").write_text("NO STRATEGY ACCEPTED"), "strategy_accepted"),
        (lambda t: (t / "drills" / "kill.json").unlink(), "kill_switch_fired_in_testing"),
        (lambda t: (t / "paper_summary.json").write_text(json.dumps(
            {"days": 10, "trades": 5, "sharpe": 1.6, "hit_rate": 0.56})),
         "paper_matches_backtest"),
        (lambda t: (t / "paper_summary.json").write_text(json.dumps(
            {"days": 60, "trades": 50, "sharpe": 0.3, "hit_rate": 0.56})),
         "paper_matches_backtest"),
        (lambda t: (t / "backtest_summary.json").write_text(json.dumps({
            "metrics": {"sharpe": 1.8, "hit_rate": 0.58},
            "swaps": [{"accepted": False, "alarms": ["label_conflict"]}] * 10})),
         "refit_labels_stable"),
    ],
)
def test_preflight_each_check_can_fail(tmp_path: Path, model_file: Path, breakit: Any,
                                       check: str) -> None:
    env = good_env(tmp_path, model_file)
    breakit(tmp_path)
    ok, rows = run_preflight(**env)
    assert not ok and [r.name for r in rows if not r.ok] == [check]


def test_preflight_runs_the_safety_tests(tmp_path: Path, model_file: Path) -> None:
    env = good_env(tmp_path, model_file)
    asked: list[str] = []
    env["run_tests"] = lambda sel: (asked.append(sel), "lookahead" not in sel)[1]
    ok, rows = run_preflight(**env)
    assert not ok and [r.name for r in rows if not r.ok] == ["filtered_probabilities_only"]
    assert any("limits" in s or "research" in s for s in asked)


def test_go_live_doc_answers_the_questions() -> None:
    text = (ROOT / "docs" / "GO_LIVE.md").read_text()
    assert "WHAT COULD BLOW UP THIS ACCOUNT?" in text
    for q in ("Does paper match the backtest", "filtered probabilities only",
              "kill switch fire", "delegated to a model", "state labels stable",
              "What market would break this"):
        assert q.lower() in text.lower(), q
