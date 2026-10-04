import json
import threading
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from regimebot.live import LiveRunner
from regimebot.ops.alerts import Alerter
from regimebot.ops.dashboard import serve, snapshot
from regimebot.ops.report import daily_report, render_report
from test_m13_live import (
    BARS,
    Feed,
    cfg,
    closes,
    drive,
    journal,
)


class Sink:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, payload: dict[str, Any]) -> None:
        self.posts.append((url, payload))


def test_alerter_logs_redacts_and_posts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    sink = Sink()
    a = Alerter(tmp_path / "alerts.jsonl", token="123:SECRET", chat_id="42", post=sink)
    a.send("error", "boom with 123:SECRET inside")
    rec = json.loads((tmp_path / "alerts.jsonl").read_text())
    assert rec["kind"] == "error" and "123:SECRET" not in rec["msg"]
    (url, payload), = sink.posts
    assert url == "https://api.telegram.org/bot123:SECRET/sendMessage"
    assert payload["chat_id"] == "42" and "SECRET" not in payload["text"]


def test_alerter_never_raises(tmp_path: Path) -> None:
    def broken(url: str, payload: dict[str, Any]) -> None:
        raise OSError("telegram down")

    a = Alerter(tmp_path / "alerts.jsonl", token="t", chat_id="c", post=broken)
    a.send("fill", "x")  # must not raise
    lines = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert any(json.loads(x)["kind"] == "alert_error" for x in lines)


def test_alerter_without_token_only_logs(tmp_path: Path) -> None:
    sink = Sink()
    Alerter(tmp_path / "alerts.jsonl", token=None, chat_id=None, post=sink).send("fill", "x")
    assert not sink.posts and (tmp_path / "alerts.jsonl").exists()


def run_with_alerts(tmp: Path, model_file: Path, **kw: Any) -> tuple[LiveRunner, list[str]]:
    kinds: list[str] = []
    a = Alerter(tmp / "alerts.jsonl", token=None, chat_id=None, post=Sink())
    orig = a.send
    a.send = lambda kind, msg: (kinds.append(kind), orig(kind, msg))[1]  # type: ignore[method-assign]
    r = LiveRunner(cfg(tmp, model_file, **kw), Feed(BARS), alerter=a)
    return r, kinds


def test_runner_alerts_on_fill_switch_and_error(tmp_path: Path, model_file: Path) -> None:
    r, kinds = run_with_alerts(tmp_path, model_file, auto_approve=True)
    drive(r, 0, 1400)
    assert {"fill", "switch"} <= set(kinds)
    r.feed.fail = True  # type: ignore[attr-defined]
    r.step(closes(1399) + timedelta(minutes=5))
    assert "error" in kinds


def test_runner_alerts_on_kill(tmp_path: Path, model_file: Path) -> None:
    r, kinds = run_with_alerts(tmp_path, model_file)
    drive(r, 0, 100)
    r.risk.kill.fire("test")
    drive(r, 100, 102)
    assert kinds.count("kill") == 1  # once, not every candle


def test_journal_is_append_only(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file), Feed(BARS))
    drive(r, 0, 200)
    before = (tmp_path / "journal.jsonl").read_text()
    drive(LiveRunner(cfg(tmp_path, model_file), Feed(BARS)), 200, 260)
    after = (tmp_path / "journal.jsonl").read_text()
    assert after.startswith(before) and len(after) > len(before)


def test_daily_report_fields(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file, auto_approve=True), Feed(BARS))
    drive(r, 0, 1400)
    rep = daily_report(journal(tmp_path), r.model)
    for k in ("current_state", "time_in_state", "trades", "pnl_per_state", "win_rate",
              "largest_loss", "calibration", "equity"):
        assert k in rep
    assert abs(sum(rep["time_in_state"].values()) - 1.0) < 1e-9
    text = render_report(rep)
    assert "Current state" in text and "Calibration" in text


def test_dashboard_snapshot_and_server(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file, auto_approve=True), Feed(BARS))
    drive(r, 0, 700)
    snap = snapshot(tmp_path, r.model, r.books)
    for k in ("probs", "current_state", "expected_duration", "playbook", "action", "result"):
        assert k in snap
    server = serve(tmp_path, r.model, r.books, host="127.0.0.1", port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        assert json.loads(urllib.request.urlopen(f"{base}/api/state").read())["current_state"] \
            == snap["current_state"]
        assert b"<html" in urllib.request.urlopen(base + "/").read().lower()
        with urllib.request.urlopen(f"{base}/events?once=1") as ev:
            line = ev.readline().decode()
            assert line.startswith("data: ")
        req = urllib.request.Request(f"{base}/api/state", method="POST", data=b"{}")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 405
    finally:
        server.shutdown()


def test_dashboard_refuses_public_bind(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file), Feed(BARS))
    with pytest.raises(ValueError):
        serve(tmp_path, r.model, r.books, host="0.0.0.0", port=0)
