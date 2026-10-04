import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from regime_data import regime_bars
from regimebot.backtest.costs import CostModel
from regimebot.data.features import compute_features
from regimebot.decide.switcher import SwitchParams
from regimebot.exec.approvals import ApprovalQueue
from regimebot.exec.paper import PaperBroker
from regimebot.hmm.fit import fit_regime_model
from regimebot.hmm.labels import auto_label
from regimebot.live import LiveConfig, LiveRunner

ROOT = Path(__file__).resolve().parents[1]
BARS, _ = regime_bars(1400, seed=4)
H = timedelta(hours=1)


@pytest.fixture(scope="module")
def model_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raw = compute_features(BARS.iloc[:900]).dropna().to_numpy()
    m, _ = fit_regime_model(raw, ks=(4,), restarts=2, seed=0)
    m = m.with_labels(auto_label(m))
    p = tmp_path_factory.mktemp("models") / "current.json"
    p.write_text(json.dumps(m.to_dict()))
    return p


class Feed:
    """Serves the synthetic candles that have closed by ``now``."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars
        self.calls = 0
        self.fail = False

    def __call__(self, since: datetime, now: datetime) -> pd.DataFrame:
        self.calls += 1
        if self.fail:
            raise OSError("network down")
        b = self.bars
        return b[(b["opened_at"] >= since) & (b["closed_at"] <= now)].reset_index(drop=True)


def cfg(tmp: Path, model_file: Path, **kw: Any) -> LiveConfig:
    base: dict[str, Any] = dict(
        state_dir=tmp, model_path=model_file, playbooks_dir=ROOT / "playbooks",
        symbol="SYNTH", bar=H, start=BARS["opened_at"].iloc[0].to_pydatetime(),
        costs=CostModel(10.0, 2.0), qty_step=1e-5, session_close=None, tz="UTC",
        switch=SwitchParams(), initial_cash=100_000.0,
        calibrated={"CALM_UP": True, "CHOP": True, "STRESS": True, "CRASH": False},
        kelly={"CALM_UP": 2.0, "CHOP": 1.0, "STRESS": 0.5, "CRASH": 0.0},
    )
    base.update(kw)
    return LiveConfig(**base)


def closes(i: int) -> datetime:
    return BARS["closed_at"].iloc[i].to_pydatetime()


def journal(tmp: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (tmp / "journal.jsonl").read_text().splitlines()]


def drive(r: "LiveRunner", i0: int, i1: int) -> None:
    """Step like production: shortly after each candle closes."""
    for i in range(i0, i1):
        r.step(closes(i) + timedelta(seconds=20))


def decisions(tmp: Path) -> list[str]:
    return [json.dumps(j["result"], sort_keys=True) for j in journal(tmp) if j["kind"] == "candle"]


def test_processes_each_closed_candle_once(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file), Feed(BARS))
    r.step(closes(300) + timedelta(seconds=20))
    r.step(closes(300) + timedelta(seconds=50))  # nothing new
    r.step(closes(305) + timedelta(seconds=20))
    assert len(decisions(tmp_path)) == 306


def test_restart_resumes_exactly(tmp_path: Path, model_file: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    feed = Feed(BARS)
    drive(LiveRunner(cfg(a, model_file, auto_approve=True), feed), 0, 1300)
    for i in range(0, 1300, 100):  # a fresh process every 100 candles
        drive(LiveRunner(cfg(b, model_file, auto_approve=True), feed), i, i + 100)
    assert any(j["result"]["orders"] for j in journal(a) if j["kind"] == "candle")
    assert decisions(a) == decisions(b)
    assert PaperBroker.load(a / "broker.json").fills == PaperBroker.load(b / "broker.json").fills


def test_entries_always_carry_a_stop(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file, auto_approve=True), Feed(BARS))
    drive(r, 0, 1400)
    entries = [o for j in journal(tmp_path) if j["kind"] == "candle"
               for o in j["result"]["orders"] if o["qty"] > 0]
    assert entries, "expected entries on synthetic regimes"
    assert all(o["stop_price"] and o["stop_price"] < o["price"] for o in entries)


def test_stale_feed_flattens(tmp_path: Path, model_file: Path) -> None:
    feed = Feed(BARS)
    r = LiveRunner(cfg(tmp_path, model_file), feed)
    r.step(closes(800) + timedelta(seconds=20))
    r.broker.qty, r.broker.entry_price = 0.1, 400.0  # pretend we hold a position
    r.broker.save(tmp_path / "broker.json")
    feed.fail = True
    r.step(closes(800) + 3 * H)
    stale = [j for j in journal(tmp_path) if j["kind"] == "stale"]
    assert stale and stale[-1]["orders"][0]["qty"] == pytest.approx(-0.1)


def test_reconcile_mismatch_fires_kill(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file), Feed(BARS))
    r.step(closes(400) + timedelta(seconds=20))
    b = PaperBroker.load(tmp_path / "broker.json")
    b.qty = 5.0  # broker state no longer matches the journal
    b.save(tmp_path / "broker.json")
    LiveRunner(cfg(tmp_path, model_file), Feed(BARS))
    assert (tmp_path / "KILLED").exists()
    assert "reconcile" in json.loads((tmp_path / "KILLED").read_text())["reason"]


def test_live_uses_the_shared_engine(tmp_path: Path, model_file: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    import regimebot.live as live

    calls = []
    real = live.on_candle
    monkeypatch.setattr(live, "on_candle", lambda *a: calls.append(1) or real(*a))
    LiveRunner(cfg(tmp_path, model_file), Feed(BARS)).step(closes(100) + timedelta(seconds=20))
    assert len(calls) == 101


def test_large_orders_wait_for_approval(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file), Feed(BARS))
    drive(r, 0, 1400)
    q = ApprovalQueue(tmp_path / "approvals.json")
    assert q.pending()
    assert all(f.qty <= 0 or f.qty * f.price <= 5_000 * 1.01 for f in r.broker.fills)


def test_approval_queue(tmp_path: Path) -> None:
    q = ApprovalQueue(tmp_path / "approvals.json")
    now = datetime.fromisoformat("2025-01-01T00:00:00+00:00")
    rid = q.request(qty=0.2, price=90_000.0, state="CALM_UP", now=now)
    assert q.request(qty=0.21, price=90_100.0, state="CALM_UP", now=now) == rid  # deduped
    assert not q.granted("CALM_UP", now)
    q.grant(rid, now=now)
    assert q.granted("CALM_UP", now + timedelta(minutes=90))
    assert not q.granted("CALM_UP", now + timedelta(hours=3))  # grants expire
    assert not q.granted("CHOP", now)


def test_live_needs_no_secrets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                               model_file: Path) -> None:
    for k in ("ALPACA_KEY_ID", "ALPACA_SECRET_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    LiveRunner(cfg(tmp_path, model_file), Feed(BARS)).step(closes(60) + timedelta(seconds=20))


def test_catch_up_history_is_ingested_but_never_traded(tmp_path: Path, model_file: Path) -> None:
    r = LiveRunner(cfg(tmp_path, model_file, auto_approve=True), Feed(BARS))
    r.step(closes(1399) + timedelta(seconds=20))  # first start: 1400 old candles at once
    recs = [j for j in journal(tmp_path) if j["kind"] == "candle"]
    assert len(recs) == 1400
    assert all(not j["result"]["orders"] for j in recs[:-1])
