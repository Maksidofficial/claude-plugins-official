import json
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
from regime_data import regime_bars

from regimebot.data.bars import Candle, frame_to_candles
from regimebot.data.features import compute_features
from regimebot.decide.playbook import load_playbooks
from regimebot.decide.risk import KillSwitch, RiskGate
from regimebot.decide.switcher import SwitchParams
from regimebot.engine import Context, EngineState, StepResult, is_last_bar, on_candle
from regimebot.hmm.filter import FilterError, RegimeFilter
from regimebot.hmm.fit import RegimeModel, fit_regime_model
from regimebot.hmm.labels import auto_label

ROOT = Path(__file__).resolve().parents[1]
BARS, TRUE_STATES = regime_bars(2400, seed=4)
CANDLES = frame_to_candles(BARS)


@pytest.fixture(scope="module")
def model() -> RegimeModel:
    raw = compute_features(BARS.iloc[:1600]).dropna().to_numpy()
    m, _ = fit_regime_model(raw, ks=(3, 4), restarts=3, seed=0)
    return m.with_labels(auto_label(m))


def ctx_for(model: RegimeModel, tmp: Path, **kw: Any) -> Context:
    books, errors = load_playbooks(ROOT / "playbooks")
    assert not errors
    base: dict[str, Any] = dict(
        model=model,
        filt=RegimeFilter(model),
        playbooks=books,
        risk=RiskGate(KillSwitch(tmp / "KILLED")),
        switch=SwitchParams(),
        kelly={"CALM_UP": 2.0, "CHOP": 1.0, "STRESS": 0.5, "CRASH": 0.0},
        calibrated={"CALM_UP": True, "CHOP": True, "STRESS": True, "CRASH": True},
        frozen=False,
        equity=100_000.0,
        position_qty=0.0,
        entry_price=None,
        approved=False,
    )
    base.update(kw)
    return Context(**base)


class Ledger:
    """Fills every order at the candle close. Enough to exercise the engine."""

    def __init__(self) -> None:
        self.cash, self.qty = 100_000.0, 0.0
        self.entry: float | None = None

    def apply(self, res: StepResult, price: float) -> None:
        for o in res.orders:
            if self.qty == 0 and o.qty > 0:
                self.entry = price
            self.cash -= o.qty * price
            self.qty += o.qty
            if self.qty == 0:
                self.entry = None

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price


def run(
    model: RegimeModel,
    tmp: Path,
    candles: list[Candle],
    state: EngineState | None = None,
    **kw: Any,
) -> tuple[EngineState, list[StepResult], Ledger]:
    st, led, out = state or EngineState.initial(), Ledger(), []
    for c in candles:
        ctx = ctx_for(
            model, tmp, equity=led.equity(c.close), position_qty=led.qty,
            entry_price=led.entry, now=c.closed_at, **kw,
        )
        st, res = on_candle(st, c, ctx)
        led.apply(res, c.close)
        out.append(res)
    return st, out, led


def test_deterministic(model: RegimeModel, tmp_path: Path) -> None:
    _, a, _ = run(model, tmp_path, CANDLES[:900])
    _, b, _ = run(model, tmp_path, CANDLES[:900])
    assert [r.to_json() for r in a] == [r.to_json() for r in b]


def test_end_to_end_behaviour(model: RegimeModel, tmp_path: Path) -> None:
    _, out, led = run(model, tmp_path, CANDLES, approved=True)
    assert sum(r.decision["switched"] for r in out) >= 3
    assert sum(1 for r in out if r.orders) >= 4
    held = 0.0
    for r in out:
        held += sum(o.qty for o in r.orders)
        assert held >= 0
        if r.decision["active"] == "CRASH" or r.decision["uncertain"]:
            assert held == 0, r.decision


def test_vetoes_are_logged(model: RegimeModel, tmp_path: Path) -> None:
    _, out, _ = run(model, tmp_path, CANDLES[:1200], frozen=True)
    vetoes = [v for r in out for v in r.vetoes]
    assert vetoes and all(v["rule"] == "frozen" for v in vetoes)
    assert all(not r.orders for r in out)


def test_kill_switch_flattens(model: RegimeModel, tmp_path: Path) -> None:
    st, _, _ = run(model, tmp_path, CANDLES[:700])
    KillSwitch(tmp_path / "KILLED").fire("test")
    c = CANDLES[700]
    ctx = ctx_for(model, tmp_path, position_qty=25.0, entry_price=c.close, now=c.closed_at)
    _, res = on_candle(st, c, ctx)
    assert [o.qty for o in res.orders] == [-25.0]
    assert "kill" in res.decision["reason"]


def test_stale_candle_flattens(model: RegimeModel, tmp_path: Path) -> None:
    st, _, _ = run(model, tmp_path, CANDLES[:700])
    c = CANDLES[700]
    late = c.closed_at + (c.closed_at - c.opened_at) * 5
    ctx = ctx_for(model, tmp_path, position_qty=10.0, entry_price=c.close, now=late)
    _, res = on_candle(st, c, ctx)
    assert [o.qty for o in res.orders] == [-10.0]
    assert "stale" in res.decision["reason"]


class BrokenFilter:
    def step(self, prev: Any, z: Any) -> Any:
        raise FilterError("boom")


def test_model_error_flattens(model: RegimeModel, tmp_path: Path) -> None:
    st, _, _ = run(model, tmp_path, CANDLES[:700])
    c = CANDLES[700]
    ctx = ctx_for(model, tmp_path, filt=BrokenFilter(), position_qty=10.0,
                  entry_price=c.close, now=c.closed_at)
    _, res = on_candle(st, c, ctx)
    assert [o.qty for o in res.orders] == [-10.0]
    assert "error" in res.decision["reason"]


def test_state_round_trip_resumes_exactly(model: RegimeModel, tmp_path: Path) -> None:
    st, _, _ = run(model, tmp_path, CANDLES[:1000])
    restored = EngineState.from_json(st.to_json())
    assert restored.to_json() == st.to_json()
    tail = CANDLES[1000:1200]
    _, a, _ = run(model, tmp_path, tail, state=st)
    _, b, _ = run(model, tmp_path, tail, state=restored)
    assert [r.to_json() for r in a] == [r.to_json() for r in b]


def test_records_are_json(model: RegimeModel, tmp_path: Path) -> None:
    _, out, _ = run(model, tmp_path, CANDLES[:300])
    for r in out:
        d = json.loads(r.to_json())
        assert {"ts", "decision", "orders", "vetoes"} <= d.keys()


def test_engine_never_looks_ahead(model: RegimeModel, tmp_path: Path) -> None:
    k = 900
    _, base, _ = run(model, tmp_path, CANDLES[:1100])
    noise, _ = regime_bars(1100, seed=99)
    shift = CANDLES[k + 1].opened_at - noise["opened_at"].iloc[k + 1]
    noise["opened_at"] += shift
    noise["closed_at"] += shift
    tampered = CANDLES[: k + 1] + frame_to_candles(noise.iloc[k + 1 :])
    _, alt, _ = run(model, tmp_path, tampered)
    assert [r.to_json() for r in alt[: k + 1]] == [r.to_json() for r in base[: k + 1]]
    assert [r.to_json() for r in alt[k + 1 :]] != [r.to_json() for r in base[k + 1 :]]


@pytest.mark.parametrize(
    "open_ny,close_ny,expected",
    [
        ("15:30", "16:00", True),  # RTH last bar (Alpaca 1h bars end with a 30-min bar)
        ("15:00", "16:00", True),
        ("15:30", "16:30", True),  # spans the close
        ("16:00", "17:00", False),  # after hours
        ("20:30", "21:30", False),
        ("14:30", "15:30", False),
    ],
)
def test_last_bar_spans_the_close(open_ny: str, close_ny: str, expected: bool) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")

    def at(hm: str) -> datetime:
        h, m = map(int, hm.split(":"))
        return datetime(2025, 3, 14, h, m, tzinfo=ny)

    c = Candle(at(open_ny), at(close_ny), 1.0, 1.0, 1.0, 1.0, 1.0)
    assert is_last_bar(c) is expected


def test_uncalibrated_label_never_sizes(model: RegimeModel, tmp_path: Path) -> None:
    _, out, _ = run(model, tmp_path, CANDLES, approved=True, calibrated={})
    assert all(not r.orders for r in out)


def test_fractional_quantity_step(model: RegimeModel, tmp_path: Path) -> None:
    from regimebot.engine import round_qty

    assert round_qty(0.2345678, 1e-5) == pytest.approx(0.23456)
    assert round_qty(12.9, 1.0) == 12.0


def test_no_session_close_means_no_overnight_cap() -> None:
    from datetime import datetime

    from regimebot.engine import is_last_bar

    c = Candle(datetime(2025, 3, 14, 19, 30, tzinfo=UTC), datetime(2025, 3, 14, 20, 30, tzinfo=UTC),
               1.0, 1.0, 1.0, 1.0, 1.0)
    assert is_last_bar(c) is True  # spans 16:00 New York (equities default)
    assert is_last_bar(c, session_close=None) is False
