from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from regime_data import regime_bars
from scipy.stats import norm

from regimebot.backtest.baselines import buy_and_hold, static_playbook
from regimebot.backtest.calibration import (
    brier,
    calibration_report,
    pit,
    reliability_max_dev,
)
from regimebot.backtest.costs import CostModel
from regimebot.backtest.gates import Gates, evaluate
from regimebot.backtest.metrics import Metrics, Trade, max_drawdown, summarize, trades_from_fills
from regimebot.backtest.report import write_strategy
from regimebot.backtest.walkforward import WFConfig, WFResult, run_walkforward
from regimebot.data.bars import Candle
from regimebot.decide.playbook import load_playbooks
from regimebot.engine import OrderAction
from regimebot.exec.broker import Fill, SimBroker

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2025, 1, 6, 14, 30, tzinfo=UTC)
H = timedelta(hours=1)


def candle(i: int, o: float, h: float, lo: float, c: float) -> Candle:
    return Candle(T0 + i * H, T0 + (i + 1) * H, o, h, lo, c, 1e6)


COSTS = CostModel(commission_bps=1.0, slippage_bps=2.0)


# ---------- costs + broker ----------


def test_costs_per_side() -> None:
    assert COSTS.fill_price(100.0, +10) == pytest.approx(100.02)
    assert COSTS.fill_price(100.0, -10) == pytest.approx(99.98)
    assert COSTS.commission(100.0, -10) == pytest.approx(0.1)


def test_market_order_fills_at_next_open_not_close() -> None:
    b = SimBroker(10_000.0, COSTS)
    b.submit([OrderAction(10, 101.0, None, None, "entry")], ts="t0")  # decided at close 101
    b.on_bar(candle(1, o=103.0, h=104.0, lo=102.5, c=103.5))
    (f,) = b.fills
    assert f.price == pytest.approx(103.0 * 1.0002)
    assert b.qty == 10
    assert b.cash == pytest.approx(10_000 - 10 * f.price - f.commission)


def test_stop_fills_at_stop_or_gap_open() -> None:
    b = SimBroker(10_000.0, COSTS)
    b.submit([OrderAction(10, 100.0, 98.0, 110.0, "entry")], ts="t0")
    b.on_bar(candle(1, 100.0, 100.5, 99.5, 100.0))
    b.on_bar(candle(2, 99.0, 99.2, 97.0, 97.5))  # trades through 98
    assert b.qty == 0
    assert b.fills[-1].price == pytest.approx(98.0 * 0.9998)
    assert b.fills[-1].reason == "stop"

    b = SimBroker(10_000.0, COSTS)
    b.submit([OrderAction(10, 100.0, 98.0, 110.0, "entry")], ts="t0")
    b.on_bar(candle(1, 100.0, 100.5, 99.5, 100.0))
    b.on_bar(candle(2, 95.0, 96.0, 94.0, 95.5))  # gaps below the stop
    assert b.fills[-1].price == pytest.approx(95.0 * 0.9998)


def test_stop_beats_take_profit_in_same_bar() -> None:
    b = SimBroker(10_000.0, COSTS)
    b.submit([OrderAction(10, 100.0, 98.0, 102.0, "entry")], ts="t0")
    b.on_bar(candle(1, 100.0, 100.5, 99.5, 100.0))
    b.on_bar(candle(2, 100.0, 103.0, 97.0, 100.0))
    assert b.fills[-1].reason == "stop"


def test_take_profit() -> None:
    b = SimBroker(10_000.0, COSTS)
    b.submit([OrderAction(10, 100.0, 98.0, 102.0, "entry")], ts="t0")
    b.on_bar(candle(1, 100.0, 100.5, 99.5, 100.0))
    b.on_bar(candle(2, 100.5, 102.5, 100.2, 102.2))
    assert b.qty == 0 and b.fills[-1].reason == "take_profit"
    assert b.fills[-1].price == pytest.approx(102.0 * 0.9998)


def test_bracket_cleared_after_exit() -> None:
    b = SimBroker(10_000.0, COSTS)
    b.submit([OrderAction(10, 100.0, 98.0, 102.0, "entry")], ts="t0")
    b.on_bar(candle(1, 100.0, 100.5, 99.5, 100.0))
    b.submit([OrderAction(-10, 100.0, None, None, "exit")], ts="t1")
    b.on_bar(candle(2, 100.0, 100.1, 99.9, 100.0))
    b.on_bar(candle(3, 100.0, 103.0, 90.0, 100.0))
    assert b.qty == 0 and len(b.fills) == 2


# ---------- metrics ----------


def test_trades_from_fills() -> None:
    fills = [
        Fill("a", 10, 100.0, 0.1, "entry", "CALM_UP"),
        Fill("b", -10, 110.0, 0.11, "exit", "CALM_UP"),
        Fill("c", 5, 100.0, 0.05, "entry", "CHOP"),
        Fill("d", -5, 95.0, 0.05, "stop", "CHOP"),
    ]
    t1, t2 = trades_from_fills(fills)
    assert t1.state == "CALM_UP" and t1.pnl == pytest.approx(100 - 0.21)
    assert t2.state == "CHOP" and t2.pnl == pytest.approx(-25 - 0.10)
    assert t1.ret == pytest.approx((100 - 0.21) / 1000)


def test_max_drawdown() -> None:
    eq = pd.Series([100, 120, 90, 130, 117.0])
    assert max_drawdown(eq) == pytest.approx(0.25)


def test_summarize_known_series() -> None:
    days = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    eq = pd.Series([100.0, 101.0, 100.0, 102.0, 103.0], index=days)
    trades = [Trade("a", "b", "X", 1, 1, 1, 1.0, 0.01), Trade("a", "b", "X", 1, 1, 1, -1.0, -0.01)]
    m = summarize(eq, trades)
    r = eq.pct_change().dropna()
    assert m.sharpe == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert m.t_stat == pytest.approx(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
    assert m.hit_rate == 0.5 and m.n_trades == 2
    assert m.total_return == pytest.approx(0.03)


# ---------- gates ----------


GOOD = Metrics(sharpe=2.0, max_dd=0.08, hit_rate=0.6, t_stat=3.0, total_return=0.2, n_trades=80)
BASE = {
    "buy_and_hold": Metrics(1.0, 0.2, 0.0, 1.0, 0.3, 1),
    "static": Metrics(1.2, 0.1, 0.5, 1.5, 0.1, 50),
}


def test_gates_pass() -> None:
    res = evaluate(GOOD, BASE, Gates())
    assert res.passed and all(r.ok for r in res.rows)


@pytest.mark.parametrize(
    "field,value,gate",
    [
        ("sharpe", 1.5, "sharpe"),
        ("max_dd", 0.15, "max_drawdown"),
        ("hit_rate", 0.55, "hit_rate"),
        ("t_stat", 2.0, "t_stat"),
    ],
)
def test_each_gate_can_fail(field: str, value: float, gate: str) -> None:
    from dataclasses import replace

    res = evaluate(replace(GOOD, **{field: value}), BASE, Gates())
    assert not res.passed
    assert [r.name for r in res.rows if not r.ok] == [gate]


def test_must_beat_each_baseline() -> None:
    res = evaluate(GOOD, {**BASE, "static": Metrics(2.5, 0.1, 0.6, 3, 0.1, 50)}, Gates())
    assert not res.passed and [r.name for r in res.rows if not r.ok] == ["beats_static"]


def test_too_few_trades_fails() -> None:
    from dataclasses import replace

    res = evaluate(replace(GOOD, n_trades=10), BASE, Gates())
    assert not res.passed


# ---------- calibration ----------


def test_pit_of_mixture() -> None:
    p = pit(0.01, np.array([0.3, 0.7]), np.array([0.0, 0.02]), np.array([0.01, 0.02]))
    exp = 0.3 * norm.cdf(1.0) + 0.7 * norm.cdf(-0.5)
    assert p == pytest.approx(exp)


def test_brier() -> None:
    assert brier(np.array([1.0, 0.0, 0.5]), np.array([1, 0, 1])) == pytest.approx(0.25 / 3)


def test_calibrated_vs_overconfident() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, 4000)
    good = norm.cdf(r / 0.02)
    over = norm.cdf(r / 0.005)  # model claims 4x less vol than reality
    assert reliability_max_dev(good) < 0.05
    assert reliability_max_dev(over) > 0.10


def test_calibration_report_per_label() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(400):
        x = rng.normal(0, 0.01)
        rows.append(("CALM_UP", float(norm.cdf(x / 0.01)), 0.5, int(x > 0)))
        rows.append(("CHOP", float(norm.cdf(x / 0.001)), 0.5, int(x > 0)))
    rows += [("STRESS", 0.5, 0.5, 1)] * 10
    rep = calibration_report(rows, min_n=200, tol=0.10)
    assert rep.calibrated == {"CALM_UP": True, "CHOP": False, "STRESS": False}
    assert rep.per_label["STRESS"]["n"] == 10


# ---------- baselines ----------


def test_buy_and_hold() -> None:
    days = pd.date_range("2025-01-06 14:30", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"opened_at": days, "closed_at": days + pd.Timedelta(hours=1),
                       "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6})
    df.loc[47, ["close", "high"]] = 110.0
    eq, m = buy_and_hold(df, 10_000.0, COSTS)
    shares = (10_000 / (100 * 1.0002))
    assert eq.iloc[-1] == pytest.approx(10_000 + shares * (110 - 100 * 1.0002) - 2 * 0.0, rel=1e-3)
    assert m.n_trades == 1


def test_static_playbook_runs() -> None:
    bars, _ = regime_bars(1500, seed=2)
    books, _ = load_playbooks(ROOT / "playbooks")
    eq, m, fills = static_playbook(bars, books["CALM_UP"], 100_000.0, COSTS)
    assert len(eq) == len(bars) and m.n_trades > 0
    assert all(f.qty * f.price <= 0.2 * 100_000 * 1.05 for f in fills if f.qty > 0)


# ---------- walk-forward ----------


@pytest.fixture(scope="module")
def wf(tmp_path_factory: pytest.TempPathFactory):
    bars, _ = regime_bars(3200, seed=4)
    oos = bars["closed_at"].iloc[1700].to_pydatetime()
    cfg = WFConfig(oos_start=oos, refit_days=30, ks=(4,), restarts_initial=3, restarts_refit=2,
                   seed=0, calib_min_n=150, state_dir=tmp_path_factory.mktemp("state"))
    return bars, oos, run_walkforward(bars, cfg, playbooks_dir=ROOT / "playbooks")


def test_walkforward_refits_use_only_past(wf) -> None:
    bars, oos, res = wf
    assert len(res.swaps) >= 2
    for s in res.swaps:
        assert s["train_end"] <= s["at"]
        assert s["effective_from"] > s["at"]


def test_walkforward_trades_only_after_oos_start(wf) -> None:
    _, oos, res = wf
    assert res.fills, "expected some trades on the synthetic regimes"
    assert all(datetime.fromisoformat(f.ts) > oos for f in res.fills)


def test_walkforward_outputs(wf) -> None:
    bars, oos, res = wf
    assert res.equity.index.min() > oos
    assert set(res.per_state) <= {"CALM_UP", "CHOP", "STRESS", "CRASH"}
    assert "buy_and_hold" in res.baselines
    assert any(k.startswith("static_") for k in res.baselines)
    assert 0.0 <= res.metrics.max_dd <= 1.0
    assert sum(res.time_in_state.values()) == pytest.approx(1.0)


def test_walkforward_no_position_when_crash_or_uncertain(wf) -> None:
    _, _, res = wf
    for r in res.records:
        d = r.decision
        if d.get("active") == "CRASH" or d.get("uncertain"):
            assert all(o.qty < 0 for o in r.orders)


def test_strategy_md(wf, tmp_path: Path) -> None:
    _, _, res = wf
    gate = evaluate(res.metrics, res.baselines, Gates())
    p = tmp_path / "strategy.md"
    write_strategy(p, res, gate)
    text = p.read_text()
    if gate.passed:
        assert "ACCEPTED" in text and "NO STRATEGY" not in text
    else:
        assert "NO STRATEGY ACCEPTED" in text
    for name in ("sharpe", "max_drawdown", "hit_rate", "t_stat"):
        assert name in text


def test_walkforward_never_looks_ahead(tmp_path_factory: pytest.TempPathFactory) -> None:
    bars, _ = regime_bars(2600, seed=8)
    oos = bars["closed_at"].iloc[1700].to_pydatetime()
    k = 2200  # after the initial fit and the first 15-day refit (~bar 2060)

    def run(b: pd.DataFrame) -> WFResult:
        cfg = WFConfig(oos_start=oos, refit_days=15, ks=(3,), restarts_initial=2,
                       restarts_refit=2, seed=0, calib_min_n=100,
                       state_dir=tmp_path_factory.mktemp("s"))
        return run_walkforward(b, cfg, playbooks_dir=ROOT / "playbooks")

    noise, _ = regime_bars(2600, seed=123)
    tampered = bars.copy()
    for col in ("open", "high", "low", "close", "volume"):
        tampered.loc[k + 1 :, col] = noise.loc[k + 1 :, col].to_numpy()
    a, b = run(bars), run(tampered)
    cut = bars["closed_at"].iloc[k]
    past_a = [r.to_json() for r in a.records if datetime.fromisoformat(r.ts) <= cut]
    past_b = [r.to_json() for r in b.records if datetime.fromisoformat(r.ts) <= cut]
    assert past_a and past_a == past_b
    assert [s for s in a.swaps if s["at"] <= cut] == [s for s in b.swaps if s["at"] <= cut]
    assert [r.to_json() for r in a.records] != [r.to_json() for r in b.records]
