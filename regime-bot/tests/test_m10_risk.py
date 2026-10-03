from pathlib import Path

import pytest

from regimebot.decide.risk import Account, KillSwitch, Order, RiskGate


def acct(**kw: float) -> Account:
    base = dict(equity=100_000.0, peak_equity=100_000.0, day_start_equity=100_000.0,
                position_qty=0.0)
    base.update(kw)
    return Account(**base)


def buy(qty: float, price: float = 100.0) -> Order:
    return Order(qty=qty, price=price)


@pytest.fixture
def gate(tmp_path: Path) -> RiskGate:
    return RiskGate(KillSwitch(tmp_path / "KILLED"))


def test_small_entry_passes(gate: RiskGate) -> None:
    assert gate.check(buy(40), acct(), state="CALM_UP") is None  # $4k, 4%


def test_max_position(gate: RiskGate) -> None:
    v = gate.check(buy(210), acct(), state="CALM_UP", approved=True)  # 21% > 20%
    assert v is not None and v.rule == "max_position"


def test_state_leverage(gate: RiskGate) -> None:
    v = gate.check(buy(30), acct(), state="CRASH")
    assert v is not None and v.rule == "state_leverage"
    v = gate.check(buy(30), acct(), state="STRESS")  # 3% < 25% leverage, < 20% cap
    assert v is None


def test_manual_approval(gate: RiskGate) -> None:
    v = gate.check(buy(60), acct(), state="CALM_UP")  # $6k > $5k
    assert v is not None and v.rule == "needs_approval"
    assert gate.check(buy(60), acct(), state="CALM_UP", approved=True) is None


def test_daily_loss_blocks_entries(gate: RiskGate) -> None:
    v = gate.check(buy(10), acct(equity=97_900.0, peak_equity=100_000.0), state="CALM_UP")
    assert v is not None and v.rule == "daily_loss"
    assert not gate.kill.active()


def test_max_drawdown_fires_kill(gate: RiskGate) -> None:
    a = acct(equity=89_000.0, peak_equity=100_000.0, day_start_equity=89_500.0)
    v = gate.check(buy(10), a, state="CALM_UP")
    assert v is not None and v.rule == "kill_switch"
    assert gate.kill.active()
    assert "max_drawdown" in gate.kill.reason()


def test_kill_survives_restart_and_blocks_entries(tmp_path: Path) -> None:
    KillSwitch(tmp_path / "KILLED").fire("test")
    gate = RiskGate(KillSwitch(tmp_path / "KILLED"))
    v = gate.check(buy(1), acct(), state="CALM_UP")
    assert v is not None and v.rule == "kill_switch"


def test_reducing_orders_always_pass(tmp_path: Path) -> None:
    KillSwitch(tmp_path / "KILLED").fire("test")
    gate = RiskGate(KillSwitch(tmp_path / "KILLED"))
    a = acct(equity=80_000.0, position_qty=300.0)  # killed, deep drawdown, oversized
    assert gate.check(Order(qty=-300.0, price=100.0), a, state="CRASH") is None
    assert gate.check(Order(qty=-100.0, price=100.0), a, state="CRASH") is None


def test_flip_through_zero_is_not_reducing(gate: RiskGate) -> None:
    v = gate.check(Order(qty=-50.0, price=100.0), acct(position_qty=10.0), state="CALM_UP")
    assert v is not None and v.rule == "long_only"


def test_frozen_blocks_entries(gate: RiskGate) -> None:
    v = gate.check(buy(1), acct(), state="CALM_UP", frozen=True)
    assert v is not None and v.rule == "frozen"


def test_overnight_cap(gate: RiskGate) -> None:
    v = gate.check(buy(110), acct(), state="CALM_UP", approved=True, last_bar=True)
    assert v is not None and v.rule == "overnight"


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -5.0])
def test_bad_price_vetoed(gate: RiskGate, bad: float) -> None:
    v = gate.check(Order(qty=1.0, price=bad), acct(), state="CALM_UP")
    assert v is not None and v.rule == "bad_order"


def test_kill_only_cleared_by_reset(tmp_path: Path) -> None:
    k = KillSwitch(tmp_path / "KILLED")
    k.fire("x")
    k.fire("y")  # firing again keeps the first reason
    assert "x" in k.reason()
    k.reset(confirm="I have reviewed the account")
    assert not k.active()
    with pytest.raises(ValueError):
        k.fire("z")
        k.reset(confirm="")
