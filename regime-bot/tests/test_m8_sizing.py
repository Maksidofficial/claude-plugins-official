import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimebot.decide.sizing import SizeInputs, entropy_factor, target_fraction


def inputs(**kw: object) -> SizeInputs:
    base: dict[str, object] = dict(
        active="CALM_UP",
        probs={"CALM_UP": 1.0, "CHOP": 0.0, "STRESS": 0.0, "CRASH": 0.0},
        kelly=0.4,
        playbook_max=1.0,
        size_mult=1.0,
        calibrated=True,
    )
    base.update(kw)
    return SizeInputs(**base)  # type: ignore[arg-type]


def test_quarter_kelly_when_certain() -> None:
    assert target_fraction(inputs(kelly=0.4)) == pytest.approx(0.10)


def test_hard_cap_binds() -> None:
    assert target_fraction(inputs(kelly=4.0)) == pytest.approx(0.20)  # LIMITS.max_position_pct


def test_state_leverage_cap_binds() -> None:
    p = {"CALM_UP": 0.0, "CHOP": 0.0, "STRESS": 1.0, "CRASH": 0.0}
    got = target_fraction(inputs(active="STRESS", probs=p, kelly=4.0))
    assert got == pytest.approx(0.20)  # min(0.20 cap, 0.25 leverage)


def test_playbook_can_only_tighten() -> None:
    assert target_fraction(inputs(kelly=4.0, playbook_max=0.05)) == pytest.approx(0.05)
    assert target_fraction(inputs(kelly=4.0, playbook_max=9.0)) == pytest.approx(0.20)


def test_scaled_by_probability_and_entropy() -> None:
    p = {"CALM_UP": 0.8, "CHOP": 0.2, "STRESS": 0.0, "CRASH": 0.0}
    h = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2)) / math.log(4)
    assert target_fraction(inputs(probs=p)) == pytest.approx(0.10 * 0.8 * (1 - h))


@pytest.mark.parametrize(
    "kw",
    [
        {"active": "CRASH", "probs": {"CALM_UP": 0, "CHOP": 0, "STRESS": 0, "CRASH": 1.0}},
        {"active": None},
        {"size_mult": 0.0},
        {"calibrated": False},
        {"kelly": -0.3},
        {"kelly": float("nan")},
    ],
)
def test_zero_cases(kw: dict[str, object]) -> None:
    assert target_fraction(inputs(**kw)) == 0.0


def test_early_cut_halves() -> None:
    assert target_fraction(inputs(size_mult=0.5)) == pytest.approx(0.05)


@given(a=st.floats(0.34, 1.0), b=st.floats(0.34, 1.0))
def test_entropy_factor_monotone(a: float, b: float) -> None:
    def pr(top: float) -> dict[str, float]:
        r = (1 - top) / 3
        return {"A": top, "B": r, "C": r, "D": r}

    lo, hi = sorted((a, b))
    assert entropy_factor(pr(lo)) <= entropy_factor(pr(hi)) + 1e-12


def test_entropy_factor_bounds() -> None:
    assert entropy_factor({"A": 1.0, "B": 0.0}) == pytest.approx(1.0)
    assert entropy_factor({"A": 0.5, "B": 0.5}) == pytest.approx(0.0)
