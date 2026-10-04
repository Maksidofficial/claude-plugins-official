import pytest

from regimebot.decide.switcher import Switcher, SwitchParams, SwitchState

P = SwitchParams(takeover=0.70, hold=3, cooldown=6, early_cut_p=0.25, early_cut=0.5, gap=0.15)
CALM_NEXT = {"CALM_UP": 0.9, "CHOP": 0.1, "STRESS": 0.0, "CRASH": 0.0}


def probs(**kw: float) -> dict[str, float]:
    base = {"CALM_UP": 0.0, "CHOP": 0.0, "STRESS": 0.0, "CRASH": 0.0}
    base.update(kw)
    rest = 1.0 - sum(base.values())
    base["CHOP"] += rest
    return base


def feed(sw: Switcher, seq: list[dict[str, float]], nxt: dict[str, float] = CALM_NEXT):
    return [sw.step(p, nxt) for p in seq]


def test_starts_flat() -> None:
    d = Switcher(P).step(probs(CALM_UP=0.5), CALM_NEXT)
    assert d.active is None and d.size_mult == 0.0


def test_below_threshold_never_switches() -> None:
    out = feed(Switcher(P), [probs(CALM_UP=0.69)] * 20)
    assert all(d.active is None for d in out)


def test_two_candles_not_enough_three_is() -> None:
    sw = Switcher(P)
    out = feed(sw, [probs(CALM_UP=0.71)] * 2)
    assert out[-1].active is None
    d = sw.step(probs(CALM_UP=0.71), CALM_NEXT)
    assert d.active == "CALM_UP" and d.switched
    assert "CALM_UP" in d.reason and d.probs["CALM_UP"] == pytest.approx(0.71)


def test_lead_must_be_consecutive() -> None:
    sw = Switcher(P)
    seq = [probs(CALM_UP=0.8), probs(CALM_UP=0.8), probs(CALM_UP=0.6), probs(CALM_UP=0.8)]
    assert all(d.active is None for d in feed(sw, seq))


def test_cooldown_blocks_next_switch() -> None:
    sw = Switcher(P)
    feed(sw, [probs(CALM_UP=0.9)] * 3)  # switch on candle 3, cooldown 6 starts
    out = feed(sw, [probs(STRESS=0.9)] * 6)
    assert all(d.active == "CALM_UP" for d in out)
    d = sw.step(probs(STRESS=0.9), CALM_NEXT)
    assert d.active == "STRESS" and d.switched


def test_uncertain_sizes_to_zero_but_keeps_state() -> None:
    sw = Switcher(P)
    feed(sw, [probs(CALM_UP=0.9)] * 3)
    d = sw.step({"CALM_UP": 0.45, "CHOP": 0.40, "STRESS": 0.15, "CRASH": 0.0}, CALM_NEXT)
    assert d.uncertain and d.size_mult == 0.0 and d.active == "CALM_UP"


def test_early_cut_on_stress_forecast() -> None:
    sw = Switcher(P)
    feed(sw, [probs(CALM_UP=0.9)] * 3)
    risky = {"CALM_UP": 0.7, "CHOP": 0.04, "STRESS": 0.2, "CRASH": 0.06}
    d = sw.step(probs(CALM_UP=0.9), risky)
    assert d.size_mult == 0.5 and "early_cut" in d.reason
    d = sw.step(probs(CALM_UP=0.9), {"CALM_UP": 0.76, "CHOP": 0.0, "STRESS": 0.2, "CRASH": 0.04})
    assert d.size_mult == 1.0  # 0.24 is not above 0.25


@pytest.mark.parametrize("kind", ["error", "stale"])
def test_error_or_stale_goes_flat(kind: str) -> None:
    sw = Switcher(P)
    feed(sw, [probs(CALM_UP=0.9)] * 3)
    d = sw.fail(kind)
    assert d.flat and d.size_mult == 0.0 and kind in d.reason


def test_failure_resets_challenger() -> None:
    sw = Switcher(P)
    feed(sw, [probs(CALM_UP=0.9)] * 2)
    sw.fail("stale")
    d = sw.step(probs(CALM_UP=0.9), CALM_NEXT)
    assert d.active is None


def test_no_flip_flop_on_oscillation() -> None:
    sw = Switcher(P)
    feed(sw, [probs(CALM_UP=0.9)] * 3)
    seq = [probs(STRESS=0.75) if i % 2 else probs(CALM_UP=0.75) for i in range(60)]
    switches = [d for d in feed(sw, seq) if d.switched]
    assert switches == []


def test_flip_flop_bounded_by_cooldown() -> None:
    sw = Switcher(P)
    seq = []
    for _ in range(10):
        seq += [probs(CALM_UP=0.9)] * 3 + [probs(STRESS=0.9)] * 3
    n = sum(d.switched for d in feed(sw, seq))
    assert n <= len(seq) // (P.cooldown + 1) + 1


def test_state_round_trip() -> None:
    sw = Switcher(P)
    feed(sw, [probs(CALM_UP=0.9)] * 3 + [probs(STRESS=0.9)] * 2)
    clone = Switcher(P, SwitchState.from_dict(sw.state.to_dict()))
    a = feed(sw, [probs(STRESS=0.9)] * 6)
    b = feed(clone, [probs(STRESS=0.9)] * 6)
    assert [(d.active, d.switched) for d in a] == [(d.active, d.switched) for d in b]
