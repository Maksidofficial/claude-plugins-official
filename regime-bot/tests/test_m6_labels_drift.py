import numpy as np
import pytest

from regimebot.data.features import RET_BARS, Standardizer
from regimebot.hmm.drift import DriftThresholds, LiveLLMonitor, compare_models
from regimebot.hmm.fit import RegimeModel
from regimebot.hmm.labels import auto_label, match_states, relabel_refit, state_stats

D = 5


def model(stats: list[tuple[float, float]], A: np.ndarray | None = None) -> RegimeModel:
    """stats: per state (mean logret, std logret). Identity scaler, so raw == z."""
    k = len(stats)
    means = np.zeros((k, D))
    covars = np.stack([np.eye(D) for _ in range(k)])
    for i, (mu, sd) in enumerate(stats):
        means[i, 0] = mu * RET_BARS  # ret10 feature
        means[i, 1] = sd  # rvol feature
    if A is None:
        A = np.full((k, k), 0.05 / (k - 1))
        np.fill_diagonal(A, 0.95)
    sc = Standardizer(mean=np.zeros(D), std=np.ones(D))
    return RegimeModel(np.full(k, 1 / k), A, means, covars, sc, 0.0)


def test_state_stats_and_duration() -> None:
    s = state_stats(model([(0.001, 0.004), (-0.002, 0.02)]))
    assert s[0].mean_ret == pytest.approx(0.001)
    assert s[1].vol == pytest.approx(0.02)
    assert s[0].duration == pytest.approx(20.0)


@pytest.mark.parametrize(
    "stats,expected",
    [
        ([(0.001, 0.004), (-0.002, 0.02)], ["CALM_UP", "STRESS"]),
        ([(0.0, 0.008), (0.001, 0.004), (-0.002, 0.02)], ["CHOP", "CALM_UP", "STRESS"]),
        (
            [(-0.01, 0.05), (0.001, 0.004), (-0.001, 0.02), (0.0, 0.008)],
            ["CRASH", "CALM_UP", "STRESS", "CHOP"],
        ),
        (
            [(0.0, 0.007), (0.001, 0.004), (-0.001, 0.02), (0.0, 0.009), (-0.01, 0.05)],
            ["CHOP", "CALM_UP", "STRESS", "CHOP", "CRASH"],
        ),
    ],
)
def test_auto_label(stats: list[tuple[float, float]], expected: list[str]) -> None:
    assert auto_label(model(stats)) == expected


def test_calm_state_with_negative_drift_is_not_calm_up() -> None:
    assert auto_label(model([(-0.0005, 0.004), (-0.002, 0.02)])) == ["CHOP", "STRESS"]


FOUR = [(-0.01, 0.05), (0.001, 0.004), (-0.001, 0.02), (0.0, 0.008)]


def test_hungarian_restores_labels_after_permutation() -> None:
    old = model(FOUR)
    old = old.with_labels(auto_label(old))
    perm = [2, 0, 3, 1]
    new = model([FOUR[i] for i in perm])
    mapping = match_states(old, new)
    assert mapping == {0: 2, 1: 0, 2: 3, 3: 1}
    assert relabel_refit(old, new).labels == [old.labels[i] for i in perm]


def test_identical_refit_has_no_drift() -> None:
    old = model(FOUR).with_labels(auto_label(model(FOUR)))
    rep = compare_models(old, relabel_refit(old, model(FOUR)), DriftThresholds())
    assert rep.alarms == []


def test_mean_shift_alarm() -> None:
    old = model(FOUR).with_labels(auto_label(model(FOUR)))
    moved = list(FOUR)
    moved[1] = (0.001 + 0.15, 0.004)  # ret10 mean moves 1.5 feature std (scaler std is 1)
    rep = compare_models(old, relabel_refit(old, model(moved)), DriftThresholds())
    assert "mean_shift" in rep.alarms


def test_transition_matrix_alarm() -> None:
    old = model(FOUR).with_labels(auto_label(model(FOUR)))
    A = np.full((4, 4), 0.25)
    rep = compare_models(old, relabel_refit(old, model(FOUR, A)), DriftThresholds())
    assert "transmat" in rep.alarms
    assert rep.transmat_l1 == pytest.approx(2 * 0.70)


def test_state_count_change_alarm() -> None:
    old = model(FOUR).with_labels(auto_label(model(FOUR)))
    rep = compare_models(old, relabel_refit(old, model(FOUR[:3])), DriftThresholds())
    assert "k_changed" in rep.alarms


def test_states_that_trade_places_keep_their_labels() -> None:
    old = model(FOUR).with_labels(auto_label(model(FOUR)))
    swapped = list(FOUR)
    swapped[1], swapped[3] = (0.0, 0.0045), (0.001, 0.0075)
    new = relabel_refit(old, model(swapped))
    assert new.labels[3] == "CALM_UP"
    rep = compare_models(old, new, DriftThresholds(mean_shift_sigma=99, transmat_l1=99))
    assert "label_conflict" not in rep.alarms


def test_label_conflict_alarm() -> None:
    old = model(FOUR).with_labels(auto_label(model(FOUR)))
    drifted = list(FOUR)
    drifted[1] = (-0.0008, 0.004)  # still nearest old CALM_UP, but drift is now negative
    new = relabel_refit(old, model(drifted))
    assert new.labels[1] == "CALM_UP"
    rep = compare_models(old, new, DriftThresholds(mean_shift_sigma=99, transmat_l1=99))
    assert "label_conflict" in rep.alarms


def test_small_shift_is_not_drift() -> None:
    old = model(FOUR).with_labels(auto_label(model(FOUR)))
    moved = list(FOUR)
    moved[1] = (0.001 + 0.05, 0.004)  # 0.5 feature std
    rep = compare_models(old, relabel_refit(old, model(moved)), DriftThresholds())
    assert "mean_shift" not in rep.alarms
    assert rep.max_mean_shift_sigma == pytest.approx(0.5)


def test_live_ll_monitor() -> None:
    rng = np.random.default_rng(0)
    in_sample = rng.normal(-5.0, 1.0, 3000)
    mon = LiveLLMonitor.from_in_sample(in_sample, window=35)
    for _ in range(50):  # a long run of ordinary windows never alarms
        for x in rng.normal(-5.0, 1.0, 35):
            mon.add(float(x))
            assert not mon.alarm()
    for x in rng.normal(-8.0, 1.0, 35):
        mon.add(float(x))
    assert mon.alarm()


def test_live_ll_monitor_needs_full_window() -> None:
    mon = LiveLLMonitor.from_in_sample(np.zeros(100) - 5, window=35)
    mon.add(-100.0)
    assert not mon.alarm()


def test_calm_states_with_close_means_are_not_a_conflict() -> None:
    two_up = [(-0.01, 0.05), (0.0010, 0.004), (-0.001, 0.02), (0.0009, 0.008)]
    old = model(two_up).with_labels(auto_label(model(two_up)))
    flipped = list(two_up)
    flipped[3] = (0.0011, 0.008)  # now the "highest mean" calm state, but CHOP is still fine
    new = relabel_refit(old, model(flipped))
    assert new.labels[1] == "CALM_UP" and auto_label(new)[1] == "CHOP"
    rep = compare_models(old, new, DriftThresholds(mean_shift_sigma=99, transmat_l1=99))
    assert "label_conflict" not in rep.alarms


def test_stress_crash_swap_is_a_conflict() -> None:
    old = model(FOUR).with_labels(auto_label(model(FOUR)))
    new = model(FOUR).with_labels(["STRESS", "CALM_UP", "CRASH", "CHOP"])
    rep = compare_models(old, new, DriftThresholds(mean_shift_sigma=99, transmat_l1=99))
    assert "label_conflict" in rep.alarms


def test_admissible_labels() -> None:
    from regimebot.hmm.labels import admissible_labels

    adm = admissible_labels(model(FOUR))
    assert adm == [{"CRASH"}, {"CALM_UP", "CHOP"}, {"STRESS"}, {"CHOP"}]  # s3 mean is 0


def test_any_high_vol_state_is_at_least_stress() -> None:
    five = [(0.0007, 0.0029), (0.0005, 0.0055), (-0.0007, 0.0138), (-0.0008, 0.0127),
            (-0.0004, 0.0116)]
    assert auto_label(model(five)) == ["CALM_UP", "CHOP", "CRASH", "STRESS", "STRESS"]


def test_simulated_review() -> None:
    from regimebot.hmm.drift import simulated_review

    th = DriftThresholds()
    cur = model(FOUR).with_labels(auto_label(model(FOUR)))
    same = relabel_refit(cur, model(FOUR))
    moved = list(FOUR)
    moved[1] = (0.001 + 0.15, 0.004)
    shifted = relabel_refit(cur, model(moved))
    shifted_again = relabel_refit(shifted, model(moved))
    other = list(FOUR)
    other[1] = (0.001 - 0.15, 0.004)
    elsewhere = relabel_refit(cur, model(other))

    assert simulated_review(cur, same, None, th)[0] is True
    assert simulated_review(cur, shifted, None, th)[0] is False
    ok, why = simulated_review(cur, shifted_again, shifted, th)
    assert ok and "persisted" in why
    stale_labels = shifted_again.with_labels(["CALM_UP", "CHOP", "STRESS", "CRASH"])  # inadmissible
    assert simulated_review(cur, stale_labels, shifted, th)[0] is True
    assert simulated_review(cur, elsewhere, shifted, th)[0] is False
