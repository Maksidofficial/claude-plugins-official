"""Name states by their statistics and keep names stable across refits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from regimebot.data.features import RET_BARS
from regimebot.hmm.fit import RegimeModel, expected_durations

LABELS = ("CALM_UP", "CHOP", "STRESS", "CRASH")
RET, RVOL = 0, 1  # feature columns: ret10, rvol
HIGH_VOL_MULT = 2.5  # any state this much more volatile than the calmest is at least STRESS


@dataclass(frozen=True)
class StateStats:
    mean_ret: float  # mean log return per bar (ret10 mean / 10)
    vol: float  # mean realized one-bar volatility in the state (rvol mean)
    duration: float  # expected bars, 1 / (1 - A_ii)


def raw_means(m: RegimeModel) -> np.ndarray:
    return np.asarray(m.means * m.scaler.std + m.scaler.mean)


def raw_covars(m: RegimeModel) -> np.ndarray:
    s = m.scaler.std
    return np.asarray(m.covars * np.outer(s, s))


def state_stats(m: RegimeModel) -> list[StateStats]:
    mu, dur = raw_means(m), expected_durations(m.transmat)
    return [
        StateStats(float(mu[i, RET]) / RET_BARS, max(float(mu[i, RVOL]), 1e-12), float(dur[i]))
        for i in range(m.k)
    ]


def auto_label(m: RegimeModel) -> list[str]:
    st = state_stats(m)
    by_vol = sorted(range(m.k), key=lambda i: st[i].vol)
    labels = ["CHOP"] * m.k
    if m.k >= 4:
        labels[by_vol[-1]] = "CRASH"
        labels[by_vol[-2]] = "STRESS"
        calm = list(by_vol[:-2])
    else:
        labels[by_vol[-1]] = "STRESS"
        calm = list(by_vol[:-1])
    floor = HIGH_VOL_MULT * st[by_vol[0]].vol
    for i in list(calm):
        if st[i].vol > floor:
            labels[i] = "STRESS"
            calm.remove(i)
    best = max(calm, key=lambda i: st[i].mean_ret)
    if st[best].mean_ret > 0:
        labels[best] = "CALM_UP"
    return labels


def admissible_labels(m: RegimeModel) -> list[set[str]]:
    """Labels each state's statistics allow. High-volatility slots are exact; calm states may
    be CHOP, or CALM_UP when their mean return is positive."""
    st = state_stats(m)
    auto = auto_label(m)
    out: list[set[str]] = []
    for i, lab in enumerate(auto):
        if lab in ("STRESS", "CRASH"):
            out.append({lab})
        else:
            out.append({"CALM_UP", "CHOP"} if st[i].mean_ret > 0 else {"CHOP"})
    return out


def _features(m: RegimeModel) -> np.ndarray:
    return np.array([[s.mean_ret, s.vol] for s in state_stats(m)])


def match_states(old: RegimeModel, new: RegimeModel) -> dict[int, int]:
    """Hungarian match on (mean return, vol), each scaled by its spread across old states.

    Returns {new_index: old_index}. With different K some states stay unmatched.
    """
    fo, fn = _features(old), _features(new)
    scale = fo.std(axis=0) + 1e-12
    cost = np.abs(fn[:, None, :] - fo[None, :, :]) / scale
    rows, cols = linear_sum_assignment(cost.sum(axis=2))
    return {int(r): int(c) for r, c in zip(rows, cols, strict=True)}


def relabel_refit(old: RegimeModel, new: RegimeModel) -> RegimeModel:
    """Matched states inherit the old label; unmatched ones get a fresh statistical label."""
    fresh = auto_label(new)
    mapping = match_states(old, new)
    old_labels = old.labels or auto_label(old)
    labels = [old_labels[mapping[i]] if i in mapping else fresh[i] for i in range(new.k)]
    return new.with_labels(labels)
