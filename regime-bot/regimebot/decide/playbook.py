"""Parse playbooks/<STATE>.md into typed rules. Strict: anything unexpected is rejected.

Only the single fenced ``playbook`` block is read; surrounding prose is ignored. Values are
numbers in range or members of a fixed enum, so playbook text can never act as an
instruction. A rejected or missing playbook makes that state FLAT.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from regimebot.limits import LIMITS

STATES = ("CALM_UP", "CHOP", "STRESS", "CRASH")
STYLES = ("trend_following", "mean_reversion", "flat")


class PlaybookError(ValueError):
    pass


@dataclass(frozen=True)
class Playbook:
    state: str
    style: str
    entry_threshold: float  # trend_following: trend > x. mean_reversion: stretch < -x
    exit_threshold: float  # trend_following: trend < x. mean_reversion: stretch > x
    stop_pct: float
    take_profit_pct: float
    max_size: float  # fraction of equity, clamped to the hard limit
    invalidate_below_prob: float  # exit when P(state) falls below this
    max_hold_bars: int


# key -> (kind, lo, hi); bounds inclusive
_SCHEMA: dict[str, tuple[str, float, float]] = {
    "state": ("state", 0, 0),
    "style": ("style", 0, 0),
    "entry_threshold": ("float", -10.0, 10.0),
    "exit_threshold": ("float", -10.0, 10.0),
    "stop_pct": ("float", 0.001, 0.2),
    "take_profit_pct": ("float", 0.001, 0.5),
    "max_size": ("float", 0.0, 10.0),
    "invalidate_below_prob": ("float", 0.0, 1.0),
    "max_hold_bars": ("int", 1, 10_000),
}

_BLOCK = re.compile(r"^```playbook[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
_LINE = re.compile(r"^([a-z_]+):[ \t]*(\S+)[ \t]*$")


def parse_playbook(text: str) -> Playbook:
    blocks = _BLOCK.findall(text)
    if len(blocks) != 1:
        raise PlaybookError(f"expected exactly one ```playbook block, found {len(blocks)}")
    vals: dict[str, object] = {}
    for raw in blocks[0].splitlines():
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        if not m:
            raise PlaybookError(f"malformed line: {raw[:60]!r}")
        key, v = m.groups()
        if key not in _SCHEMA:
            raise PlaybookError(f"unknown key {key!r}")
        if key in vals:
            raise PlaybookError(f"duplicate key {key!r}")
        kind, lo, hi = _SCHEMA[key]
        if kind == "state":
            if v not in STATES:
                raise PlaybookError(f"unknown state {v!r}")
            vals[key] = v
        elif kind == "style":
            if v not in STYLES:
                raise PlaybookError(f"unknown style {v!r}")
            vals[key] = v
        else:
            try:
                num = int(v) if kind == "int" else float(v)
            except ValueError as e:
                raise PlaybookError(f"{key}: not a number: {v!r}") from e
            if not (math.isfinite(num) and lo <= num <= hi):
                raise PlaybookError(f"{key}={v} outside [{lo}, {hi}]")
            vals[key] = num
    missing = set(_SCHEMA) - set(vals)
    if missing:
        raise PlaybookError(f"missing keys: {sorted(missing)}")
    pb = Playbook(**vals)  # type: ignore[arg-type]
    if pb.state == "CRASH" and (pb.style != "flat" or pb.max_size != 0):
        raise PlaybookError("CRASH playbook must be flat with max_size 0")
    return replace(pb, max_size=min(pb.max_size, LIMITS.max_position_pct))


def _flat(state: str) -> Playbook:
    return Playbook(state, "flat", 0.0, 0.0, 0.01, 0.01, 0.0, 1.0, 1)


FLAT: dict[str, Playbook] = {s: _flat(s) for s in STATES}


def load_playbooks(folder: Path) -> tuple[dict[str, Playbook], dict[str, str]]:
    books, errors = dict(FLAT), {}
    for s in STATES:
        path = folder / f"{s}.md"
        try:
            pb = parse_playbook(path.read_text())
            if pb.state != s:
                raise PlaybookError(f"file {path.name} declares state {pb.state}")
            books[s] = pb
        except (OSError, PlaybookError) as e:
            errors[s] = str(e)
    return books, errors
