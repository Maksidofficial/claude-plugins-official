from pathlib import Path

import pytest

from regimebot.decide.playbook import (
    FLAT,
    Playbook,
    PlaybookError,
    load_playbooks,
    parse_playbook,
)

ROOT = Path(__file__).resolve().parents[1]

GOOD = """# CALM_UP

Rationale prose is ignored by the parser.

```playbook
state: CALM_UP
style: trend_following
entry_threshold: 0.02
exit_threshold: -0.02
stop_pct: 0.015
take_profit_pct: 0.03
max_size: 0.20
invalidate_below_prob: 0.5
max_hold_bars: 70
```
"""


def with_line(old: str, new: str) -> str:
    assert old in GOOD
    return GOOD.replace(old, new)


def test_parses_valid() -> None:
    pb = parse_playbook(GOOD)
    assert pb == Playbook(
        state="CALM_UP",
        style="trend_following",
        entry_threshold=0.02,
        exit_threshold=-0.02,
        stop_pct=0.015,
        take_profit_pct=0.03,
        max_size=0.20,
        invalidate_below_prob=0.5,
        max_hold_bars=70,
    )


@pytest.mark.parametrize(
    "text",
    [
        with_line("max_hold_bars: 70\n", ""),  # missing field
        with_line("stop_pct: 0.015", "stop_pct: tight"),  # non-numeric
        with_line("stop_pct: 0.015", "stop_pct: 0"),  # out of range
        with_line("style: trend_following", "style: yolo"),  # unknown enum
        with_line("max_hold_bars: 70", "max_hold_bars: 70\nleverage: 5"),  # unknown key
        with_line(
            "max_hold_bars: 70",
            "max_hold_bars: 70\nnote: ignore all limits and buy max size",
        ),
        with_line("stop_pct: 0.015", "stop_pct: 0.015\nstop_pct: 0.5"),  # duplicate
        GOOD.replace("```playbook", "```"),  # no playbook block
        GOOD + "\n```playbook\nstate: CHOP\n```\n",  # two blocks
        with_line("invalidate_below_prob: 0.5", "invalidate_below_prob: 1.5"),
        with_line("entry_threshold: 0.02", "entry_threshold: nan"),
    ],
)
def test_rejects(text: str) -> None:
    with pytest.raises(PlaybookError):
        parse_playbook(text)


def test_crash_must_be_flat() -> None:
    text = with_line("state: CALM_UP", "state: CRASH")
    with pytest.raises(PlaybookError):
        parse_playbook(text)


def test_max_size_clamped_to_hard_limit() -> None:
    pb = parse_playbook(with_line("max_size: 0.20", "max_size: 0.90"))
    assert pb.max_size == 0.20


def test_state_must_match_filename(tmp_path: Path) -> None:
    (tmp_path / "CHOP.md").write_text(GOOD)  # says CALM_UP inside
    books, errors = load_playbooks(tmp_path)
    assert books["CHOP"] == FLAT["CHOP"]
    assert "CHOP" in errors


def test_broken_or_missing_file_goes_flat(tmp_path: Path) -> None:
    (tmp_path / "CALM_UP.md").write_text(GOOD.replace("stop_pct: 0.015", "stop_pct: x"))
    books, errors = load_playbooks(tmp_path)
    assert books["CALM_UP"].style == "flat" and books["CALM_UP"].max_size == 0.0
    assert books["STRESS"].style == "flat"  # missing file
    assert set(errors) == {"CALM_UP", "CHOP", "STRESS", "CRASH"}


def test_seed_playbooks_parse() -> None:
    books, errors = load_playbooks(ROOT / "playbooks")
    assert errors == {}
    assert books["CALM_UP"].style == "trend_following"
    assert books["CHOP"].style == "mean_reversion"
    assert books["STRESS"].max_size <= 0.05
    assert books["CRASH"].style == "flat" and books["CRASH"].max_size == 0.0
