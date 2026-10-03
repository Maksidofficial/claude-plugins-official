import ast
import dataclasses
from pathlib import Path

import pytest

from regimebot import limits
from regimebot.limits import LIMITS

SRC = Path(limits.__file__).read_text()


def test_values_match_spec() -> None:
    assert LIMITS.max_position_pct == 0.20
    assert LIMITS.daily_loss_pct == 0.02
    assert LIMITS.max_drawdown_pct == 0.10
    assert LIMITS.manual_approval_usd == 5_000.0
    assert LIMITS.max_overnight_pct == 0.10
    assert LIMITS.state_leverage == {
        "CALM_UP": 1.0,
        "CHOP": 0.5,
        "STRESS": 0.25,
        "CRASH": 0.0,
    }


def test_unknown_state_has_zero_leverage() -> None:
    assert LIMITS.leverage_for("SOMETHING_NEW") == 0.0


def test_limits_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        LIMITS.max_position_pct = 1.0  # type: ignore[misc]


def test_state_leverage_is_read_only() -> None:
    with pytest.raises(TypeError):
        LIMITS.state_leverage["CRASH"] = 5.0  # type: ignore[index]


def test_limits_module_reads_nothing_external() -> None:
    tree = ast.parse(SRC)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "eval", "exec", "__import__"}
    allowed = {"__future__", "dataclasses", "types", "typing"}
    assert imported <= allowed, imported - allowed
    for banned in ("environ", "getenv", "toml", "json", "regimebot"):
        assert banned not in SRC
