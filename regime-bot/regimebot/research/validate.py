"""The harness grades proposals, never the model that wrote them.

A proposal's playbooks are parsed strictly, written to a scratch copy, and the full
walk-forward is re-run against them. Only a proposal that clears every gate is promoted.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from regimebot.backtest.gates import GateResult
from regimebot.decide.playbook import PlaybookError, parse_playbook
from regimebot.research.model import Proposal

# playbooks dir -> (strategy.md text, gate result)
Backtest = Callable[[Path], tuple[str, GateResult]]


@dataclass(frozen=True)
class ValidationResult:
    promoted: bool
    reason: str


def validate_and_promote(proposal: Proposal, root: Path, backtest: Backtest) -> ValidationResult:
    if not proposal.playbook_changes:
        return ValidationResult(False, "no playbook change to validate")
    for ch in proposal.playbook_changes:
        try:
            pb = parse_playbook(ch.content)
        except PlaybookError as e:
            return ValidationResult(False, f"{ch.state}: invalid playbook: {e}")
        if pb.state != ch.state:
            return ValidationResult(False, f"{ch.state}: file declares {pb.state}")
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "playbooks"
        shutil.copytree(root / "playbooks", scratch)
        for ch in proposal.playbook_changes:
            (scratch / f"{ch.state}.md").write_text(ch.content)
        strategy, gate = backtest(scratch)
        if not gate.passed:
            failed = ", ".join(r.name for r in gate.rows if not r.ok)
            return ValidationResult(False, f"failed gates: {failed}")
        for ch in proposal.playbook_changes:
            (root / "playbooks" / f"{ch.state}.md").write_text(ch.content)
        (root / "strategy.md").write_text(strategy)
    return ValidationResult(True, "cleared every gate; promoted")
