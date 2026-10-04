"""Go-live preflight (SPEC §12). Every check must pass; there is no override flag.

Live money is not wired anywhere in this codebase. This gate exists so that adding it later
starts from evidence, not hope.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]

CHECKS = (
    "strategy_accepted",
    "paper_matches_backtest",
    "filtered_probabilities_only",
    "kill_switch_fired_in_testing",
    "no_limit_delegated_to_model",
    "refit_labels_stable",
    "breaking_markets_documented",
)

MIN_PAPER_DAYS = 30
MIN_PAPER_TRADES = 30


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _json(p: Path) -> dict[str, Any]:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def pytest_selector(selector: str) -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-k", selector],
                       cwd=CODE_ROOT, capture_output=True, text=True)
    return r.returncode == 0


def run_preflight(root: Path, state: Path,
                  run_tests: Callable[[str], bool] = pytest_selector) -> tuple[bool, list[Check]]:
    rows: list[Check] = []

    text = (root / "strategy.md").read_text() if (root / "strategy.md").exists() else ""
    ok = "Status: ACCEPTED" in text and "NO STRATEGY ACCEPTED" not in text
    rows.append(Check("strategy_accepted", ok, "strategy.md status"))

    bt, paper = _json(state / "backtest_summary.json"), _json(state / "paper_summary.json")
    bsh = float(bt.get("metrics", {}).get("sharpe", 0.0))
    bhit = float(bt.get("metrics", {}).get("hit_rate", 0.0))
    days, trades = int(paper.get("days", 0)), int(paper.get("trades", 0))
    psh, phit = float(paper.get("sharpe", 0.0)), float(paper.get("hit_rate", 0.0))
    ok = (days >= MIN_PAPER_DAYS and trades >= MIN_PAPER_TRADES and bsh > 0
          and psh >= 0.5 * bsh and abs(phit - bhit) <= 0.10)
    rows.append(Check("paper_matches_backtest", ok,
                      f"paper {days}d {trades} trades sharpe {psh:.2f} hit {phit:.2f} vs "
                      f"backtest sharpe {bsh:.2f} hit {bhit:.2f}"))

    ok = run_tests("lookahead or smoothing_and_viterbi")
    rows.append(Check("filtered_probabilities_only", ok, "no-lookahead and AST tests"))

    drill = _json(state / "drills" / "kill.json")
    ok = all(drill.get(k) for k in ("fired", "flattened", "entries_blocked", "reset"))
    ok = ok and run_tests("kill")
    rows.append(Check("kill_switch_fired_in_testing", ok, "kill drill record + kill tests"))

    ok = run_tests("limits_module_reads_nothing or trading_path_never_imports_research "
                   "or playbook_can_only_tighten or max_size_clamped")
    rows.append(Check("no_limit_delegated_to_model", ok, "limits are code; research isolated"))

    swaps = bt.get("swaps", [])
    acc = sum(1 for s in swaps if s.get("accepted")) / len(swaps) if swaps else 0.0
    last_ok = bool(swaps) and swaps[-1].get("accepted") and \
        "label_conflict" not in swaps[-1].get("alarms", [])
    ok = acc >= 0.8 and bool(last_ok)
    rows.append(Check("refit_labels_stable", ok, f"{acc:.0%} refits accepted"))

    doc = CODE_ROOT / "docs" / "GO_LIVE.md"
    ok = doc.exists() and "WHAT COULD BLOW UP THIS ACCOUNT?" in doc.read_text()
    rows.append(Check("breaking_markets_documented", ok, str(doc.name)))

    return all(r.ok for r in rows), rows
