"""Run every minute by a systemd timer. A live loop that stops writing its heartbeat (hung,
not crashed: systemd restarts crashes) trips the kill switch."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from regimebot.decide.risk import KillSwitch

MAX_AGE = timedelta(minutes=3)


def check_heartbeat(state_dir: Path, now: datetime, alert: Callable[[str], None],
                    max_age: timedelta = MAX_AGE) -> bool:
    try:
        beat = datetime.fromisoformat((state_dir / "heartbeat").read_text().strip())
        age = now - beat
    except (OSError, ValueError):
        age = None
    if age is not None and age <= max_age:
        return True
    reason = f"watchdog: heartbeat {'missing' if age is None else f'{age} old'}"
    KillSwitch(state_dir / "KILLED").fire(reason)
    alert(f"KILL SWITCH: {reason}")
    return False
