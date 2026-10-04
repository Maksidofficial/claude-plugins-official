"""Alerts: always to a local JSONL log, and to Telegram when a bot token is configured.

Secrets are redacted before anything is written or sent. A failing alert never raises:
trading must not depend on a chat service.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from regimebot.ops.redact import redact

Post = Callable[[str, dict[str, Any]], None]


def _http_post(url: str, payload: dict[str, Any]) -> None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10):  # noqa: S310 - fixed https host
        pass


class Alerter:
    def __init__(self, log_path: Path, token: str | None, chat_id: str | None,
                 post: Post | None = None) -> None:
        self.log_path, self.token, self.chat_id = log_path, token, chat_id
        self.post = post or _http_post

    def _clean(self, text: str) -> str:
        text = redact(text)
        return text.replace(self.token, "***") if self.token else text

    def _log(self, kind: str, msg: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"at": datetime.now(UTC).isoformat(), "kind": kind, "msg": self._clean(msg)}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def send(self, kind: str, msg: str) -> None:
        self._log(kind, msg)
        if not (self.token and self.chat_id):
            return
        try:
            self.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                      {"chat_id": self.chat_id, "text": self._clean(f"[{kind}] {msg}")})
        except Exception as e:  # never let alerting break trading
            self._log("alert_error", f"telegram send failed: {e}")
