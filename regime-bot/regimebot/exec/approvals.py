"""Manual approval for orders above LIMITS.manual_approval_usd.

The engine vetoes such an order and the live loop files a request here. A human grants it
(CLI or Telegram); the grant lets the same regime's entry through for ``ttl`` and then
expires. Nothing in the code path grants approvals.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TTL = timedelta(hours=2)


class ApprovalQueue:
    def __init__(self, path: Path, ttl: timedelta = TTL) -> None:
        self.path, self.ttl = path, ttl

    def _read(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text())
            return list(data) if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _write(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1))
        os.replace(tmp, self.path)

    def request(self, qty: float, price: float, state: str | None, now: datetime) -> str:
        items = self._read()
        for it in items:
            if (it["status"] == "pending" and it["state"] == state
                    and (it["qty"] > 0) == (qty > 0)):
                return str(it["id"])
        rid = f"{now.strftime('%Y%m%d%H%M')}-{len(items)}"
        items.append({"id": rid, "qty": qty, "price": price, "state": state,
                      "notional": abs(qty) * price, "requested_at": now.isoformat(),
                      "status": "pending", "granted_at": None})
        self._write(items)
        return rid

    def grant(self, rid: str, now: datetime) -> None:
        items = self._read()
        for it in items:
            if it["id"] == rid and it["status"] == "pending":
                it["status"], it["granted_at"] = "granted", now.isoformat()
                self._write(items)
                return
        raise KeyError(f"no pending approval {rid}")

    def granted(self, state: str | None, now: datetime) -> bool:
        for it in self._read():
            if it["status"] == "granted" and it["state"] == state and it["granted_at"]:
                t = datetime.fromisoformat(it["granted_at"])
                if t <= now <= t + self.ttl:
                    return True
        return False

    def pending(self) -> list[dict[str, Any]]:
        return [it for it in self._read() if it["status"] == "pending"]
