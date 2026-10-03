"""Scrub secret values from any text before it is logged or sent anywhere."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

_SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD")


def _secret_values(env: Mapping[str, str]) -> list[str]:
    vals = [v for k, v in env.items() if v and any(m in k.upper() for m in _SECRET_MARKERS)]
    return sorted(vals, key=len, reverse=True)


def redact(text: str, env: Mapping[str, str] | None = None) -> str:
    for v in _secret_values(os.environ if env is None else env):
        text = text.replace(v, "***")
    return text


class RedactingFilter(logging.Filter):
    """Attach to every handler so no record can carry a secret."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True
