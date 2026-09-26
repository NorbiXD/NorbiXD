"""Deterministic identifiers.

Replays must be bit-for-bit reproducible, so ids are derived from counters and content hashes,
never from uuid4() or wall-clock time.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def content_hash(obj: Any, length: int = 12) -> str:
    """Stable short hash of a JSON-serializable object (sorted keys, no whitespace)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


class Sequence:
    """Monotonic counter producing prefixed ids, e.g. ``D000042``."""

    def __init__(self, prefix: str, width: int = 6, start: int = 0) -> None:
        self.prefix = prefix
        self.width = width
        self._n = start

    def next(self) -> str:
        self._n += 1
        return f"{self.prefix}{self._n:0{self.width}d}"

    @property
    def value(self) -> int:
        return self._n
