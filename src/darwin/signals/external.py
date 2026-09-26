"""External signal feeds: licensed webhooks, analyst feeds, communities, other authorized inputs.

All feeds implement :class:`ExternalSignalFeed` and emit ``IntelligenceSignal`` events stamped
with their *receive* time. Nothing here is coupled to a specific platform.

* :class:`WebhookSignalFeed` — push-based. A provider POSTs JSON to ``/api/signals/{source}``
  with header ``X-Darwin-Signature: sha256=<hex HMAC-SHA256(secret, raw body)>``. Unsigned,
  badly signed, malformed, replayed or rate-exceeding requests are rejected.
* :class:`DisabledFeed` — placeholder for connectors whose platform terms do not currently
  permit this AI workflow (e.g. scraping a chat platform). It refuses to start and says why.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from darwin.core.events import IntelligenceSignal

Emit = Callable[[IntelligenceSignal], None]


class ExternalSignalFeed(Protocol):
    name: str

    @property
    def enabled(self) -> bool: ...


class ExternalSignalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=128)
    symbol: str | None = None
    topic: str = "external"
    value: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    half_life_ms: int = Field(default=3_600_000, gt=0, le=7 * 86_400_000)
    observed_ts: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WebhookRejected(Exception):
    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class WebhookSignalFeed:
    def __init__(
        self,
        name: str,
        secret: str,
        emit: Emit,
        clock_ms: Callable[[], int],
        allowed_symbols: tuple[str, ...] = (),
        max_per_minute: int = 120,
        max_body_bytes: int = 32_768,
    ) -> None:
        if len(secret) < 16:
            raise ValueError("webhook secret must be >= 16 characters")
        self.name = name
        self._secret = secret.encode()
        self.emit = emit
        self.clock_ms = clock_ms
        self.allowed_symbols = set(allowed_symbols)
        self.max_per_minute = max_per_minute
        self.max_body_bytes = max_body_bytes
        self._recent: deque[int] = deque()
        self._seen_ids: deque[str] = deque(maxlen=10_000)
        self.accepted = 0
        self.rejected = 0

    @property
    def enabled(self) -> bool:
        return True

    @staticmethod
    def sign(secret: str, body: bytes) -> str:
        return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    def ingest(self, body: bytes, signature: str | None) -> IntelligenceSignal:
        try:
            return self._ingest(body, signature)
        except WebhookRejected:
            self.rejected += 1
            raise

    def _ingest(self, body: bytes, signature: str | None) -> IntelligenceSignal:
        if len(body) > self.max_body_bytes:
            raise WebhookRejected(413, "body too large")
        expected = "sha256=" + hmac.new(self._secret, body, hashlib.sha256).hexdigest()
        if not signature or not hmac.compare_digest(signature, expected):
            raise WebhookRejected(401, "bad signature")
        now = self.clock_ms()
        while self._recent and now - self._recent[0] > 60_000:
            self._recent.popleft()
        if len(self._recent) >= self.max_per_minute:
            raise WebhookRejected(429, "rate limit")
        try:
            p = ExternalSignalPayload.model_validate(json.loads(body))
        except (ValidationError, json.JSONDecodeError) as e:
            raise WebhookRejected(422, f"invalid payload: {e}") from e
        if p.symbol is not None and self.allowed_symbols and p.symbol not in self.allowed_symbols:
            raise WebhookRejected(422, f"symbol {p.symbol} not allowed")
        sid = f"EXT:{self.name}:{p.id or hashlib.sha256(body).hexdigest()[:16]}"
        if sid in self._seen_ids:
            raise WebhookRejected(409, "duplicate signal id")
        self._seen_ids.append(sid)
        self._recent.append(now)
        sig = IntelligenceSignal(
            ts=now,
            signal_id=sid,
            source=f"external:{self.name}",
            symbol=p.symbol,
            topic=p.topic,
            value=p.value,
            confidence=p.confidence,
            half_life_ms=p.half_life_ms,
            observed_ts=p.observed_ts,
            provider=self.name,
            payload=p.payload,
        )
        self.accepted += 1
        self.emit(sig)
        return sig


class DisabledFeed:
    """A connector we deliberately do not run (terms of service / licensing)."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    @property
    def enabled(self) -> bool:
        return False

    def start(self) -> None:
        raise RuntimeError(f"feed {self.name!r} is disabled: {self.reason}")


def webhook_feeds_from_env(
    names: tuple[str, ...], emit: Emit, clock_ms: Callable[[], int], allowed_symbols: tuple[str, ...]
) -> dict[str, WebhookSignalFeed]:
    """Create webhook feeds whose secrets are in ``DARWIN_WEBHOOK_SECRET_<NAME>``; skip missing."""
    feeds = {}
    for n in names:
        secret = os.environ.get(f"DARWIN_WEBHOOK_SECRET_{n.upper()}")
        if secret:
            feeds[n] = WebhookSignalFeed(n, secret, emit, clock_ms, allowed_symbols)
    return feeds
