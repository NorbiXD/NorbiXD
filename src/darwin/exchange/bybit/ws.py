"""Reconnecting Bybit V5 WebSocket client.

Handles: (re)connect with exponential backoff + jitter, authentication for private streams,
chunked subscription, application-level ping (Bybit closes idle connections), stale-feed
detection (no traffic -> force reconnect), and targeted resubscription (used to obtain a fresh
order-book snapshot after a sequence gap). All callbacks run on the event loop thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from darwin.exchange.bybit.signing import Credentials, ws_auth_args

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BybitEndpoints:
    public_linear_ws: str
    private_ws: str
    rest: str


MAINNET = BybitEndpoints(
    "wss://stream.bybit.com/v5/public/linear", "wss://stream.bybit.com/v5/private", "https://api.bybit.com"
)
TESTNET = BybitEndpoints(
    "wss://stream-testnet.bybit.com/v5/public/linear",
    "wss://stream-testnet.bybit.com/v5/private",
    "https://api-testnet.bybit.com",
)


def now_ms() -> int:
    return int(time.time() * 1000)


MessageHandler = Callable[[dict[str, Any], int], None]
StatusHandler = Callable[[str, str], None]


@dataclass
class WsStats:
    connects: int = 0
    disconnects: int = 0
    messages: int = 0
    resyncs: int = 0
    last_message_ms: int = 0
    errors: list[str] = field(default_factory=list)


class AuthError(RuntimeError):
    pass


class ReconnectingWebSocket:
    def __init__(
        self,
        name: str,
        url: str,
        topics: list[str],
        on_message: MessageHandler,
        on_status: StatusHandler,
        credentials: Credentials | None = None,
        ping_interval_s: float = 20.0,
        stale_after_s: float = 30.0,
        backoff_initial_s: float = 0.5,
        backoff_max_s: float = 30.0,
        max_args_per_subscribe: int = 10,
        clock_ms: Callable[[], int] = now_ms,
    ) -> None:
        self.name = name
        self.url = url
        self.topics = list(topics)
        self.on_message = on_message
        self.on_status = on_status
        self.credentials = credentials
        self.ping_interval_s = ping_interval_s
        self.stale_after_s = stale_after_s
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self.max_args = max_args_per_subscribe
        self.clock_ms = clock_ms
        self.stats = WsStats()
        self._resync: set[str] = set()
        self._ws: Any = None
        self._req_id = 0

    # ------------------------------------------------------------------ control
    def request_resync(self, topic: str) -> None:
        """Unsubscribe + resubscribe ``topic`` on the live connection (fresh snapshot)."""
        self._resync.add(topic)

    async def _send(self, ws: Any, payload: dict[str, Any]) -> None:
        self._req_id += 1
        payload.setdefault("req_id", f"{self.name}-{self._req_id}")
        await ws.send(json.dumps(payload))

    async def _subscribe(self, ws: Any, topics: list[str], op: str = "subscribe") -> None:
        for i in range(0, len(topics), self.max_args):
            await self._send(ws, {"op": op, "args": topics[i : i + self.max_args]})

    async def _authenticate(self, ws: Any) -> None:
        assert self.credentials is not None
        expires = self.clock_ms() + 10_000
        await self._send(ws, {"op": "auth", "args": ws_auth_args(self.credentials, expires)})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(deadline - time.monotonic(), 0.1))
            msg = json.loads(raw)
            if msg.get("op") == "auth":
                if msg.get("success"):
                    return
                raise AuthError(str(msg.get("ret_msg")))
        raise AuthError("auth timeout")

    async def _pinger(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(self.ping_interval_s)
            await self._send(ws, {"op": "ping"})

    # ------------------------------------------------------------------ main loop
    async def run(self, stop: asyncio.Event) -> None:
        backoff = self.backoff_initial_s
        while not stop.is_set():
            ping_task: asyncio.Task[None] | None = None
            try:
                async with websockets.connect(
                    self.url, open_timeout=10, ping_interval=None, max_size=2**23
                ) as ws:
                    self._ws = ws
                    if self.credentials is not None:
                        await self._authenticate(ws)
                    await self._subscribe(ws, self.topics)
                    self.stats.connects += 1
                    self.on_status("connected", self.url)
                    backoff = self.backoff_initial_s
                    ping_task = asyncio.create_task(self._pinger(ws))
                    last_rx = time.monotonic()
                    while not stop.is_set():
                        if self._resync:
                            topics = sorted(self._resync)
                            self._resync.clear()
                            self.stats.resyncs += 1
                            await self._subscribe(ws, topics, "unsubscribe")
                            await self._subscribe(ws, topics, "subscribe")
                            self.on_status("resync", ",".join(topics))
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except TimeoutError:
                            if time.monotonic() - last_rx > self.stale_after_s:
                                self.on_status("stale", f"no traffic for {self.stale_after_s}s")
                                break
                            continue
                        last_rx = time.monotonic()
                        self._handle_raw(raw)
            except AuthError as e:
                self.stats.errors.append(f"auth: {e}")
                self.on_status("auth_failed", str(e))
                raise
            except (OSError, ConnectionClosed, InvalidHandshake, TimeoutError, json.JSONDecodeError) as e:
                self.stats.errors.append(repr(e))
                log.warning("%s websocket error: %r", self.name, e)
            finally:
                self._ws = None
                if ping_task is not None:
                    ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await ping_task
            if stop.is_set():
                break
            self.stats.disconnects += 1
            self.on_status("disconnected", self.url)
            delay = backoff * (0.5 + random.random())
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=delay)
            backoff = min(backoff * 2, self.backoff_max_s)

    def _handle_raw(self, raw: str | bytes) -> None:
        msg = json.loads(raw)
        if not isinstance(msg, dict):
            return
        op = msg.get("op")
        if op in ("pong", "ping") or msg.get("ret_msg") == "pong":
            return
        if op in ("subscribe", "unsubscribe"):
            if not msg.get("success", True):
                self.stats.errors.append(f"{op}: {msg.get('ret_msg')}")
                self.on_status("error", f"{op} failed: {msg.get('ret_msg')}")
            return
        if "topic" in msg:
            self.stats.messages += 1
            recv = self.clock_ms()
            self.stats.last_message_ms = recv
            try:
                self.on_message(msg, recv)
            except Exception as e:  # a bad message must not kill the feed
                log.exception("%s: failed to handle message", self.name)
                self.stats.errors.append(f"handler: {e!r}")
