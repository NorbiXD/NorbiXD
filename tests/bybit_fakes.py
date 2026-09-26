"""Fakes for Bybit: a local WebSocket server speaking the V5 protocol, and message builders."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Iterable
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve

from darwin.core.events import (
    BookDelta,
    BookSnapshot,
    Event,
    LiquidationEvent,
    TickerEvent,
    TradeEvent,
)


def to_bybit(ev: Event) -> dict[str, Any] | None:
    """Normalized event -> Bybit V5 public message (inverse of the parser, for fakes)."""
    if isinstance(ev, (BookSnapshot, BookDelta)):
        return {
            "topic": f"orderbook.50.{ev.symbol}",
            "type": "snapshot" if isinstance(ev, BookSnapshot) else "delta",
            "ts": ev.ts,
            "data": {
                "s": ev.symbol,
                "b": [[str(p), str(q)] for p, q in ev.bids],
                "a": [[str(p), str(q)] for p, q in ev.asks],
                "u": ev.update_id,
                "seq": ev.update_id * 10,
            },
            "cts": ev.ts,
        }
    if isinstance(ev, TradeEvent):
        return {
            "topic": f"publicTrade.{ev.symbol}",
            "type": "snapshot",
            "ts": ev.ts,
            "data": [
                {
                    "T": ev.ts,
                    "s": ev.symbol,
                    "S": ev.taker_side.value,
                    "v": str(ev.qty),
                    "p": str(ev.price),
                    "L": "PlusTick",
                    "i": ev.trade_id,
                    "BT": False,
                }
            ],
        }
    if isinstance(ev, TickerEvent):
        data: dict[str, Any] = {"symbol": ev.symbol}
        for k, v in (
            ("lastPrice", ev.last_price),
            ("markPrice", ev.mark_price),
            ("indexPrice", ev.index_price),
            ("fundingRate", ev.funding_rate),
            ("nextFundingTime", ev.next_funding_ts),
            ("openInterest", ev.open_interest),
        ):
            if v is not None:
                data[k] = str(v)
        return {"topic": f"tickers.{ev.symbol}", "type": "snapshot", "data": data, "cs": 1, "ts": ev.ts}
    if isinstance(ev, LiquidationEvent):
        return {
            "topic": f"allLiquidation.{ev.symbol}",
            "type": "snapshot",
            "ts": ev.ts,
            "data": [{"T": ev.ts, "s": ev.symbol, "S": ev.side.value, "v": str(ev.qty), "p": str(ev.price)}],
        }
    return None


class FakeBybitServer:
    """Local V5-ish WebSocket server.

    ``script(conn_index)`` returns the messages to push after subscription on that connection;
    ``close_after`` maps connection index -> number of messages after which the server drops
    the connection (simulating an exchange restart / network cut).
    """

    def __init__(
        self,
        script: Callable[[int], Iterable[dict[str, Any]]],
        close_after: dict[int, int] | None = None,
        interval_s: float = 0.0,
        require_auth: str | None = None,
        silent_after_subscribe: bool = False,
        pace: Callable[[dict[str, Any]], float] | None = None,
    ) -> None:
        self.script = script
        self.close_after = close_after or {}
        self.interval_s = interval_s
        self.require_auth = require_auth
        self.silent = silent_after_subscribe
        self.pace = pace
        self.connections = 0
        self.received: list[dict[str, Any]] = []
        self.server: Server | None = None
        self.port = 0

    async def __aenter__(self) -> FakeBybitServer:
        self.server = await serve(self._handler, "127.0.0.1", 0)
        self.port = next(iter(self.server.sockets)).getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self.server is not None
        self.server.close()
        with contextlib.suppress(Exception):
            await self.server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    async def _handler(self, ws: ServerConnection) -> None:
        self.connections += 1
        idx = self.connections
        subscribed = asyncio.Event()

        async def reader() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                self.received.append({"conn": idx, **msg})
                op = msg.get("op")
                if op == "auth":
                    ok = self.require_auth is None or msg["args"][0] == self.require_auth
                    await ws.send(
                        json.dumps({"op": "auth", "success": ok, "ret_msg": "" if ok else "invalid key"})
                    )
                elif op in ("subscribe", "unsubscribe"):
                    await ws.send(
                        json.dumps({"op": op, "success": True, "ret_msg": "", "req_id": msg.get("req_id")})
                    )
                    subscribed.set()
                elif op == "ping":
                    await ws.send(json.dumps({"op": "pong", "success": True}))

        rtask = asyncio.create_task(reader())
        try:
            await asyncio.wait_for(subscribed.wait(), timeout=5)
            if self.silent:
                await ws.wait_closed()
                return
            for n, m in enumerate(self.script(idx), 1):
                if self.pace is not None:
                    wait = self.pace(m)
                    if wait > 0:
                        await asyncio.sleep(wait)
                await ws.send(json.dumps(m))
                if self.interval_s:
                    await asyncio.sleep(self.interval_s)
                if self.close_after.get(idx) == n:
                    await ws.close()
                    return
            await ws.wait_closed()
        except Exception:
            return
        finally:
            rtask.cancel()
