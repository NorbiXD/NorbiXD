"""Fakes for Bybit: a local WebSocket server speaking the V5 protocol, message builders, and a
stateful fake exchange (REST handler + private-stream pushes) for end-to-end testnet-mode tests."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from collections.abc import Callable, Iterable
from typing import Any

import httpx
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
        live: asyncio.Queue[dict[str, Any]] | None = None,
    ) -> None:
        self.script = script
        self.close_after = close_after or {}
        self.interval_s = interval_s
        self.require_auth = require_auth
        self.silent = silent_after_subscribe
        self.pace = pace
        self.live = live  # pushed after the script: private-stream messages created on the fly
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
            if self.live is not None:
                closed = asyncio.ensure_future(ws.wait_closed())
                while not closed.done():
                    get = asyncio.ensure_future(self.live.get())
                    done, _ = await asyncio.wait({get, closed}, return_when=asyncio.FIRST_COMPLETED)
                    if get in done:
                        await ws.send(json.dumps(get.result()))
                    else:
                        get.cancel()
            await ws.wait_closed()
        except Exception:
            return
        finally:
            rtask.cancel()


class FakeBybitExchange:
    """Stateful V5 fake: a REST handler (for ``httpx.MockTransport``) whose order creation fills
    marketable orders at their limit price and pushes ``order`` / ``execution`` / ``position``
    messages onto the private stream, like the real venue (ACK over REST, fills over WS)."""

    FEE = 0.00055

    def __init__(self, equity: float = 1_000.0) -> None:
        self.private: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.cash = equity
        self.positions: dict[str, list[float]] = {}  # symbol -> [qty, avg]
        self.orders: dict[str, dict[str, Any]] = {}
        self.execs: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.mark: Callable[[str], float | None] = lambda _s: None
        #: when this returns True, order creation fails with a retryable error (rate limit)
        self.reject_create: Callable[[], bool] = lambda: False
        self.rejected_creates = 0
        self._ids = itertools.count(1)

    # ---- accounting
    def _apply(self, symbol: str, side: str, qty: float, px: float) -> None:
        sgn = 1.0 if side == "Buy" else -1.0
        pos = self.positions.setdefault(symbol, [0.0, 0.0])
        q, avg = pos
        dq = sgn * qty
        if q == 0 or (q > 0) == (dq > 0):
            pos[1] = (abs(q) * avg + qty * px) / (abs(q) + qty)
        else:
            closed = min(abs(q), qty)
            self.cash += closed * (px - avg) * (1 if q > 0 else -1)
            if qty > abs(q):
                pos[1] = px
        pos[0] = round(q + dq, 10)
        if pos[0] == 0:
            pos[1] = 0.0
        self.cash -= qty * px * self.FEE

    def equity(self) -> float:
        eq = self.cash
        for sym, (q, avg) in self.positions.items():
            m = self.mark(sym) or avg
            eq += q * (m - avg)
        return eq

    # ---- REST
    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        body = json.loads(request.content) if request.content else {}
        args = {**params, **body}
        self.calls.append((path, args))
        ok: dict[str, Any] = {"retCode": 0, "retMsg": "OK", "result": {}}
        if path == "/v5/account/wallet-balance":
            eq = f"{self.equity():.6f}"
            ok["result"] = {"list": [{"totalEquity": eq, "totalWalletBalance": f"{self.cash:.6f}"}]}
        elif path == "/v5/account/info":
            ok["result"] = {"marginMode": "REGULAR_MARGIN"}
        elif path in ("/v5/position/switch-mode", "/v5/position/set-leverage"):
            pass
        elif path == "/v5/position/list":
            ok["result"] = {"list": [self._position_row(s) for s in self.positions]}
        elif path == "/v5/order/create":
            if self.reject_create():
                self.rejected_creates += 1
                return httpx.Response(
                    200, json={"retCode": 10006, "retMsg": "Too many visits!", "result": {}}
                )
            ok["result"] = self._create(args)
        elif path == "/v5/order/cancel":
            o = self.orders.get(args["orderLinkId"])
            if o is None or o["orderStatus"] != "New":
                return httpx.Response(
                    200, json={"retCode": 110001, "retMsg": "order not exists", "result": {}}
                )
            o["orderStatus"] = "Cancelled"
            self.private.put_nowait({"topic": "order", "data": [dict(o)]})
            ok["result"] = {"orderId": o["orderId"], "orderLinkId": o["orderLinkId"]}
        elif path == "/v5/order/realtime":
            link = args.get("orderLinkId")
            rows = [o for o in self.orders.values() if o["orderStatus"] == "New"]
            ok["result"] = {"list": [o for o in rows if link is None or o["orderLinkId"] == link]}
        elif path == "/v5/order/history":
            o = self.orders.get(args.get("orderLinkId", ""))
            ok["result"] = {"list": [o] if o else []}
        elif path == "/v5/execution/list":
            ok["result"] = {"list": self.execs.get(args.get("orderLinkId", ""), [])}
        elif path in ("/v5/market/kline", "/v5/market/tickers"):
            ok["result"] = {"list": []}  # no history: the run starts cold
        else:
            return httpx.Response(404)
        return httpx.Response(200, json=ok)

    def _position_row(self, symbol: str) -> dict[str, Any]:
        q, avg = self.positions[symbol]
        side = "Buy" if q > 0 else "Sell" if q < 0 else ""
        return {"symbol": symbol, "side": side, "size": f"{abs(q):.10g}", "entryPrice": f"{avg:.10g}"}

    def _create(self, a: dict[str, Any]) -> dict[str, Any]:
        oid = f"X{next(self._ids)}"
        qty, px = float(a["qty"]), float(a["price"])
        o = {
            "category": "linear",
            "symbol": a["symbol"],
            "orderId": oid,
            "orderLinkId": a["orderLinkId"],
            "side": a["side"],
            "orderStatus": "New",
            "cumExecQty": "0",
            "avgPrice": "0",
            "rejectReason": "EC_NoError",
            "updatedTime": "0",
        }
        self.orders[a["orderLinkId"]] = o
        if a.get("timeInForce") == "PostOnly":
            self.private.put_nowait({"topic": "order", "data": [dict(o)]})  # rests until cancelled
            return {"orderId": oid, "orderLinkId": a["orderLinkId"]}
        if a.get("reduceOnly"):  # Bybit: never opens or flips; capped at the position size
            q = self.positions.get(a["symbol"], [0.0, 0.0])[0]
            if q == 0 or (q > 0) == (a["side"] == "Buy"):
                o.update(orderStatus="Rejected", rejectReason="EC_ReduceOnlyNotAllowed")
                self.private.put_nowait({"topic": "order", "data": [dict(o)]})
                return {"orderId": oid, "orderLinkId": a["orderLinkId"]}
            qty = min(qty, abs(q))
            a = {**a, "qty": f"{qty:.10g}"}
        self._apply(a["symbol"], a["side"], qty, px)
        o.update(orderStatus="Filled", cumExecQty=a["qty"], avgPrice=a["price"])
        ex = {
            "category": "linear",
            "symbol": a["symbol"],
            "execFee": f"{qty * px * self.FEE:.8f}",
            "execId": f"E{oid}",
            "execPrice": a["price"],
            "execQty": a["qty"],
            "execType": "Trade",
            "isMaker": False,
            "orderId": oid,
            "orderLinkId": a["orderLinkId"],
            "side": a["side"],
            "execTime": "0",
        }
        self.execs[a["orderLinkId"]] = [ex]
        # like the real venue, the order and execution streams are not mutually ordered
        self.private.put_nowait({"topic": "execution", "data": [ex]})
        self.private.put_nowait({"topic": "order", "data": [dict(o)]})
        self.private.put_nowait({"topic": "position", "data": [self._position_row(a["symbol"])]})
        return {"orderId": oid, "orderLinkId": a["orderLinkId"]}
