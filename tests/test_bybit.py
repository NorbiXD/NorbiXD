from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from darwin.config.challenge import InstrumentSpec
from darwin.core.events import (
    BookDelta,
    BookSnapshot,
    Event,
    FillEvent,
    FundingPayment,
    LiquidationEvent,
    OrderUpdate,
    PositionSnapshot,
    TickerEvent,
    TradeEvent,
    WalletSnapshot,
)
from darwin.core.types import OrderStatus, OrderType, Side, TimeInForce
from darwin.exchange.bybit.gateway import BybitExecutionGateway, format_step
from darwin.exchange.bybit.parser import parse_private, parse_public
from darwin.exchange.bybit.rest import BybitRest
from darwin.exchange.bybit.signing import Credentials, rest_headers, rest_signature, ws_auth_args
from darwin.exchange.bybit.ws import AuthError, ReconnectingWebSocket
from darwin.execution.orders import OrderRequest
from tests.bybit_fakes import FakeBybitServer

# --------------------------------------------------------------------------- parser (V5 doc formats)

OB_SNAPSHOT = {
    "topic": "orderbook.50.BTCUSDT",
    "type": "snapshot",
    "ts": 1672304484978,
    "data": {
        "s": "BTCUSDT",
        "b": [["16493.50", "0.006"], ["16493.00", "0.100"]],
        "a": [["16611.00", "0.029"], ["16612.00", "0.213"]],
        "u": 18521288,
        "seq": 7961638724,
    },
    "cts": 1672304484976,
}
OB_DELTA = {
    "topic": "orderbook.50.BTCUSDT",
    "type": "delta",
    "ts": 1687940967466,
    "data": {
        "s": "BTCUSDT",
        "b": [["16493.50", "0"]],
        "a": [["16611.00", "0.5"]],
        "u": 18521289,
        "seq": 7961638725,
    },
    "cts": 1687940967464,
}
TRADES = {
    "topic": "publicTrade.BTCUSDT",
    "type": "snapshot",
    "ts": 1672304486868,
    "data": [
        {
            "T": 1672304486865,
            "s": "BTCUSDT",
            "S": "Buy",
            "v": "0.001",
            "p": "16578.50",
            "L": "PlusTick",
            "i": "20f43950-d8dd-5b31-9112-a178eb6023af",
            "BT": False,
        }
    ],
}
TICKER_DELTA = {
    "topic": "tickers.BTCUSDT",
    "type": "delta",
    "data": {"symbol": "BTCUSDT", "markPrice": "17217.33", "fundingRate": "-0.000144"},
    "cs": 24987956059,
    "ts": 1673272861686,
}
LIQ = {
    "topic": "allLiquidation.ROSEUSDT",
    "type": "snapshot",
    "ts": 1739502303204,
    "data": [{"T": 1739502302929, "s": "ROSEUSDT", "S": "Sell", "v": "20000", "p": "0.04499"}],
}


def test_parse_orderbook_trades_ticker_liquidation() -> None:
    [snap] = parse_public(OB_SNAPSHOT, 1000)
    assert isinstance(snap, BookSnapshot) and snap.update_id == 18521288 and snap.bids[0] == (16493.5, 0.006)
    assert snap.ts == 1000 and snap.exch_ts == 1672304484978  # engine clock = receive time
    [delta] = parse_public(OB_DELTA, 1001)
    assert isinstance(delta, BookDelta) and delta.bids == ((16493.5, 0.0),)
    [tr] = parse_public(TRADES, 1002)
    assert isinstance(tr, TradeEvent) and tr.taker_side is Side.BUY and tr.trade_id.startswith("20f4")
    [tk] = parse_public(TICKER_DELTA, 1003)
    assert (
        isinstance(tk, TickerEvent) and tk.mark_price == 17217.33 and tk.last_price is None
    )  # delta: unchanged
    [lq] = parse_public(LIQ, 1004)
    assert isinstance(lq, LiquidationEvent) and lq.side is Side.SELL and lq.qty == 20000


def test_parse_private_streams() -> None:
    order = {
        "topic": "order",
        "data": [
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "orderId": "o1",
                "orderLinkId": "C-A0001-0000001",
                "side": "Buy",
                "orderStatus": "PartiallyFilledCanceled",
                "cumExecQty": "0.004",
                "avgPrice": "30000.5",
                "rejectReason": "EC_NoError",
                "updatedTime": "1672364262457",
            }
        ],
    }
    [ou] = parse_private(order, 5, "challenge")
    assert isinstance(ou, OrderUpdate) and ou.status is OrderStatus.CANCELED and ou.cum_qty == 0.004
    assert ou.client_order_id == "C-A0001-0000001" and ou.reason == "ioc_remainder_cancelled"
    execution = {
        "topic": "execution",
        "data": [
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "execFee": "0.0066",
                "execId": "e1",
                "execPrice": "30000",
                "execQty": "0.004",
                "execType": "Trade",
                "isMaker": False,
                "orderId": "o1",
                "orderLinkId": "C-A0001-0000001",
                "side": "Buy",
                "execTime": "1672364174443",
            },
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "execFee": "0.03",
                "execId": "e2",
                "execPrice": "30000",
                "execQty": "0.01",
                "execType": "Funding",
                "side": "Buy",
                "feeRate": "0.0001",
                "orderId": "",
            },
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "execFee": "0",
                "execId": "e3",
                "execPrice": "29000",
                "execQty": "0.01",
                "execType": "BustTrade",
                "side": "Sell",
                "orderId": "liq",
                "orderLinkId": "",
            },
        ],
    }
    evs = parse_private(execution, 6, "challenge")
    assert isinstance(evs[0], FillEvent) and evs[0].exec_id == "e1" and not evs[0].is_liquidation
    assert isinstance(evs[1], FundingPayment) and evs[1].amount == 0.03
    assert isinstance(evs[2], FillEvent) and evs[2].is_liquidation
    pos = {
        "topic": "position",
        "data": [
            {"category": "linear", "symbol": "ETHUSDT", "side": "Sell", "size": "0.5", "entryPrice": "2000"}
        ],
    }
    [ps] = parse_private(pos, 7, "challenge")
    assert isinstance(ps, PositionSnapshot) and ps.qty == -0.5
    wallet = {
        "topic": "wallet",
        "data": [
            {
                "accountType": "UNIFIED",
                "totalEquity": "201.5",
                "totalWalletBalance": "200",
                "totalAvailableBalance": "150",
            }
        ],
    }
    [w] = parse_private(wallet, 8, "challenge")
    assert isinstance(w, WalletSnapshot) and w.equity == 201.5


# --------------------------------------------------------------------------- signing


def test_rest_and_ws_signatures() -> None:
    c = Credentials("XXXXXXXXXX", "YYYYYYYYYY")
    body = '{"category":"linear","symbol":"BTCUSDT"}'
    expected = hmac.new(
        b"YYYYYYYYYY", f"1658384314791XXXXXXXXXX5000{body}".encode(), hashlib.sha256
    ).hexdigest()
    assert rest_signature(c, 1658384314791, 5000, body) == expected
    h = rest_headers(c, 1658384314791, 5000, body)
    assert h["X-BAPI-SIGN"] == expected and h["X-BAPI-API-KEY"] == "XXXXXXXXXX"
    args = ws_auth_args(c, 1662350400000)
    assert args[2] == hmac.new(b"YYYYYYYYYY", b"GET/realtime1662350400000", hashlib.sha256).hexdigest()
    assert "YYYY" not in repr(c)


def test_format_step() -> None:
    assert format_step(0.0199999, 0.001) == "0.019"
    assert format_step(30000.06, 0.1, "ROUND_HALF_UP") == "30000.1"
    assert format_step(12.0, 0.1) == "12"


# --------------------------------------------------------------------------- websocket client


async def _collect(ws: ReconnectingWebSocket, stop: asyncio.Event, until: Any, timeout: float = 10.0) -> None:
    task = asyncio.create_task(ws.run(stop))
    try:
        for _ in range(int(timeout / 0.02)):
            if until():
                break
            await asyncio.sleep(0.02)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=5)


async def test_ws_reconnects_and_resubscribes_after_server_drop() -> None:
    def script(conn: int) -> list[dict[str, Any]]:
        return [OB_SNAPSHOT, TRADES, TRADES]

    msgs: list[dict[str, Any]] = []
    statuses: list[str] = []
    async with FakeBybitServer(script, close_after={1: 2}) as srv:
        ws = ReconnectingWebSocket(
            "t",
            srv.url,
            ["orderbook.50.BTCUSDT", "publicTrade.BTCUSDT"],
            lambda m, ts: msgs.append(m),
            lambda s, d: statuses.append(s),
            backoff_initial_s=0.05,
        )
        await _collect(ws, asyncio.Event(), lambda: srv.connections >= 2 and len(msgs) >= 5)
    assert srv.connections >= 2
    assert statuses[:3] == ["connected", "disconnected", "connected"]
    subs = [r for r in srv.received if r.get("op") == "subscribe"]
    assert {r["conn"] for r in subs} >= {1, 2}  # resubscribed on the new connection
    assert sum(1 for m in msgs if m["topic"].startswith("orderbook")) >= 2  # fresh snapshot after reconnect


async def test_ws_resync_unsubscribes_and_resubscribes() -> None:
    async with FakeBybitServer(lambda c: [OB_SNAPSHOT], interval_s=0.01) as srv:
        ws = ReconnectingWebSocket(
            "t", srv.url, ["orderbook.50.BTCUSDT"], lambda m, ts: None, lambda s, d: None
        )
        stop = asyncio.Event()
        task = asyncio.create_task(ws.run(stop))
        for _ in range(200):
            if srv.connections:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        ws.request_resync("orderbook.50.BTCUSDT")
        for _ in range(300):
            if any(r.get("op") == "unsubscribe" for r in srv.received):
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=5)
    ops = [r["op"] for r in srv.received if r.get("op") in ("subscribe", "unsubscribe")]
    assert ops[:3] == ["subscribe", "unsubscribe", "subscribe"]
    assert ws.stats.resyncs == 1


async def test_ws_stale_feed_forces_reconnect() -> None:
    statuses: list[str] = []
    async with FakeBybitServer(lambda c: [], silent_after_subscribe=True) as srv:
        ws = ReconnectingWebSocket(
            "t",
            srv.url,
            ["tickers.BTCUSDT"],
            lambda m, ts: None,
            lambda s, d: statuses.append(s),
            stale_after_s=0.3,
            backoff_initial_s=0.05,
        )
        await _collect(ws, asyncio.Event(), lambda: srv.connections >= 2, timeout=10)
    assert "stale" in statuses and srv.connections >= 2


async def test_ws_auth_failure_raises() -> None:
    async with FakeBybitServer(lambda c: [], require_auth="GOODKEY") as srv:
        ws = ReconnectingWebSocket(
            "p",
            srv.url,
            ["order"],
            lambda m, ts: None,
            lambda s, d: None,
            credentials=Credentials("BADKEY", "s"),
        )
        with pytest.raises(AuthError):
            await asyncio.wait_for(ws.run(asyncio.Event()), timeout=5)


async def test_ws_bad_handler_does_not_kill_feed() -> None:
    seen: list[int] = []

    def handler(m: dict[str, Any], ts: int) -> None:
        seen.append(1)
        if len(seen) == 1:
            raise ValueError("boom")

    async with FakeBybitServer(lambda c: [TRADES, TRADES, TRADES]) as srv:
        ws = ReconnectingWebSocket("t", srv.url, ["publicTrade.BTCUSDT"], handler, lambda s, d: None)
        await _collect(ws, asyncio.Event(), lambda: len(seen) >= 3)
    assert len(seen) == 3 and ws.stats.errors


# --------------------------------------------------------------------------- REST + gateway

SPEC = {"BTCUSDT": InstrumentSpec(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001)}


class MockBybit:
    """httpx transport emulating the V5 REST endpoints the gateway uses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.create_behaviour: list[Any] = []
        self.orders: dict[str, dict[str, Any]] = {}
        self.execs: dict[str, list[dict[str, Any]]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        body = json.loads(request.content) if request.content else {}
        self.calls.append((path, {**params, **body}, dict(request.headers)))
        if path == "/v5/order/create":
            b = self.create_behaviour.pop(0) if self.create_behaviour else "ok"
            if b == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            if isinstance(b, tuple):
                return httpx.Response(200, json={"retCode": b[0], "retMsg": b[1], "result": {}})
            self.orders[body["orderLinkId"]] = {
                "orderId": "X1",
                "orderLinkId": body["orderLinkId"],
                "orderStatus": "Filled",
                "cumExecQty": body["qty"],
            }
            return httpx.Response(
                200,
                json={
                    "retCode": 0,
                    "retMsg": "OK",
                    "result": {"orderId": "X1", "orderLinkId": body["orderLinkId"]},
                },
            )
        if path in ("/v5/order/realtime", "/v5/order/history"):
            o = self.orders.get(params.get("orderLinkId", ""))
            if path == "/v5/order/realtime":
                return httpx.Response(200, json={"retCode": 0, "result": {"list": []}})  # not open anymore
            return httpx.Response(200, json={"retCode": 0, "result": {"list": [o] if o else []}})
        if path == "/v5/execution/list":
            return httpx.Response(
                200, json={"retCode": 0, "result": {"list": self.execs.get(params["orderLinkId"], [])}}
            )
        if path == "/v5/position/list":
            return httpx.Response(
                200,
                json={
                    "retCode": 0,
                    "result": {
                        "list": [{"symbol": "BTCUSDT", "side": "Buy", "size": "0.002", "entryPrice": "30000"}]
                    },
                },
            )
        if path == "/v5/account/wallet-balance":
            return httpx.Response(
                200,
                json={
                    "retCode": 0,
                    "result": {"list": [{"totalEquity": "199.5", "totalWalletBalance": "199"}]},
                },
            )
        return httpx.Response(404)


def _gateway(mock: MockBybit) -> tuple[BybitExecutionGateway, list[Event]]:
    out: list[Event] = []
    rest = BybitRest(
        "https://api-testnet.bybit.com",
        Credentials("k", "s"),
        transport=httpx.MockTransport(mock),
        clock_ms=lambda: 1_000,
    )
    gw = BybitExecutionGateway("challenge", rest, "challenge", out.append, SPEC, clock_ms=lambda: 2_000)
    return gw, out


def _order(cid: str = "C-A0001-0000001") -> OrderRequest:
    return OrderRequest(
        client_order_id=cid,
        account="challenge",
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=0.0021,
        order_type=OrderType.LIMIT,
        tif=TimeInForce.IOC,
        limit_price=30000.06,
        ts=0,
    )


async def test_gateway_submit_success_is_ack_not_fill() -> None:
    mock = MockBybit()
    gw, out = _gateway(mock)
    gw.submit(_order())
    await gw.drain()
    path, params, headers = mock.calls[0]
    assert path == "/v5/order/create" and params["qty"] == "0.002" and params["price"] == "30000.1"
    assert (
        params["timeInForce"] == "IOC"
        and params["positionIdx"] == 0
        and params["orderLinkId"] == "C-A0001-0000001"
    )
    assert "X-BAPI-SIGN".lower() in {k.lower() for k in headers}
    assert len(out) == 1 and isinstance(out[0], OrderUpdate) and out[0].status is OrderStatus.NEW
    assert not any(isinstance(e, FillEvent) for e in out)


async def test_gateway_definitive_reject() -> None:
    mock = MockBybit()
    mock.create_behaviour = [(110007, "ab not enough for new order")]
    gw, out = _gateway(mock)
    gw.submit(_order())
    await gw.drain()
    assert isinstance(out[0], OrderUpdate) and out[0].status is OrderStatus.REJECTED
    assert out[0].reason.startswith("bybit:110007")


async def test_gateway_ambiguous_timeout_then_query_recovers_fills() -> None:
    mock = MockBybit()
    mock.create_behaviour = ["timeout"]
    gw, out = _gateway(mock)
    gw.submit(_order())
    await gw.drain()
    assert out == []  # unknown outcome: no fake reject, no fake ack
    # in reality the order landed: the exchange knows it and has an execution
    mock.orders["C-A0001-0000001"] = {
        "orderId": "X9",
        "orderLinkId": "C-A0001-0000001",
        "orderStatus": "Filled",
        "cumExecQty": "0.002",
    }
    mock.execs["C-A0001-0000001"] = [
        {
            "category": "linear",
            "symbol": "BTCUSDT",
            "execId": "E9",
            "execPrice": "30000",
            "execQty": "0.002",
            "execFee": "0.033",
            "execType": "Trade",
            "side": "Buy",
            "orderLinkId": "C-A0001-0000001",
            "orderId": "X9",
        }
    ]
    gw.query_order("challenge", "C-A0001-0000001", "BTCUSDT", 0)
    await gw.drain()
    fills = [e for e in out if isinstance(e, FillEvent)]
    ups = [e for e in out if isinstance(e, OrderUpdate)]
    assert fills and fills[0].exec_id == "E9"
    assert ups[-1].status is OrderStatus.FILLED and ups[-1].reason == "reconciled"


async def test_gateway_duplicate_link_id_reconciles_instead_of_rejecting() -> None:
    mock = MockBybit()
    mock.create_behaviour = [(110072, "OrderLinkedID is duplicate")]
    mock.orders["C-A0001-0000001"] = {
        "orderId": "X1",
        "orderLinkId": "C-A0001-0000001",
        "orderStatus": "New",
        "cumExecQty": "0",
    }
    gw, out = _gateway(mock)
    gw.submit(_order())
    await gw.drain()
    ups = [e for e in out if isinstance(e, OrderUpdate)]
    assert ups and ups[-1].status is OrderStatus.NEW and ups[-1].reason == "reconciled"


async def test_gateway_query_not_found_is_definitive() -> None:
    mock = MockBybit()
    gw, out = _gateway(mock)
    gw.query_order("challenge", "C-nope", "BTCUSDT", 0)
    await gw.drain()
    assert isinstance(out[0], OrderUpdate) and out[0].status is OrderStatus.REJECTED
    assert out[0].reason == "order_not_found"


async def test_gateway_positions_and_wallet() -> None:
    mock = MockBybit()
    gw, out = _gateway(mock)
    gw.query_positions("challenge", 0)
    await gw.drain()
    pos = [e for e in out if isinstance(e, PositionSnapshot)]
    assert pos and pos[0].qty == 0.002
    assert any(isinstance(e, WalletSnapshot) and e.equity == 199.5 for e in out)
