"""Bybit execution gateway (testnet / live) implementing :class:`ExecutionGateway`.

Semantics that matter:

* A successful ``/v5/order/create`` response only means *request accepted*; the order's real
  state comes from the private ``order``/``execution`` streams. We emit an ``OrderUpdate(NEW)``
  on REST success (an acknowledgement, never a fill).
* A **definitive** rejection (bad params, insufficient balance, ...) becomes
  ``OrderUpdate(REJECTED)``. An **ambiguous** failure (timeout, 5xx, rate limit, network error)
  emits nothing: the engine's ACK timeout triggers :meth:`query_order`, which resolves the order
  by ``orderLinkId`` (open orders -> history -> "not found") and replays its executions (the
  engine dedups by ``execId``). ``orderLinkId is duplicate`` means an earlier attempt landed.
* After a private-stream reconnect the gateway re-queries every open order and the positions,
  closing any gap in the event stream.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any

import httpx

from darwin.config.challenge import InstrumentSpec
from darwin.core.events import Event, OrderUpdate, PositionSnapshot, WalletSnapshot
from darwin.core.types import OrderStatus, TimeInForce
from darwin.exchange.bybit.parser import ORDER_STATUS_MAP, parse_private
from darwin.exchange.bybit.rest import DUPLICATE_LINK_ID, BybitApiError, BybitRest
from darwin.exchange.bybit.ws import now_ms
from darwin.execution.orders import OrderRequest

log = logging.getLogger(__name__)


def format_step(value: float, step: float, rounding: str = ROUND_DOWN) -> str:
    q = Decimal(str(step))
    d = (Decimal(str(value)) / q).to_integral_value(rounding=rounding) * q
    return format(d.normalize(), "f")


class BybitExecutionGateway:
    def __init__(
        self,
        venue: str,
        rest: BybitRest,
        account: str,
        emit: Callable[[Event], None],
        instruments: dict[str, InstrumentSpec],
        clock_ms: Callable[[], int] = now_ms,
    ) -> None:
        self.venue = venue
        self.rest = rest
        self.account = account
        self.emit = emit
        self.instruments = instruments
        self.clock_ms = clock_ms
        self._tasks: set[asyncio.Task[Any]] = set()
        self.open_ids: dict[str, str] = {}  # client_order_id -> symbol (for reconnect reconciliation)
        self.errors = 0

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        # wait only on unfinished tasks: since Python 3.12 gather() over already-finished tasks
        # completes without yielding, so looping on the set (whose done-callbacks never get to
        # run) spins forever and starves the event loop, timeouts included
        while pending := [t for t in self._tasks if not t.done()]:
            await asyncio.gather(*pending, return_exceptions=True)

    async def aclose(self, timeout_s: float = 10.0) -> None:
        """Let in-flight REST calls finish (bounded), then cancel the rest — before the REST
        client is closed underneath them."""
        try:
            await asyncio.wait_for(self.drain(), timeout=timeout_s)
        except TimeoutError:
            for t in list(self._tasks):
                t.cancel()
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------ ExecutionGateway
    def submit(self, order: OrderRequest) -> None:
        self.open_ids[order.client_order_id] = order.symbol
        self._spawn(self._submit(order))

    def cancel(self, account: str, client_order_id: str, symbol: str, ts: int) -> None:
        self._spawn(self._cancel(symbol, client_order_id))

    def query_order(self, account: str, client_order_id: str, symbol: str, ts: int) -> None:
        self._spawn(self._query(symbol, client_order_id))

    def query_positions(self, account: str, ts: int) -> None:
        self._spawn(self._positions())

    def query_open_orders(self, account: str, ts: int) -> None:
        self._spawn(self._open_orders())

    async def _open_orders(self) -> None:
        """Venue-wide sweep of working orders: catches lost final reports (an order we think is
        working but the venue no longer lists) and orders on our symbols we did not place."""
        try:
            rows = [r for r in await self.rest.open_orders() if r.get("symbol") in self.instruments]
        except (BybitApiError, httpx.HTTPError, TimeoutError) as e:
            log.info("open-order sweep failed: %s", e)
            return
        recv = self.clock_ms()
        listed: set[str] = set()
        for ev in parse_private({"topic": "order", "data": rows}, recv, self.account):
            if isinstance(ev, OrderUpdate):
                listed.add(ev.client_order_id)
                self.emit(ev.model_copy(update={"reason": "open_orders_sweep"}))
        for cid, sym in list(self.open_ids.items()):
            if cid not in listed:
                await self._query(sym, cid)  # left the book: fetch its final state and executions

    # ------------------------------------------------------------------ internals
    def _order_params(self, o: OrderRequest) -> dict[str, Any]:
        spec = self.instruments[o.symbol]
        tif = {
            TimeInForce.IOC: "IOC",
            TimeInForce.POST_ONLY: "PostOnly",
            TimeInForce.GTC: "GTC",
            TimeInForce.FOK: "FOK",
        }[o.tif]
        params: dict[str, Any] = {
            "symbol": o.symbol,
            "side": o.side.value,
            "orderType": o.order_type.value,
            "qty": format_step(o.qty, spec.qty_step),
            "timeInForce": tif,
            "orderLinkId": o.client_order_id,
            "positionIdx": 0,  # one-way mode
            "reduceOnly": o.reduce_only,
        }
        if o.limit_price is not None:
            params["price"] = format_step(o.limit_price, spec.tick_size, ROUND_HALF_UP)
        return params

    def _update(self, cid: str, symbol: str, status: OrderStatus, reason: str = "", **kw: Any) -> None:
        self.emit(
            OrderUpdate(
                ts=self.clock_ms(),
                account=self.account,
                client_order_id=cid,
                symbol=symbol,
                status=status,
                reason=reason,
                **kw,
            )
        )

    async def _submit(self, o: OrderRequest) -> None:
        try:
            res = await self.rest.place_order(**self._order_params(o))
            self._update(
                o.client_order_id, o.symbol, OrderStatus.NEW, "rest_ack", exchange_order_id=res.get("orderId")
            )
        except BybitApiError as e:
            if e.ret_code == DUPLICATE_LINK_ID:
                await self._query(o.symbol, o.client_order_id)
            elif e.definitive:
                self.open_ids.pop(o.client_order_id, None)
                self._update(
                    o.client_order_id, o.symbol, OrderStatus.REJECTED, f"bybit:{e.ret_code}:{e.ret_msg}"
                )
            else:
                self.errors += 1
                log.warning("ambiguous order failure %s: %s (will reconcile)", o.client_order_id, e)
        except (httpx.HTTPError, TimeoutError) as e:
            self.errors += 1
            log.warning("order %s transport failure %r (will reconcile)", o.client_order_id, e)

    async def _cancel(self, symbol: str, cid: str) -> None:
        try:
            await self.rest.cancel_order(symbol, cid)
        except (BybitApiError, httpx.HTTPError, TimeoutError) as e:
            log.info("cancel %s failed: %s (state will come from stream/reconcile)", cid, e)

    async def _query(self, symbol: str, cid: str) -> None:
        try:
            row = await self.rest.order_by_link_id(symbol, cid)
            if row is None:
                self.open_ids.pop(cid, None)
                self._update(cid, symbol, OrderStatus.REJECTED, "order_not_found")
                return
            status = ORDER_STATUS_MAP.get(row.get("orderStatus", ""), OrderStatus.UNKNOWN)
            execs = await self.rest.executions_by_link_id(symbol, cid)
            recv = self.clock_ms()
            # replay executions first (idempotent by execId), then the order state
            for ev in parse_private({"topic": "execution", "data": execs}, recv, self.account):
                self.emit(ev)
            self._update(
                cid,
                symbol,
                status,
                "reconciled",
                exchange_order_id=row.get("orderId"),
                cum_qty=float(row.get("cumExecQty") or 0.0),
            )
            if status.terminal:
                self.open_ids.pop(cid, None)
        except (BybitApiError, httpx.HTTPError, TimeoutError) as e:
            self.errors += 1
            log.warning("query %s failed: %s", cid, e)

    async def _positions(self) -> None:
        try:
            rows = await self.rest.positions()
            recv = self.clock_ms()
            seen: set[str] = set()
            for ev in parse_private({"topic": "position", "data": rows}, recv, self.account):
                if isinstance(ev, PositionSnapshot):
                    seen.add(ev.symbol)
                self.emit(ev)
            for sym in self.instruments:
                if sym not in seen:  # flat symbols are not listed: report explicit zero
                    self.emit(PositionSnapshot(ts=recv, account=self.account, symbol=sym, qty=0.0))
            w = await self.rest.wallet()
            if w is not None:
                self.emit(
                    WalletSnapshot(
                        ts=recv,
                        account=self.account,
                        equity=float(w["totalEquity"]),
                        wallet_balance=float(w.get("totalWalletBalance") or w["totalEquity"]),
                    )
                )
        except (BybitApiError, httpx.HTTPError, TimeoutError) as e:
            self.errors += 1
            log.warning("position query failed: %s", e)

    def on_private_reconnect(self) -> None:
        for cid, sym in list(self.open_ids.items()):
            self._spawn(self._query(sym, cid))
        self._spawn(self._positions())

    def on_private_event(self, ev: Event) -> None:
        if isinstance(ev, OrderUpdate) and ev.status.terminal:
            self.open_ids.pop(ev.client_order_id, None)
