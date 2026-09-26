"""Simulated perpetual-futures venue.

Used for (a) every agent's shadow evaluation book, and (b) the challenge account in replay/sim/
paper modes. It deliberately behaves like a real venue from the engine's point of view:

* orders arrive after a sampled network latency; acknowledgements come back after another one;
* marketable orders walk the *current* L2 book (liquidity consumed per account until the next
  book update), plus a size-dependent impact term; IOC remainders are cancelled (partial fills);
* post-only orders rest and only fill when the tape trades *through* their price (conservative);
* taker/maker fees, funding settlements, and maintenance-margin liquidations;
* an optional chaos layer injects rejects, lost ACKs, duplicate fills and out-of-order delivery
  so the execution engine's idempotency and reconciliation paths are exercised in tests.

The venue keeps its own positions/cash (the "exchange truth") independently of the engine's
ledger; reconciliation compares the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from darwin.config.challenge import FeeSchedule, InstrumentSpec, SimSettings
from darwin.core.events import (
    BookDelta,
    BookSnapshot,
    Event,
    FillEvent,
    FundingPayment,
    FundingSettlement,
    OrderUpdate,
    PositionSnapshot,
    TickerEvent,
    TradeEvent,
    WalletSnapshot,
)
from darwin.core.types import OrderStatus, OrderType, Side, TimeInForce
from darwin.exchange.scheduler import EventTarget, Scheduler
from darwin.execution.orders import OrderRequest
from darwin.market.state import MarketDataEvent, MarketState


class OrderArrival(Event):
    kind: Literal["order_arrival"] = "order_arrival"
    order: OrderRequest


class CancelArrival(Event):
    kind: Literal["cancel_arrival"] = "cancel_arrival"
    account: str
    client_order_id: str
    symbol: str


class QueryArrival(Event):
    kind: Literal["query_arrival"] = "query_arrival"
    account: str
    what: Literal["order", "positions", "open_orders"]
    client_order_id: str | None = None
    symbol: str | None = None


@dataclass(frozen=True)
class Chaos:
    reject_prob: float = 0.0
    drop_ack_prob: float = 0.0
    duplicate_fill_prob: float = 0.0
    #: deliver fills *before* the ACK (ACK delayed by this many ms)
    late_ack_ms: int = 0


@dataclass
class _SimOrder:
    req: OrderRequest
    status: OrderStatus = OrderStatus.NEW
    #: executions, replayed on query like Bybit's /v5/execution/list (the engine dedupes)
    fills: list[FillEvent] = field(default_factory=list)
    cum_qty: float = 0.0
    notional: float = 0.0
    exchange_order_id: str = ""

    @property
    def remaining(self) -> float:
        return max(self.req.qty - self.cum_qty, 0.0)


@dataclass
class _SimAccount:
    cash: float
    positions: dict[str, list[float]] = field(default_factory=dict)  # symbol -> [qty, avg]
    liquidated: int = 0


class SimExchange:
    def __init__(
        self,
        venue: str,
        symbols: tuple[str, ...] | list[str],
        fees: FeeSchedule,
        instruments: dict[str, InstrumentSpec],
        sim: SimSettings,
        scheduler: Scheduler,
        engine: EventTarget,
        maintenance_margin_rate: float = 0.005,
        chaos: Chaos | None = None,
        seed_offset: int = 0,
    ) -> None:
        self.venue = venue
        self.fees = fees
        self.instruments = instruments
        self.sim = sim
        self.scheduler = scheduler
        self.engine = engine
        self.mmr = maintenance_margin_rate
        self.chaos = chaos or Chaos()
        self.market = MarketState(symbols)
        self.rng = np.random.default_rng(sim.seed + 10_007 * (seed_offset + 1))
        self.accounts: dict[str, _SimAccount] = {}
        self.orders: dict[tuple[str, str], _SimOrder] = {}
        self.resting: dict[str, list[_SimOrder]] = {s: [] for s in symbols}
        self._consumed: dict[tuple[str, str, str, float], float] = {}
        self._exec_seq = 0
        self._order_seq = 0
        self.now = 0

    # ------------------------------------------------------------------ accounts
    def open_account(self, account: str, cash: float) -> None:
        self.accounts[account] = _SimAccount(cash=cash)

    def position(self, account: str, symbol: str) -> float:
        p = self.accounts[account].positions.get(symbol)
        return p[0] if p else 0.0

    def equity(self, account: str) -> float:
        acct = self.accounts[account]
        eq = acct.cash
        for sym, (qty, avg) in acct.positions.items():
            mark = self.market[sym].ref_price()
            if qty and mark:
                eq += qty * (mark - avg)
        return eq

    # ------------------------------------------------------------------ gateway API
    def submit(self, order: OrderRequest) -> None:
        arrive = order.ts + self._latency(self.sim.latency)
        self.scheduler.schedule(arrive, OrderArrival(ts=arrive, order=order), self)

    def cancel(self, account: str, client_order_id: str, symbol: str, ts: int) -> None:
        arrive = ts + self._latency(self.sim.latency)
        ev = CancelArrival(ts=arrive, account=account, client_order_id=client_order_id, symbol=symbol)
        self.scheduler.schedule(arrive, ev, self)

    def query_order(self, account: str, client_order_id: str, symbol: str, ts: int) -> None:
        arrive = ts + self._latency(self.sim.latency)
        ev = QueryArrival(
            ts=arrive, account=account, what="order", client_order_id=client_order_id, symbol=symbol
        )
        self.scheduler.schedule(arrive, ev, self)

    def query_positions(self, account: str, ts: int) -> None:
        arrive = ts + self._latency(self.sim.latency)
        self.scheduler.schedule(arrive, QueryArrival(ts=arrive, account=account, what="positions"), self)

    def query_open_orders(self, account: str, ts: int) -> None:
        arrive = ts + self._latency(self.sim.latency)
        self.scheduler.schedule(arrive, QueryArrival(ts=arrive, account=account, what="open_orders"), self)

    # ------------------------------------------------------------------ event entry points
    def on_market(self, ev: MarketDataEvent) -> None:
        self.now = max(self.now, ev.ts)
        self.market.apply(ev)
        if isinstance(ev, (BookSnapshot, BookDelta)):
            self._consumed = {k: v for k, v in self._consumed.items() if k[1] != ev.symbol}
        elif isinstance(ev, TradeEvent):
            self._match_resting(ev)
        elif isinstance(ev, TickerEvent) and ev.mark_price is not None:
            self._check_liquidations(ev.symbol, ev.ts)
        elif isinstance(ev, FundingSettlement):
            self._settle_funding(ev)

    def handle(self, event: Event) -> None:
        self.now = max(self.now, event.ts)
        if isinstance(event, OrderArrival):
            self._on_arrival(event.order, event.ts)
        elif isinstance(event, CancelArrival):
            self._on_cancel(event)
        elif isinstance(event, QueryArrival):
            self._on_query(event)

    # ------------------------------------------------------------------ internals
    def _latency(self, model: object) -> int:
        lo, hi = model.min_ms, model.max_ms  # type: ignore[attr-defined]
        return int(self.rng.integers(lo, hi + 1)) if hi > lo else int(lo)

    def _emit(self, ts: int, ev: Event) -> None:
        self.scheduler.schedule(ts, ev, self.engine)

    def _update(self, o: _SimOrder, ts: int, reason: str = "") -> OrderUpdate:
        avg = o.notional / o.cum_qty if o.cum_qty > 0 else None
        return OrderUpdate(
            ts=ts,
            account=o.req.account,
            client_order_id=o.req.client_order_id,
            exchange_order_id=o.exchange_order_id,
            symbol=o.req.symbol,
            status=o.status,
            cum_qty=o.cum_qty,
            avg_price=avg,
            reason=reason,
        )

    def _on_arrival(self, req: OrderRequest, ts: int) -> None:
        ack_ts = ts + self._latency(self.sim.ack_latency)
        key = (req.account, req.client_order_id)
        if key in self.orders:
            # idempotent venue: duplicate client ids are rejected, never double-executed
            self._emit(
                ack_ts,
                OrderUpdate(
                    ts=ack_ts,
                    account=req.account,
                    client_order_id=req.client_order_id,
                    symbol=req.symbol,
                    status=OrderStatus.REJECTED,
                    reason="duplicate_client_order_id",
                ),
            )
            return
        self._order_seq += 1
        o = _SimOrder(req=req, exchange_order_id=f"{self.venue}-O{self._order_seq}")
        self.orders[key] = o
        if req.account not in self.accounts:
            o.status = OrderStatus.REJECTED
            self._emit(ack_ts, self._update(o, ack_ts, "unknown_account"))
            return
        if self.chaos.reject_prob and self.rng.random() < self.chaos.reject_prob:
            o.status = OrderStatus.REJECTED
            self._emit(ack_ts, self._update(o, ack_ts, "chaos_reject"))
            return
        st = self.market[req.symbol]
        if not st.book.valid:
            o.status = OrderStatus.REJECTED
            self._emit(ack_ts, self._update(o, ack_ts, "book_unavailable"))
            return
        if req.reduce_only:
            pos = self.position(req.account, req.symbol)
            if pos == 0 or (pos > 0) == (req.side is Side.BUY):
                o.status = OrderStatus.REJECTED
                self._emit(ack_ts, self._update(o, ack_ts, "reduce_only_would_increase"))
                return
            if req.qty > abs(pos):  # like Bybit: a reduce-only order never flips the position
                req = req.model_copy(update={"qty": abs(pos)})
                o.req = req

        drop_ack = bool(self.chaos.drop_ack_prob) and self.rng.random() < self.chaos.drop_ack_prob
        ack_emit_ts = ack_ts + self.chaos.late_ack_ms
        if req.tif is TimeInForce.POST_ONLY:
            bb, ba = st.book.best_bid(), st.book.best_ask()
            limit = req.limit_price
            crosses = limit is not None and (
                (req.side is Side.BUY and ba is not None and limit >= ba)
                or (req.side is Side.SELL and bb is not None and limit <= bb)
            )
            if crosses or limit is None:
                o.status = OrderStatus.REJECTED
                self._emit(ack_ts, self._update(o, ack_ts, "post_only_would_cross"))
                return
            o.status = OrderStatus.NEW
            self.resting[req.symbol].append(o)
            if not drop_ack:
                self._emit(ack_emit_ts, self._update(o, ack_emit_ts))
            return

        # marketable order: ACK then fills from walking the book
        o.status = OrderStatus.NEW
        if not drop_ack:
            self._emit(ack_emit_ts, self._update(o, ack_emit_ts))
        fills = self._walk(req)
        for price, qty in fills:
            self._fill(o, price, qty, is_maker=False, ts=ack_ts)
        if o.remaining > 1e-12:
            # IOC/market remainder cancelled => partial fill (or no fill at all)
            o.status = OrderStatus.CANCELED
            self._emit(
                ack_ts, self._update(o, ack_ts, "ioc_remainder_cancelled" if fills else "no_liquidity")
            )
        else:
            o.status = OrderStatus.FILLED
            self._emit(ack_ts, self._update(o, ack_ts))

    def _walk(self, req: OrderRequest) -> list[tuple[float, float]]:
        book = self.market[req.symbol].book
        levels = book.top_asks(100) if req.side is Side.BUY else book.top_bids(100)
        limit = req.limit_price if req.order_type is OrderType.LIMIT else None
        remaining = req.qty
        out: list[tuple[float, float]] = []
        top_notional = sum(p * q for p, q in levels[:5]) or 1.0
        order_notional = req.qty * (levels[0][0] if levels else 0.0)
        impact = self.sim.impact_bps_per_book * order_notional / top_notional / 1e4
        for price, avail in levels:
            if limit is not None and (
                (req.side is Side.BUY and price > limit) or (req.side is Side.SELL and price < limit)
            ):
                break
            ckey = (req.account, req.symbol, req.side.value, price)
            used = self._consumed.get(ckey, 0.0)
            take = min(remaining, avail - used)
            if take <= 0:
                continue
            self._consumed[ckey] = used + take
            px = price * (1 + req.side.sign * impact)
            if limit is not None:
                px = min(px, limit) if req.side is Side.BUY else max(px, limit)
            out.append((px, take))
            remaining -= take
            if remaining <= 1e-12:
                break
        return out

    def _fill(
        self, o: _SimOrder, price: float, qty: float, is_maker: bool, ts: int, liquidation: bool = False
    ) -> None:
        fee_bps = self.fees.maker_bps if is_maker else self.fees.taker_bps
        fee = qty * price * fee_bps / 1e4
        if liquidation:
            fee = qty * price * self.sim.liquidation_fee_rate
        o.cum_qty += qty
        o.notional += qty * price
        self._apply_position(o.req.account, o.req.symbol, o.req.side, qty, price, fee)
        self._exec_seq += 1
        fill = FillEvent(
            ts=ts,
            account=o.req.account,
            client_order_id=o.req.client_order_id,
            exec_id=f"{self.venue}-E{self._exec_seq}",
            symbol=o.req.symbol,
            side=o.req.side,
            qty=qty,
            price=price,
            fee=fee,
            is_maker=is_maker,
            is_liquidation=liquidation,
        )
        o.fills.append(fill)
        self._emit(ts, fill)
        if self.chaos.duplicate_fill_prob and self.rng.random() < self.chaos.duplicate_fill_prob:
            self._emit(ts + 1, fill)  # at-least-once delivery: the engine must dedupe by exec_id

    def _apply_position(
        self, account: str, symbol: str, side: Side, qty: float, price: float, fee: float
    ) -> None:
        acct = self.accounts[account]
        pos = acct.positions.setdefault(symbol, [0.0, 0.0])
        cur, avg = pos
        signed = qty * side.sign
        if cur == 0 or (cur > 0) == (signed > 0):
            new = cur + signed
            pos[1] = (avg * abs(cur) + price * qty) / abs(new)
            pos[0] = round(new, 10)
        else:
            closing = min(abs(signed), abs(cur))
            acct.cash += closing * (price - avg) * (1 if cur > 0 else -1)
            new = round(cur + signed, 10)
            pos[0] = new
            if abs(signed) > abs(cur):
                pos[1] = price
            elif new == 0:
                pos[1] = 0.0
        acct.cash -= fee

    def _match_resting(self, tr: TradeEvent) -> None:
        book = self.resting.get(tr.symbol)
        if not book:
            return
        available = tr.qty
        still: list[_SimOrder] = []
        for o in book:
            lim = o.req.limit_price
            assert lim is not None
            through = (o.req.side is Side.BUY and tr.price < lim) or (
                o.req.side is Side.SELL and tr.price > lim
            )
            if o.status.terminal:
                continue
            if through and available > 0:
                q = min(o.remaining, available)
                available -= q
                self._fill(o, lim, q, is_maker=True, ts=tr.ts)
                if o.remaining <= 1e-12:
                    o.status = OrderStatus.FILLED
                    self._emit(tr.ts, self._update(o, tr.ts))
                    continue
                o.status = OrderStatus.PARTIALLY_FILLED
                self._emit(tr.ts, self._update(o, tr.ts))
            still.append(o)
        self.resting[tr.symbol] = still

    def _on_cancel(self, ev: CancelArrival) -> None:
        o = self.orders.get((ev.account, ev.client_order_id))
        ack = ev.ts + self._latency(self.sim.ack_latency)
        if o is None or o.status.terminal:
            self._emit(
                ack,
                OrderUpdate(
                    ts=ack,
                    account=ev.account,
                    client_order_id=ev.client_order_id,
                    symbol=ev.symbol,
                    status=o.status if o else OrderStatus.REJECTED,
                    cum_qty=o.cum_qty if o else 0.0,
                    reason="cancel_rejected:" + ("terminal" if o else "unknown_order"),
                ),
            )
            return
        o.status = OrderStatus.CANCELED
        self.resting[o.req.symbol] = [x for x in self.resting[o.req.symbol] if x is not o]
        self._emit(ack, self._update(o, ack, "cancelled_by_user"))

    def _on_query(self, ev: QueryArrival) -> None:
        ack = ev.ts + self._latency(self.sim.ack_latency)
        if ev.what == "open_orders":
            for (owner, _cid), working in self.orders.items():
                if owner == ev.account and not working.status.terminal:
                    self._emit(ack, self._update(working, ack, "open_orders_sweep"))
            return
        if ev.what == "positions":
            acct = self.accounts.get(ev.account)
            if acct is None:
                return
            for sym in self.market.symbols:
                qty, avg = acct.positions.get(sym, [0.0, 0.0])
                self._emit(
                    ack,
                    PositionSnapshot(
                        ts=ack, account=ev.account, symbol=sym, qty=qty, entry_price=avg or None
                    ),
                )
            self._emit(
                ack,
                WalletSnapshot(
                    ts=ack, account=ev.account, equity=self.equity(ev.account), wallet_balance=acct.cash
                ),
            )
            return
        assert ev.client_order_id is not None and ev.symbol is not None
        o = self.orders.get((ev.account, ev.client_order_id))
        if o is None:
            # the order never reached us: definitive "does not exist" => engine may treat as rejected
            self._emit(
                ack,
                OrderUpdate(
                    ts=ack,
                    account=ev.account,
                    client_order_id=ev.client_order_id,
                    symbol=ev.symbol,
                    status=OrderStatus.REJECTED,
                    reason="order_not_found",
                ),
            )
            return
        for f in o.fills:  # replay executions first (idempotent by exec_id), then the state
            self._emit(ack, f.model_copy(update={"ts": ack}))
        self._emit(ack, self._update(o, ack, "query"))

    def _check_liquidations(self, symbol: str, ts: int) -> None:
        for account, acct in self.accounts.items():
            if not any(q for q, _ in acct.positions.values()):
                continue
            if symbol not in acct.positions or acct.positions[symbol][0] == 0:
                continue
            equity = acct.cash
            maint = 0.0
            for sym, (qty, avg) in acct.positions.items():
                if not qty:
                    continue
                mark = self.market[sym].ref_price() or avg
                equity += qty * (mark - avg)
                maint += abs(qty) * mark * self.mmr
            if equity > maint:
                continue
            acct.liquidated += 1
            for sym, (qty, _avg) in list(acct.positions.items()):
                if not qty:
                    continue
                mark = self.market[sym].ref_price() or _avg
                self._order_seq += 1
                req = OrderRequest(
                    client_order_id=f"LIQ-{self._order_seq}",
                    account=account,
                    symbol=sym,
                    side=Side.SELL if qty > 0 else Side.BUY,
                    qty=abs(qty),
                    order_type=OrderType.MARKET,
                    tif=TimeInForce.IOC,
                    ts=ts,
                )
                o = _SimOrder(req=req, exchange_order_id=f"{self.venue}-O{self._order_seq}")
                self.orders[(account, req.client_order_id)] = o
                self._fill(o, mark, abs(qty), is_maker=False, ts=ts, liquidation=True)

    def _settle_funding(self, ev: FundingSettlement) -> None:
        for account, acct in self.accounts.items():
            pos = acct.positions.get(ev.symbol)
            if not pos or pos[0] == 0:
                continue
            amount = pos[0] * ev.mark_price * ev.rate
            acct.cash -= amount
            self._emit(
                ev.ts,
                FundingPayment(
                    ts=ev.ts,
                    account=account,
                    symbol=ev.symbol,
                    position_qty=pos[0],
                    rate=ev.rate,
                    mark_price=ev.mark_price,
                    amount=amount,
                ),
            )
