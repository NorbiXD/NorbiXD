"""Execution engine: approved RiskDecisions -> venue orders -> fills -> ledger.

Guarantees
----------
* **Idempotent client ids.** Every order gets a deterministic ``client_order_id`` (<= 36 chars,
  Bybit ``orderLinkId`` compatible). Venues reject duplicates, so a retry can never double-fill.
* **ACK != fill.** Positions change only on ``FillEvent``s. ``OrderUpdate``s drive the state
  machine but never the ledger.
* **At-least-once tolerant.** Fills are deduplicated by ``exec_id``; stale/out-of-order order
  updates are ignored by the state machine's monotonic ranking.
* **Timeouts are reconciled, not assumed.** An order with no ACK past the timeout is queried by
  client id. Repeated silence marks it UNKNOWN and degrades venue health, which blocks new risk
  until reconciliation succeeds.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from darwin.core.events import FillEvent, OrderUpdate
from darwin.core.intent import TradeIntent
from darwin.core.types import OrderStatus, OrderType, Side, TimeInForce, Urgency
from darwin.execution.orders import ExecutionGateway, ManagedOrder, OrderRequest
from darwin.market.state import MarketState
from darwin.portfolio.ledger import Ledger, RoundTrip
from darwin.risk.governor import Reservations, RiskDecision, VenueHealth

log = logging.getLogger(__name__)

FillCallback = Callable[[ManagedOrder, FillEvent, list[RoundTrip]], None]

_BUSINESS_REJECTS = frozenset(
    {
        "post_only_would_cross",
        "duplicate_client_order_id",
        "reduce_only_would_increase",
        "order_not_found",
        "book_unavailable",
    }
)
# Bybit retCodes that are business outcomes of a healthy API (balance, price band, min size,
# reduce-only, order-count limits, bad params), unlike auth/permission/transport failures
_BYBIT_BUSINESS = re.compile(r"^bybit:(10001|110\d{3}):")


def is_business_reject(reason: str) -> bool:
    """A reject that says "the venue is working and said no" — it must not degrade venue
    health (which halts all new risk), unlike lost requests or auth failures."""
    return (
        reason in _BUSINESS_REJECTS
        or reason.startswith("cancel_rejected")
        or reason.startswith("EC_")  # Bybit stream rejectReason codes
        or _BYBIT_BUSINESS.match(reason) is not None
    )


@dataclass
class OrphanFill:
    """A fill for an order we don't know (manual trade, lost state, venue liquidation)."""

    event: FillEvent
    reason: str


class ExecutionEngine:
    def __init__(
        self,
        ledger: Ledger,
        market: MarketState,
        gateways: dict[str, ExecutionGateway],
        account_venue: Callable[[str], str],
        health: dict[str, VenueHealth],
        ack_timeout_ms: int = 3_000,
        passive_ttl_ms: int = 30_000,
        max_queries: int = 3,
        on_fill: FillCallback | None = None,
        run_tag: str = "",
    ) -> None:
        self.ledger = ledger
        self.market = market
        self.gateways = gateways
        self.account_venue = account_venue
        self.health = health
        self.ack_timeout_ms = ack_timeout_ms
        self.passive_ttl_ms = passive_ttl_ms
        self.max_queries = max_queries
        self.on_fill_cb = on_fill
        self.run_tag = run_tag
        self.orders: dict[str, ManagedOrder] = {}
        self._open_by_key: dict[tuple[str, str, str], set[str]] = {}
        self._open_ids: set[str] = set()
        self._seen_exec: dict[str, None] = {}  # insertion-ordered: pruning keeps the newest
        self._seq = 0
        self.orphans: list[OrphanFill] = []
        self.duplicate_fills = 0
        self.stale_updates = 0
        self._tick: dict[str, float] = {}

    # ------------------------------------------------------------------ queries
    def pending_qty(self, account: str, agent_id: str, symbol: str) -> float:
        ids = self._open_by_key.get((account, agent_id, symbol))
        if not ids:
            return 0.0
        return sum(self.orders[i].signed_remaining for i in ids)

    def pending_split(self, account: str, agent_id: str, symbol: str) -> tuple[float, float]:
        """Signed in-flight quantity split into (risk-increasing, risk-reducing) orders."""
        inc = red = 0.0
        for i in self._open_by_key.get((account, agent_id, symbol), ()):
            o = self.orders[i]
            if o.risk_increasing:
                inc += o.signed_remaining
            else:
                red += o.signed_remaining
        return inc, red

    def open_orders(self) -> list[ManagedOrder]:
        return [self.orders[i] for i in sorted(self._open_ids)]

    def reservations(self, account: str) -> Reservations:
        """Exposure committed by all open risk-increasing orders in ``account`` (all agents)."""
        qty: dict[str, float] = {}
        new_pos: set[tuple[str, str]] = set()
        acct = self.ledger.accounts.get(account)
        for cid in self._open_ids:
            o = self.orders[cid]
            if o.account != account or not o.risk_increasing:
                continue
            qty[o.symbol] = qty.get(o.symbol, 0.0) + o.remaining
            if acct is not None and acct.agent_qty(o.agent_id, o.symbol) == 0:
                new_pos.add((o.agent_id, o.symbol))
        return Reservations(qty_by_symbol=qty, new_positions=len(new_pos))

    def has_open(self, account: str, symbol: str) -> bool:
        return any(
            self.orders[i].account == account and self.orders[i].symbol == symbol for i in self._open_ids
        )

    def has_open_account(self, account: str) -> bool:
        return any(self.orders[i].account == account for i in self._open_ids)

    # ------------------------------------------------------------------ submission
    def _next_id(self, account: str, agent_id: str) -> str:
        self._seq += 1
        prefix = "S" if account.startswith("shadow:") else "C"
        # e.g. C3f9a1-A0042-0000123: deterministic per run, unique across runs (exchange
        # orderLinkIds must never collide with a previous run's), <= 36 chars
        return f"{prefix}{self.run_tag}-{agent_id[-8:]}-{self._seq:07d}"

    def submit(self, decision: RiskDecision, intent: TradeIntent, now: int) -> ManagedOrder | None:
        if not decision.approved or decision.order_qty == 0:
            return None
        venue = self.account_venue(decision.account_id)
        gw = self.gateways[venue]
        st = self.market[decision.symbol]
        side = Side.BUY if decision.order_qty > 0 else Side.SELL
        qty = abs(decision.order_qty)
        ref = decision.ref_price or st.ref_price(now) or 0.0
        order_type = OrderType.LIMIT
        if intent.urgency is Urgency.PASSIVE and decision.risk_increasing:
            price = st.book.best_bid() if side is Side.BUY else st.book.best_ask()
            tif = TimeInForce.POST_ONLY
        else:
            # marketable limit (IOC) with an explicit slippage cap: a "market" order that can
            # never fill at an absurd price on a thin or stale book
            slip = intent.max_slippage_bps / 1e4
            base = ref
            if not decision.risk_increasing:
                # exits favour certainty of execution: a wider cap, anchored on the worse of the
                # reference price and the executable touch, so a book that gapped away from a
                # lagging mark cannot leave a stop unfillable
                slip = max(slip, 0.02)
                touch = st.book.best_ask() if side is Side.BUY else st.book.best_bid()
                if touch and st.book.valid:
                    base = max(ref, touch) if side is Side.BUY else min(ref, touch)
            price = base * (1 + slip) if side is Side.BUY else base * (1 - slip)
            tif = TimeInForce.IOC
        if price is None or price <= 0:
            log.warning("no price for %s order; skipping", decision.symbol)
            return None
        spec_tick = self._tick.get(decision.symbol)
        if spec_tick:
            price = round(round(price / spec_tick) * spec_tick, 10)
        req = OrderRequest(
            client_order_id=self._next_id(decision.account_id, decision.agent_id),
            account=decision.account_id,
            symbol=decision.symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            tif=tif,
            limit_price=price,
            reduce_only=False,  # one-way netting across agents makes per-agent reduce-only unsafe
            ts=now,
        )
        mo = ManagedOrder(
            request=req,
            agent_id=decision.agent_id,
            genome_id=intent.genome_id,
            intent_id=intent.intent_id,
            decision_id=decision.decision_id,
            intent_reason=intent.reason,
            confidence=intent.confidence,
            ref_price=ref,
            stop_loss_pct=intent.stop_loss_pct,
            take_profit_pct=intent.take_profit_pct,
            max_hold_ms=intent.max_hold_ms,
            risk_increasing=decision.risk_increasing,
            created_ts=now,
            last_update_ts=now,
        )
        self.orders[req.client_order_id] = mo
        self._open_by_key.setdefault((req.account, decision.agent_id, req.symbol), set()).add(
            req.client_order_id
        )
        self._open_ids.add(req.client_order_id)
        gw.submit(req)
        return mo

    def cancel_open(
        self,
        account: str,
        now: int,
        agent_id: str | None = None,
        symbol: str | None = None,
        increasing_only: bool = True,
    ) -> int:
        """Request cancellation of working orders (kill switch, breakers, exits).

        Cancels are re-sent at most once per ACK timeout while the order stays working, so a
        lost cancel is retried without flooding the venue. Returns the number of requests sent.
        """
        sent = 0
        for cid in sorted(self._open_ids):
            o = self.orders[cid]
            if o.account != account or o.status.terminal:
                continue
            if agent_id is not None and o.agent_id != agent_id:
                continue
            if symbol is not None and o.symbol != symbol:
                continue
            if increasing_only and not o.risk_increasing:
                continue
            if o.cancel_requested_ts is not None and now - o.cancel_requested_ts < self.ack_timeout_ms:
                continue
            o.cancel_requested_ts = now
            self.gateways[self.account_venue(account)].cancel(account, cid, o.symbol, now)
            sent += 1
        return sent

    def set_tick_sizes(self, ticks: dict[str, float]) -> None:
        self._tick = dict(ticks)

    # ------------------------------------------------------------------ venue events
    def on_order_update(self, ev: OrderUpdate) -> ManagedOrder | None:
        mo = self.orders.get(ev.client_order_id)
        if mo is None:
            log.debug("order update for unknown order %s", ev.client_order_id)
            return None
        venue = self.account_venue(mo.account)
        if ev.exchange_order_id:
            mo.exchange_order_id = ev.exchange_order_id
        mo.reported_cum_qty = max(mo.reported_cum_qty, ev.cum_qty)
        if ev.status is OrderStatus.REJECTED:
            if not is_business_reject(ev.reason):
                self.health[venue].consecutive_errors += 1
        elif ev.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.health[venue].consecutive_errors = 0
        if ev.reason.startswith("cancel_rejected"):
            return mo
        was_unknown = mo.status is OrderStatus.UNKNOWN
        if not mo.apply_status(ev.status, ev.ts):
            self.stale_updates += 1
        if ev.reason:
            mo.reason = ev.reason
        self._maybe_close(mo)
        if was_unknown and mo.status is not OrderStatus.UNKNOWN:
            self._refresh_reconcile_state(venue)
        return mo

    def _refresh_reconcile_state(self, venue: str) -> None:
        unknown = [
            o
            for o in self.open_orders()
            if o.status is OrderStatus.UNKNOWN and self.account_venue(o.account) == venue
        ]
        h = self.health[venue]
        if not unknown and h.reconcile_detail.startswith("order "):
            h.reconcile_ok = True
            h.reconcile_detail = ""

    def on_fill(self, ev: FillEvent) -> tuple[ManagedOrder | None, list[RoundTrip]]:
        if ev.exec_id in self._seen_exec:
            self.duplicate_fills += 1
            return None, []
        self._seen_exec[ev.exec_id] = None
        mo = self.orders.get(ev.client_order_id)
        if mo is None:
            return None, self._orphan_fill(ev)
        if mo.account != ev.account or mo.symbol != ev.symbol or mo.request.side is not ev.side:
            self.orphans.append(OrphanFill(ev, "fill does not match order"))
            return None, []
        mo.apply_fill(ev.qty, ev.price, ev.fee, ev.exec_id, ev.ts)
        acct = self.ledger[mo.account]
        closed = self.ledger.on_fill(
            account_id=mo.account,
            agent_id=mo.agent_id,
            genome_id=mo.genome_id,
            symbol=mo.symbol,
            side=ev.side,
            qty=ev.qty,
            price=ev.price,
            fee=ev.fee,
            ts=ev.ts,
            intent_id=mo.intent_id,
            intent_reason=mo.intent_reason,
            confidence=mo.confidence,
            ref_price=mo.ref_price,
            equity_hint=acct.last_equity,
            stop_loss_pct=mo.stop_loss_pct,
            take_profit_pct=mo.take_profit_pct,
            max_hold_ms=mo.max_hold_ms,
        )
        self.health[self.account_venue(mo.account)].consecutive_errors = 0
        self._maybe_close(mo)
        if self.on_fill_cb:
            self.on_fill_cb(mo, ev, closed)
        return mo, closed

    def _orphan_fill(self, ev: FillEvent) -> list[RoundTrip]:
        if not ev.is_liquidation:
            self.orphans.append(OrphanFill(ev, "unknown client_order_id"))
            return []
        # venue liquidation: attribute pro-rata to the agents holding that side
        closed: list[RoundTrip] = []
        alloc = self.ledger.liquidation_allocation(ev.account, ev.symbol, ev.side)
        acct = self.ledger[ev.account]
        for agent_id, frac in alloc:
            pos_qty = abs(acct.agent_qty(agent_id, ev.symbol))
            q = min(ev.qty * frac, pos_qty)
            if q <= 0:
                continue
            rt = self.ledger.open_trades.get((ev.account, agent_id, ev.symbol))
            closed += self.ledger.on_fill(
                account_id=ev.account,
                agent_id=agent_id,
                genome_id=rt.genome_id if rt else "",
                symbol=ev.symbol,
                side=ev.side,
                qty=q,
                price=ev.price,
                fee=ev.fee * frac,
                ts=ev.ts,
                intent_id=None,
                intent_reason="venue_liquidation",
                confidence=0.0,
                ref_price=None,
                equity_hint=acct.last_equity,
            )
        return closed

    def _maybe_close(self, mo: ManagedOrder) -> None:
        if mo.open:
            return
        self._open_ids.discard(mo.client_order_id)
        key = (mo.account, mo.agent_id, mo.symbol)
        ids = self._open_by_key.get(key)
        if ids:
            ids.discard(mo.client_order_id)
            if not ids:
                del self._open_by_key[key]

    # ------------------------------------------------------------------ housekeeping
    def check_timeouts(self, now: int) -> None:
        for mo in self.open_orders():
            venue = self.account_venue(mo.account)
            gw = self.gateways[venue]
            if mo.status.terminal and mo.awaiting_fills > 0:
                # the venue says it executed more than we have fills for: ask for the executions
                if now - mo.last_update_ts >= self.ack_timeout_ms:
                    if mo.queries >= self.max_queries:
                        # give up waiting; trust fills and let position reconciliation arbitrate
                        mo.reported_cum_qty = mo.filled_qty
                        h = self.health[venue]
                        h.reconcile_ok = False
                        h.reconcile_detail = (
                            f"{mo.account}:{mo.symbol} missing fills for {mo.client_order_id}"
                        )
                        self._maybe_close(mo)
                        continue
                    mo.queries += 1
                    mo.last_update_ts = now
                    gw.query_order(mo.account, mo.client_order_id, mo.symbol, now)
                continue
            if mo.status is OrderStatus.UNKNOWN:
                # keep asking, at a slower cadence, until the venue gives a definitive answer
                if now - mo.last_update_ts >= 10 * self.ack_timeout_ms:
                    mo.last_update_ts = now
                    gw.query_order(mo.account, mo.client_order_id, mo.symbol, now)
                continue
            if mo.acked_ts is None and now - mo.last_update_ts >= self.ack_timeout_ms:
                if mo.queries >= self.max_queries:
                    mo.status = OrderStatus.UNKNOWN
                    mo.last_update_ts = now
                    self.health[venue].consecutive_errors += 1
                    self.health[venue].reconcile_ok = False
                    self.health[venue].reconcile_detail = f"order {mo.client_order_id} unknown"
                    continue
                mo.queries += 1
                mo.last_update_ts = now
                gw.query_order(mo.account, mo.client_order_id, mo.symbol, now)
            elif (
                mo.request.tif is TimeInForce.POST_ONLY
                and mo.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)
                and now - mo.created_ts >= self.passive_ttl_ms
                and now - mo.last_update_ts >= self.ack_timeout_ms
            ):
                mo.last_update_ts = now
                gw.cancel(mo.account, mo.client_order_id, mo.symbol, now)

    def prune(self, now: int, keep_ms: int = 3_600_000) -> int:
        """Forget resolved orders older than ``keep_ms`` and bound the exec-id memory."""
        stale = [c for c, o in self.orders.items() if not o.open and now - o.last_update_ts > keep_ms]
        for c in stale:
            del self.orders[c]
        if len(self._seen_exec) > 200_000:
            # exec ids of long-resolved orders can no longer arrive; keep the most recent half
            self._seen_exec = dict.fromkeys(list(self._seen_exec)[-100_000:])
        return len(stale)

    def drop_account(self, account: str) -> None:
        """Forget closed orders of a retired shadow account (memory hygiene)."""
        for cid in [c for c, o in self.orders.items() if o.account == account and not o.open]:
            del self.orders[cid]
