from __future__ import annotations

from typing import Any

import pytest

from darwin.config.challenge import FeeSchedule, InstrumentSpec, LatencyModel, SimSettings
from darwin.core.events import (
    Event,
    FillEvent,
    FundingPayment,
    FundingSettlement,
    OrderUpdate,
    PositionSnapshot,
)
from darwin.core.intent import TradeIntent
from darwin.core.types import OrderStatus, OrderType, Side, TimeInForce, Urgency
from darwin.exchange.sim.venue import Chaos, SimExchange
from darwin.execution.engine import ExecutionEngine
from darwin.execution.orders import OrderRequest
from darwin.market.state import MarketState
from darwin.portfolio.ledger import Ledger
from darwin.risk.governor import RiskDecision, VenueHealth
from tests.conftest import T0, Collector, ManualScheduler, book, ticker, trade

SPEC = {"BTCUSDT": InstrumentSpec(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001)}
SIM = SimSettings(
    latency=LatencyModel(min_ms=50, max_ms=50),
    ack_latency=LatencyModel(min_ms=10, max_ms=10),
    impact_bps_per_book=0.0,
)


def venue(sched: ManualScheduler, target: Any, chaos: Chaos | None = None) -> SimExchange:
    v = SimExchange(
        "challenge",
        ["BTCUSDT"],
        FeeSchedule(taker_bps=5.5, maker_bps=2.0),
        SPEC,
        SIM,
        sched,
        target,
        chaos=chaos,
    )
    v.open_account("challenge", 1_000.0)
    v.on_market(book("BTCUSDT", T0, 100.0, spread=1.0, qty=1.0, levels=5, step=1.0))
    v.on_market(ticker("BTCUSDT", T0, 100.0))
    return v


def req(
    qty: float,
    side: Side = Side.BUY,
    limit: float | None = None,
    tif: TimeInForce = TimeInForce.IOC,
    cid: str = "C-1",
    ts: int = T0,
) -> OrderRequest:
    return OrderRequest(
        client_order_id=cid,
        account="challenge",
        symbol="BTCUSDT",
        side=side,
        qty=qty,
        order_type=OrderType.LIMIT if limit else OrderType.MARKET,
        tif=tif,
        limit_price=limit,
        ts=ts,
    )


# ----------------------------------------------------------------------------- venue


def test_market_order_walks_book_with_fees() -> None:
    sched, col = ManualScheduler(), Collector()
    v = venue(sched, col)
    v.submit(req(1.5))  # asks: 100.5 x1, 101.5 x1 ...
    sched.drain()
    fills = col.of(FillEvent)
    assert [(f.price, f.qty) for f in fills] == [(100.5, 1.0), (101.5, 0.5)]
    assert fills[0].fee == pytest.approx(100.5 * 1.0 * 5.5 / 1e4)
    assert col.of(OrderUpdate)[-1].status is OrderStatus.FILLED
    assert v.position("challenge", "BTCUSDT") == pytest.approx(1.5)


def test_ioc_limit_partial_fill_then_cancel() -> None:
    sched, col = ManualScheduler(), Collector()
    v = venue(sched, col)
    v.submit(req(3.0, limit=101.5))
    sched.drain()
    assert sum(f.qty for f in col.of(FillEvent)) == pytest.approx(2.0)
    final = col.of(OrderUpdate)[-1]
    assert final.status is OrderStatus.CANCELED and final.cum_qty == pytest.approx(2.0)


def test_liquidity_consumed_until_next_book() -> None:
    sched, col = ManualScheduler(), Collector()
    v = venue(sched, col)
    v.submit(req(1.0, cid="C-1"))
    v.submit(req(1.0, cid="C-2"))
    sched.drain()
    prices = [f.price for f in col.of(FillEvent)]
    assert prices == [100.5, 101.5]  # the second order cannot re-use the consumed level


def test_post_only_rejects_when_crossing_and_fills_on_trade_through() -> None:
    sched, col = ManualScheduler(), Collector()
    v = venue(sched, col)
    v.submit(req(1.0, limit=100.5, tif=TimeInForce.POST_ONLY, cid="C-x"))
    sched.drain()
    assert col.of(OrderUpdate)[-1].status is OrderStatus.REJECTED
    v.submit(req(1.0, limit=99.5, tif=TimeInForce.POST_ONLY, cid="C-y", ts=T0 + 1))
    sched.drain()
    assert col.of(OrderUpdate)[-1].status is OrderStatus.NEW
    v.on_market(trade("BTCUSDT", T0 + 500, 99.5, qty=5.0, side=Side.SELL))  # at price: not through
    assert not col.of(FillEvent)
    v.on_market(trade("BTCUSDT", T0 + 600, 99.4, qty=5.0, side=Side.SELL))
    sched.drain()
    f = col.of(FillEvent)
    assert len(f) == 1 and f[0].is_maker and f[0].price == 99.5
    assert f[0].fee == pytest.approx(99.5 * 2.0 / 1e4)


def test_duplicate_client_order_id_never_double_executes() -> None:
    sched, col = ManualScheduler(), Collector()
    v = venue(sched, col)
    v.submit(req(0.5, cid="C-dup"))
    v.submit(req(0.5, cid="C-dup", ts=T0 + 1))
    sched.drain()
    assert sum(f.qty for f in col.of(FillEvent)) == pytest.approx(0.5)
    assert any(u.reason == "duplicate_client_order_id" for u in col.of(OrderUpdate))


def test_funding_settlement_charges_longs() -> None:
    sched, col = ManualScheduler(), Collector()
    v = venue(sched, col)
    v.submit(req(1.0))
    sched.drain()
    v.on_market(FundingSettlement(ts=T0 + 1000, symbol="BTCUSDT", rate=0.001, mark_price=100.0))
    sched.drain()
    pay = col.of(FundingPayment)
    assert len(pay) == 1 and pay[0].amount == pytest.approx(0.1)


def test_venue_liquidates_underwater_account() -> None:
    sched, col = ManualScheduler(), Collector()
    v = venue(sched, col)
    v.accounts["challenge"].cash = 10.0
    v.submit(req(2.0))
    sched.drain()
    v.on_market(ticker("BTCUSDT", T0 + 1000, 90.0))  # -2*~11 = -22 > equity 10
    sched.drain()
    liq = [f for f in col.of(FillEvent) if f.is_liquidation]
    assert liq and liq[0].side is Side.SELL
    assert v.position("challenge", "BTCUSDT") == 0


# ----------------------------------------------------------------------------- engine


class Harness:
    """ExecutionEngine + ledger + sim venue wired through one scheduler."""

    def __init__(self, chaos: Chaos | None = None, ack_timeout_ms: int = 3_000) -> None:
        self.sched = ManualScheduler()
        self.ledger = Ledger()
        self.ledger.open_account("challenge", "challenge", 1_000.0)
        self.market = MarketState(["BTCUSDT"])
        self.market.apply(book("BTCUSDT", T0, 100.0, spread=1.0, qty=1.0, levels=5, step=1.0))
        self.health = {"challenge": VenueHealth()}
        self.received: list[Event] = []
        self.venue = venue(self.sched, self, chaos)
        self.exe = ExecutionEngine(
            self.ledger,
            self.market,
            {"challenge": self.venue},
            lambda a: "challenge",
            self.health,
            ack_timeout_ms=ack_timeout_ms,
        )

    def handle(self, ev: Event) -> None:
        self.received.append(ev)
        if isinstance(ev, FillEvent):
            self.exe.on_fill(ev)
        elif isinstance(ev, OrderUpdate):
            self.exe.on_order_update(ev)

    def submit(self, qty: float, did: str = "RD1", now: int = T0) -> Any:
        intent = TradeIntent(
            intent_id="I" + did,
            ts=now,
            agent_id="A1",
            genome_id="G",
            symbol="BTCUSDT",
            target_exposure=0.1,
            confidence=0.5,
            reason="entry",
            stop_loss_pct=0.02,
            urgency=Urgency.MARKET,
            max_slippage_bps=200,
        )
        d = RiskDecision(
            decision_id=did,
            intent_id=intent.intent_id,
            account_id="challenge",
            agent_id="A1",
            symbol="BTCUSDT",
            ts=now,
            approved=True,
            risk_increasing=True,
            current_qty=0,
            pending_qty=0,
            requested_qty=qty,
            target_qty=qty,
            order_qty=qty,
            ref_price=100.0,
            agent_capital=1000,
            account_equity=1000,
            reasons=(),
            clipped=False,
            limits_fingerprint="x",
        )
        return self.exe.submit(d, intent, now)


def test_ack_is_not_a_fill() -> None:
    h = Harness()
    mo = h.submit(0.5)
    # deliver only the ACK (NEW) by stepping to the arrival + ack latency, then inspect before fills
    h.exe.on_order_update(
        OrderUpdate(
            ts=T0 + 60,
            account="challenge",
            client_order_id=mo.client_order_id,
            symbol="BTCUSDT",
            status=OrderStatus.NEW,
        )
    )
    assert mo.status is OrderStatus.NEW
    assert h.ledger["challenge"].agent_qty("A1", "BTCUSDT") == 0.0
    assert h.exe.pending_qty("challenge", "A1", "BTCUSDT") == pytest.approx(0.5)


def test_filled_status_without_fill_events_does_not_move_position() -> None:
    h = Harness()
    mo = h.submit(0.5)
    h.exe.on_order_update(
        OrderUpdate(
            ts=T0 + 60,
            account="challenge",
            client_order_id=mo.client_order_id,
            symbol="BTCUSDT",
            status=OrderStatus.FILLED,
            cum_qty=0.5,
        )
    )
    assert h.ledger["challenge"].agent_qty("A1", "BTCUSDT") == 0.0  # only executions move positions


def test_duplicate_fills_are_deduplicated() -> None:
    h = Harness(chaos=Chaos(duplicate_fill_prob=1.0))
    h.submit(0.5)
    h.sched.drain()
    assert h.exe.duplicate_fills >= 1
    assert h.ledger["challenge"].agent_qty("A1", "BTCUSDT") == pytest.approx(0.5)
    assert h.venue.position("challenge", "BTCUSDT") == pytest.approx(0.5)


def test_late_ack_after_fill_is_ignored() -> None:
    h = Harness(chaos=Chaos(late_ack_ms=500))
    mo = h.submit(0.5)
    h.sched.drain()
    assert mo.status is OrderStatus.FILLED
    assert h.exe.stale_updates >= 1  # the late NEW could not regress FILLED


def test_lost_ack_timeout_query_recovers() -> None:
    h = Harness(chaos=Chaos(drop_ack_prob=1.0))
    mo = h.submit(0.5)
    h.sched.drain()
    # fills still arrive (fill implies acceptance)
    assert mo.status is OrderStatus.FILLED and mo.acked_ts is not None


def test_order_never_reaches_venue_goes_unknown_then_resolves() -> None:
    h = Harness(ack_timeout_ms=1_000)

    class BlackHole:
        venue = "challenge"

        def __init__(self) -> None:
            self.queries = 0

        def submit(self, order: OrderRequest) -> None:
            pass

        def cancel(self, *a: Any) -> None:
            pass

        def query_order(self, *a: Any) -> None:
            self.queries += 1

        def query_positions(self, *a: Any) -> None:
            pass

        def query_open_orders(self, *a: Any) -> None:
            pass

    bh = BlackHole()
    h.exe.gateways["challenge"] = bh  # type: ignore[assignment]
    mo = h.submit(0.5)
    for i in range(1, 6):
        h.exe.check_timeouts(T0 + i * 1_000)
    assert bh.queries == 3
    assert mo.status is OrderStatus.UNKNOWN
    assert not h.health["challenge"].reconcile_ok  # new risk now blocked by the governor
    assert h.exe.pending_qty("challenge", "A1", "BTCUSDT") == pytest.approx(0.5)  # still counted as in flight
    # venue finally answers: the order never existed
    h.exe.on_order_update(
        OrderUpdate(
            ts=T0 + 9_000,
            account="challenge",
            client_order_id=mo.client_order_id,
            symbol="BTCUSDT",
            status=OrderStatus.REJECTED,
            reason="order_not_found",
        )
    )
    assert mo.status is OrderStatus.REJECTED
    assert h.health["challenge"].reconcile_ok
    assert h.exe.pending_qty("challenge", "A1", "BTCUSDT") == 0.0


def test_partial_fill_then_cancel_books_partial_position() -> None:
    h = Harness()
    mo = h.submit(3.0)  # only 5 levels x 1.0 but slippage cap 2% => 100.5,101.5 (102.5 > 102)
    h.sched.drain()
    assert mo.status is OrderStatus.CANCELED
    assert h.ledger["challenge"].agent_qty("A1", "BTCUSDT") == pytest.approx(mo.filled_qty)
    assert 0 < mo.filled_qty < 3.0


def test_orphan_fill_is_recorded_not_booked() -> None:
    h = Harness()
    h.exe.on_fill(
        FillEvent(
            ts=T0,
            account="challenge",
            client_order_id="MANUAL-1",
            exec_id="e-x",
            symbol="BTCUSDT",
            side=Side.BUY,
            qty=1.0,
            price=100.0,
            fee=0.0,
        )
    )
    assert h.exe.orphans and h.ledger["challenge"].agent_qty("A1", "BTCUSDT") == 0.0


def test_liquidation_fill_attributed_pro_rata() -> None:
    h = Harness()
    acct = h.ledger["challenge"]
    acct.pos("A1", "BTCUSDT").apply_fill(Side.BUY, 1.0, 100.0)
    acct.pos("A2", "BTCUSDT").apply_fill(Side.BUY, 3.0, 100.0)
    h.exe.on_fill(
        FillEvent(
            ts=T0,
            account="challenge",
            client_order_id="LIQ-1",
            exec_id="e-l",
            symbol="BTCUSDT",
            side=Side.SELL,
            qty=4.0,
            price=90.0,
            fee=0.0,
            is_liquidation=True,
        )
    )
    assert acct.agent_qty("A1", "BTCUSDT") == 0 and acct.agent_qty("A2", "BTCUSDT") == 0


def test_reconciliation_snapshot_matches_ledger() -> None:
    h = Harness()
    h.submit(0.7)
    h.sched.drain()
    h.venue.query_positions("challenge", T0 + 1_000)
    h.sched.drain()
    snaps = [e for e in h.received if isinstance(e, PositionSnapshot)]
    assert snaps and snaps[0].qty == pytest.approx(h.ledger["challenge"].net_qty("BTCUSDT"))
