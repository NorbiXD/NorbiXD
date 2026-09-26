"""Flatten-until-flat: kill switch, breakers and challenge end must get the account flat even when
the venue rejects, partially fills or loses orders (QM iteration 2, critical blocker 1)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from darwin.core.events import Event, TimerEvent
from darwin.core.intent import TradeIntent
from darwin.core.types import OrderStatus, Urgency
from darwin.evolution.allocator import AllocationCandidate
from darwin.exchange.sim.venue import Chaos
from darwin.execution.orders import ExecutionGateway, OrderRequest
from darwin.market.synthetic import SyntheticMarket
from darwin.runtime.engine import CHALLENGE, CHALLENGE_VENUE, DarwinEngine
from darwin.runtime.replay import ReplayHandles, build_replay
from tests.conftest import T0, make_config

SYMBOLS = ["ETHUSDT", "SOLUSDT"]


@dataclass
class Step:
    """Run ``action`` at the first market event at/after ``at`` for which ``when`` holds."""

    at: int
    action: Callable[[DarwinEngine, int], None]
    when: Callable[[DarwinEngine], bool] = lambda _e: True


@dataclass
class Script:
    events: Iterator[Event]
    steps: list[Step]
    engine: DarwinEngine | None = None
    log: dict[str, int] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Event]:
        for ev in self.events:
            eng = self.engine
            assert eng is not None
            while self.steps and ev.ts >= self.steps[0].at and self.steps[0].when(eng):
                self.steps.pop(0).action(eng, ev.ts)
            yield ev


def fund_everyone(
    cands: list[AllocationCandidate], regime: str, rng: Any
) -> dict[str, float]:  # forced allocator: every alive agent trades the challenge book
    alive = [c for c in cands if c.status.value != "dead"]
    return {c.agent_id: 1.0 / max(len(alive), 1) for c in alive}


def funded_replay(
    steps: list[Step],
    hours: float = 4,
    seed: int = 21,
    chaos: Chaos | None = None,
    shadow_chaos: Chaos | None = None,
) -> tuple[ReplayHandles, Script]:
    cfg = make_config(
        challenge={"duration_hours": hours, "symbols": SYMBOLS},
        evolution={"population_size": 6, "generation_bars": 60, "min_trades": 2},
        allocator={"rebalance_bars": 5},
    )
    m = SyntheticMarket(
        symbols=tuple(SYMBOLS), start_ts=T0, duration_ms=cfg.duration_ms + 60_000, step_ms=5_000, seed=seed
    )
    script = Script(iter(m.events()), steps)
    h = build_replay(cfg, script, T0, chaos=chaos, shadow_chaos=shadow_chaos)
    script.engine = h.engine
    h.engine.allocator.allocate = fund_everyone  # type: ignore[method-assign]
    h.engine.execution.prune = lambda now, keep_ms=0: 0  # type: ignore[method-assign]  # keep orders to inspect
    return h, script


def has_positions(eng: DarwinEngine) -> bool:
    return len(eng.ledger[CHALLENGE].open_positions()) >= 2


def challenge_orders_since(eng: DarwinEngine, ts: int) -> list[Any]:
    return [o for o in eng.execution.orders.values() if o.account == CHALLENGE and o.created_ts >= ts]


def assert_everything_flat(h: ReplayHandles) -> None:
    eng = h.engine
    assert not eng.execution.open_orders(), "no working orders left"
    for acct_id, acct in eng.ledger.accounts.items():
        assert not acct.open_positions(), f"{acct_id} not flat: {acct.open_positions()}"
        venue = h.shadow if acct_id.startswith("shadow:") else h.challenge
        for s in SYMBOLS:
            assert venue.position(acct_id, s) == pytest.approx(0.0, abs=1e-9)


class FlakyGateway:
    """Wraps the challenge gateway: can drop every request (network loss) or shrink orders
    (the venue fills only part of each order)."""

    def __init__(self, inner: ExecutionGateway) -> None:
        self.inner = inner
        self.venue = inner.venue
        self.down = False
        self.fill_fraction = 1.0
        self.dropped = 0

    def submit(self, order: OrderRequest) -> None:
        if self.down:
            self.dropped += 1
            return
        if self.fill_fraction < 1.0:
            step = 0.01 if order.symbol == "ETHUSDT" else 0.1
            q = round(max(order.qty * self.fill_fraction // step, 1) * step, 10)
            if q < order.qty:
                order = order.model_copy(update={"qty": q})
        self.inner.submit(order)

    def cancel(self, account: str, client_order_id: str, symbol: str, ts: int) -> None:
        if not self.down:
            self.inner.cancel(account, client_order_id, symbol, ts)

    def query_order(self, account: str, client_order_id: str, symbol: str, ts: int) -> None:
        if not self.down:
            self.inner.query_order(account, client_order_id, symbol, ts)

    def query_positions(self, account: str, ts: int) -> None:
        if not self.down:
            self.inner.query_positions(account, ts)


def watch_flat(script: Script, key: str, after: int) -> Step:
    def mark(eng: DarwinEngine, ts: int) -> None:
        script.log[key] = ts

    return Step(after, mark, lambda e: e.is_flat(CHALLENGE))


# ----------------------------------------------------------------------------- tests


def test_kill_switch_flattens_through_a_minute_of_venue_rejects() -> None:
    h: ReplayHandles
    steps: list[Step] = []
    h, script = funded_replay(steps)
    t_kill = T0 + 90 * 60_000

    def kill(eng: DarwinEngine, ts: int) -> None:
        script.log["kill"] = ts
        script.log["positions"] = len(eng.ledger[CHALLENGE].open_positions())
        h.challenge.chaos = Chaos(reject_prob=1.0)
        eng.governor.engage_kill_switch()

    def recover(eng: DarwinEngine, ts: int) -> None:
        script.log["recover"] = ts
        script.log["not_flat_before_recovery"] = int(not eng.is_flat(CHALLENGE))
        h.challenge.chaos = Chaos()

    steps += [Step(t_kill, kill, has_positions)]
    steps += [Step(t_kill + 60_000, recover, lambda e: "kill" in script.log)]
    steps += [watch_flat(script, "flat", t_kill + 60_000)]
    h.driver.run()
    eng = h.engine
    assert script.log["positions"] >= 2
    assert script.log["not_flat_before_recovery"] == 1  # rejects really blocked the first exits
    assert "flat" in script.log and script.log["flat"] - script.log["recover"] <= 30_000
    after = challenge_orders_since(eng, script.log["kill"])
    assert after and not any(o.risk_increasing for o in after), "no new risk after the kill"
    retried = [o for o in after if o.intent_reason == "kill_switch"]
    assert len(retried) > script.log["positions"], "rejected exits were re-issued"
    assert eng.stats["sys:flatten_complete"] >= 1 and eng.stats["sys:flatten_incomplete"] == 0
    assert_everything_flat(h)


def test_partial_exit_fills_are_retried_until_flat() -> None:
    steps: list[Step] = []
    h, script = funded_replay(steps)
    flaky = FlakyGateway(h.engine.gateways[CHALLENGE_VENUE])
    h.engine.gateways[CHALLENGE_VENUE] = flaky
    t_kill = T0 + 90 * 60_000

    def kill(eng: DarwinEngine, ts: int) -> None:
        script.log["kill"] = ts
        flaky.fill_fraction = 0.3  # each exit only gets ~30% filled
        eng.governor.engage_kill_switch()

    steps += [Step(t_kill, kill, has_positions), watch_flat(script, "flat", t_kill)]
    h.driver.run()
    eng = h.engine
    exits = [o for o in challenge_orders_since(eng, script.log["kill"]) if o.intent_reason == "kill_switch"]
    by_key: dict[tuple[str, str], int] = {}
    for o in exits:
        by_key[(o.agent_id, o.symbol)] = by_key.get((o.agent_id, o.symbol), 0) + 1
    assert max(by_key.values()) >= 2, "a partially filled exit was followed by another"
    assert script.log["flat"] - script.log["kill"] <= 60_000
    assert_everything_flat(h)


def test_lost_exit_orders_become_unknown_then_resolve_and_flatten() -> None:
    steps: list[Step] = []
    h, script = funded_replay(steps)
    flaky = FlakyGateway(h.engine.gateways[CHALLENGE_VENUE])
    h.engine.gateways[CHALLENGE_VENUE] = flaky
    t_kill = T0 + 90 * 60_000

    def kill(eng: DarwinEngine, ts: int) -> None:
        script.log["kill"] = ts
        flaky.down = True
        eng.governor.engage_kill_switch()

    def check_unknown(eng: DarwinEngine, ts: int) -> None:
        script.log["unknown"] = sum(o.status is OrderStatus.UNKNOWN for o in eng.execution.open_orders())
        script.log["recover"] = ts
        flaky.down = False

    steps += [Step(t_kill, kill, has_positions)]
    steps += [Step(t_kill + 45_000, check_unknown, lambda e: "kill" in script.log)]
    steps += [watch_flat(script, "flat", t_kill + 45_000)]
    h.driver.run()
    assert flaky.dropped >= 2 and script.log["unknown"] >= 2
    # an UNKNOWN exit may have executed, so it blocks a second exit until the venue answers
    # (re-queried every 10 ACK timeouts); then the retry gets the account flat
    assert script.log["flat"] - script.log["recover"] <= 90_000
    assert h.engine.health[CHALLENGE_VENUE].reconcile_ok
    assert_everything_flat(h)


def test_breaker_trip_flattens_until_flat_under_rejects_and_blocks_new_risk() -> None:
    steps: list[Step] = []
    h, script = funded_replay(steps)
    t_trip = T0 + 90 * 60_000

    def trip(eng: DarwinEngine, ts: int) -> None:
        script.log["trip"] = ts
        h.challenge.chaos = Chaos(reject_prob=1.0)
        eng.ledger[CHALLENGE].peak_equity = 10 * eng.ledger[CHALLENGE].last_equity

    def recover(eng: DarwinEngine, ts: int) -> None:
        script.log["recover"] = ts
        h.challenge.chaos = Chaos()

    steps += [Step(t_trip, trip, has_positions)]
    steps += [Step(t_trip + 30_000, recover, lambda e: "trip" in script.log)]
    steps += [watch_flat(script, "flat", t_trip + 30_000)]
    h.driver.run()
    eng = h.engine
    assert eng.governor.breaker(CHALLENGE) is not None and eng.stats["sys:circuit_breaker"] == 1
    assert script.log["flat"] - script.log["recover"] <= 30_000
    after = challenge_orders_since(eng, script.log["trip"])
    assert after and not any(o.risk_increasing for o in after)
    assert_everything_flat(h)


def test_challenge_end_under_chaos_leaves_every_account_flat() -> None:
    chaos = Chaos(reject_prob=0.3, drop_ack_prob=0.2, duplicate_fill_prob=0.1, late_ack_ms=300)
    h, _script = funded_replay([], hours=3, seed=5, chaos=chaos, shadow_chaos=chaos)
    h.driver.run()
    eng = h.engine
    assert eng.stats["orders"] > 30 and eng.ended
    assert eng.stats["sys:flatten_incomplete"] == 0 and eng.stats["sys:flatten_complete"] == 1
    assert_everything_flat(h)


def test_kill_cancels_a_resting_entry_and_exits_are_not_blocked_by_it() -> None:
    cfg = make_config(challenge={"symbols": ["ETHUSDT"]})
    h = build_replay(cfg, iter([]), T0)
    eng = h.engine
    from tests.conftest import book, ticker, trade

    for ev in (book("ETHUSDT", T0 + 1, 3000.0, qty=50.0), ticker("ETHUSDT", T0 + 1, 3000.0)):
        h.driver._dispatch_market(ev)
    h.driver._dispatch_market(trade("ETHUSDT", T0 + 2, 3000.0, tid="x"))
    eng.now = T0 + 2
    eng.marks = {"ETHUSDT": 3000.0}
    eng.weights = {"A0001": 0.5}
    passive = TradeIntent(
        intent_id="I-passive",
        ts=eng.now,
        agent_id="A0001",
        genome_id="G",
        symbol="ETHUSDT",
        target_exposure=1.0,
        confidence=0.6,
        reason="entry",
        urgency=Urgency.PASSIVE,
        stop_loss_pct=0.02,
    )
    eng._route(passive, only=CHALLENGE)
    (resting,) = [o for o in eng.execution.open_orders() if o.account == CHALLENGE]
    assert resting.risk_increasing
    _pump(h, T0 + 500)
    assert resting.status is OrderStatus.NEW
    eng.governor.engage_kill_switch()
    eng.handle(TimerEvent(ts=T0 + 1_000, name="heartbeat"))
    _pump(h, T0 + 2_000)
    assert resting.status is OrderStatus.CANCELED and resting.reason == "cancelled_by_user"
    assert eng.is_flat(CHALLENGE)


def _pump(h: ReplayHandles, until: int) -> None:
    import heapq

    while h.driver._heap and h.driver._heap[0][0] <= until:
        _t, _p, _s, target, ev = heapq.heappop(h.driver._heap)
        target.handle(ev)


def test_stop_loss_exit_goes_through_while_an_increasing_order_is_working() -> None:
    from tests.conftest import book, ticker, trade

    cfg = make_config(challenge={"symbols": ["ETHUSDT"]})
    h = build_replay(cfg, iter([]), T0)
    eng = h.engine
    for ev in (book("ETHUSDT", T0 + 1, 3000.0, qty=50.0), ticker("ETHUSDT", T0 + 1, 3000.0)):
        h.driver._dispatch_market(ev)
    h.driver._dispatch_market(trade("ETHUSDT", T0 + 2, 3000.0, tid="a"))
    eng.now = T0 + 2
    eng.marks = {"ETHUSDT": 3000.0}
    eng.weights = {"A0001": 0.5}

    def intent(iid: str, target: float, urgency: Urgency) -> TradeIntent:
        return TradeIntent(
            intent_id=iid,
            ts=eng.now,
            agent_id="A0001",
            genome_id="G",
            symbol="ETHUSDT",
            target_exposure=target,
            confidence=0.6,
            reason="entry",
            urgency=urgency,
            stop_loss_pct=0.02,
        )

    eng._route(intent("I-1", 0.5, Urgency.MARKET), only=CHALLENGE)
    _pump(h, T0 + 1_000)
    held = eng.ledger[CHALLENGE].agent_qty("A0001", "ETHUSDT")
    assert held > 0
    eng.now = T0 + 1_000
    eng._route(intent("I-2", 1.0, Urgency.PASSIVE), only=CHALLENGE)  # scale-in rests on the bid
    (entry,) = eng.execution.open_orders()
    assert entry.risk_increasing and entry.request.tif.value == "PostOnly"
    _pump(h, T0 + 1_400)
    assert entry.status is OrderStatus.NEW  # resting at the venue
    # price gaps 3% below entry: the stop must fire despite the working entry
    h.driver._dispatch_market(book("ETHUSDT", T0 + 1_500, 2905.0, qty=50.0))
    h.driver._dispatch_market(trade("ETHUSDT", T0 + 1_500, 2910.0, tid="b"))
    stops = [o for o in eng.execution.orders.values() if o.intent_reason == "stop_loss"]
    assert len(stops) == 1 and not stops[0].risk_increasing
    assert entry.cancel_requested_ts == T0 + 1_500
    # the falling price trades through the resting bid, so the entry fills before the cancel
    # lands (a real race); the residual is stopped out on the next tick after the retry pause
    _pump(h, T0 + 2_000)
    assert entry.status.value == "Filled"
    assert eng.ledger[CHALLENGE].agent_qty("A0001", "ETHUSDT") > 0
    h.driver._dispatch_market(trade("ETHUSDT", T0 + 2_600, 2908.0, tid="c"))
    _pump(h, T0 + 4_000)
    assert len([o for o in eng.execution.orders.values() if o.intent_reason == "stop_loss"]) == 2
    assert eng.ledger[CHALLENGE].agent_qty("A0001", "ETHUSDT") == pytest.approx(0.0, abs=1e-12)
    assert not eng.execution.open_orders()


def test_lost_terminal_updates_and_fills_of_exits_are_recovered_by_query_and_flatten_completes() -> None:
    """QM iteration 3, M1: an exit whose ACK arrived but whose Filled update *and* executions were
    lost used to stay "New" forever, blocking further exits and hiding the position mismatch."""
    from darwin.core.events import FillEvent, OrderUpdate

    steps: list[Step] = []
    h, script = funded_replay(steps)
    lossy: set[str] = set()
    dropped = {"n": 0}
    real_emit = h.challenge._emit

    def emit(ts: int, ev: Any) -> None:
        cid = getattr(ev, "client_order_id", None)
        is_final = isinstance(ev, FillEvent) or (isinstance(ev, OrderUpdate) and ev.status.terminal)
        if cid in lossy and is_final and ev.ts < script.log.get("kill", 0) + 3_000:
            dropped["n"] += 1  # the venue executed it; only the reports are lost
            return
        real_emit(ts, ev)

    h.challenge._emit = emit  # type: ignore[method-assign]
    real_submit = h.challenge.submit

    def submit(order: OrderRequest) -> None:
        if "kill" in script.log and not any(o.startswith(order.client_order_id[:-8]) for o in lossy):
            lossy.add(order.client_order_id)  # the first exit per agent loses its final reports
        real_submit(order)

    h.challenge.submit = submit  # type: ignore[method-assign]
    t_kill = T0 + 90 * 60_000

    def kill(eng: DarwinEngine, ts: int) -> None:
        script.log["kill"] = ts
        script.log["positions"] = len(eng.ledger[CHALLENGE].open_positions())
        eng.governor.engage_kill_switch()

    steps += [Step(t_kill, kill, has_positions), watch_flat(script, "flat", t_kill)]
    h.driver.run()
    eng = h.engine
    assert lossy and dropped["n"] >= len(lossy)
    stuck = [eng.execution.orders[c] for c in lossy]
    assert all(o.status.terminal and not o.open for o in stuck), [o.status for o in stuck]
    assert all(o.queries >= 1 for o in stuck)  # recovered by querying the venue
    # executions were replayed by the query, so the exits counted once: no double exit
    assert script.log["flat"] - script.log["kill"] <= 60_000
    assert eng.stats["sys:flatten_incomplete"] == 0
    assert_everything_flat(h)


def test_a_position_that_only_the_venue_holds_is_adopted_and_closed_under_kill() -> None:
    """QM iteration 3, M2: a fill the engine cannot attribute (manual trade, lost state) left a
    venue position that flatten never closed."""
    from darwin.core.events import FillEvent
    from darwin.core.types import Side
    from darwin.runtime.engine import SYSTEM_AGENT

    steps: list[Step] = []
    h, script = funded_replay(steps)
    t_orphan = T0 + 80 * 60_000

    def manual_trade(eng: DarwinEngine, ts: int) -> None:
        price = eng.market["ETHUSDT"].ref_price() or 3_000.0
        h.challenge._apply_position(CHALLENGE, "ETHUSDT", Side.BUY, 0.05, price, 0.08)
        orphan = FillEvent(
            ts=ts + 50,
            account=CHALLENGE,
            client_order_id="manual-from-phone",
            exec_id="manual-1",
            symbol="ETHUSDT",
            side=Side.BUY,
            qty=0.05,
            price=price,
            fee=0.08,
        )
        h.challenge._emit(ts + 50, orphan)
        script.log["orphan"] = ts

    def kill(eng: DarwinEngine, ts: int) -> None:
        script.log["kill"] = ts
        eng.governor.engage_kill_switch()

    steps += [Step(t_orphan, manual_trade)]
    steps += [Step(t_orphan + 5 * 60_000, kill, lambda e: "orphan" in script.log)]
    steps += [watch_flat(script, "flat", t_orphan + 5 * 60_000)]
    h.driver.run()
    eng = h.engine
    assert eng.execution.orphans and eng.stats["sys:orphan_adopted"] == 1
    closes = [o for o in eng.execution.orders.values() if o.agent_id == SYSTEM_AGENT]
    assert closes and all(o.request.reduce_only and not o.risk_increasing for o in closes)
    assert script.log["flat"] - script.log["kill"] <= 3 * 60_000
    assert eng.stats["sys:reconcile_mismatch"] <= 2  # one event per episode, not per snapshot
    assert eng.stats["sys:flatten_incomplete"] == 0
    assert_everything_flat(h)
