"""Deterministic discrete-event replay driver.

Merges three sources in strict ``(ts, priority, seq)`` order:

1. the historical/synthetic market stream (priority 0 — at equal timestamps, market data is
   applied first, i.e. an order arriving at ``t`` sees the book as of ``t``),
2. events scheduled by simulated venues (order arrivals, ACKs, fills, funding) — priority 1,
3. heartbeat timers (timeouts, reconciliation) — priority 2.

Because the engine only knows time through event timestamps, running the same inputs twice
produces identical decisions, orders, fills and lineage — asserted by the test-suite.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from darwin.config.challenge import ChallengeConfig
from darwin.core.events import Event, TimerEvent
from darwin.exchange.scheduler import EventTarget
from darwin.exchange.sim.venue import Chaos, SimExchange
from darwin.market.state import MarketDataEvent
from darwin.persistence.store import AuditStore
from darwin.runtime.engine import CHALLENGE_VENUE, SHADOW_VENUE, DarwinEngine

_P_MARKET, _P_VENUE, _P_TIMER = 0, 1, 2


class ReplayDriver:
    def __init__(
        self,
        engine: DarwinEngine,
        market: Iterable[Event],
        venues: list[SimExchange],
        timer_ms: int = 5_000,
        flush_every: int = 20_000,
    ) -> None:
        self.engine = engine
        self.market: Iterator[Event] = iter(market)
        self.venues = venues
        self.timer_ms = timer_ms
        self.flush_every = flush_every
        self._heap: list[tuple[int, int, int, EventTarget, Event]] = []
        self._seq = itertools.count()
        self.events = 0
        self.last_ts = engine.start_ts

    # Scheduler protocol --------------------------------------------------------
    def schedule(self, ts: int, event: Event, target: EventTarget) -> None:
        heapq.heappush(self._heap, (ts, _P_VENUE, next(self._seq), target, event))

    def _dispatch_market(self, ev: Event) -> None:
        for v in self.venues:
            v.on_market(ev)  # type: ignore[arg-type]
        self.engine.handle(ev)

    def run(self) -> float:
        """Merge market data, venue events, bar boundaries and heartbeats in time order.

        Priorities at equal timestamps: bar boundary (-1) < market (0) < venue (1) < heartbeat (2).
        A bar ``[start, end)`` is therefore closed — and its decisions submitted — before any
        market event stamped ``end`` reaches a venue, so simulated fills can never use a book
        from after the order's arrival time.
        """
        eng = self.engine
        store = eng.store
        inf = 2**62
        next_timer = eng.start_ts + self.timer_ms
        stop_ts = eng.end_ts + eng.bar_ms
        pending = next(self.market, None)
        while pending is not None and pending.ts < stop_ts:
            t_mkt = pending.ts
            t_heap = self._heap[0][0] if self._heap else inf
            t_bar = eng.next_bar_ts
            choice = min((t_bar, -1), (t_mkt, 0), (t_heap, 1), (next_timer, 2))
            kind = choice[1]
            with eng.lock:
                if kind == -1:
                    eng.handle(TimerEvent(ts=t_bar, name="bar"))
                elif kind == 0:
                    self._dispatch_market(pending)
                    self.last_ts = pending.ts
                    pending = next(self.market, None)
                elif kind == 1:
                    _, _p, _s, target, ev = heapq.heappop(self._heap)
                    target.handle(ev)
                else:
                    eng.handle(TimerEvent(ts=next_timer, name="heartbeat"))
                    next_timer += self.timer_ms
            self.events += 1
            if store is not None and self.events % self.flush_every == 0:
                store.flush()
        # end of stream: close the final bars (triggers challenge end + flattening), then drain
        with eng.lock:
            eng.advance_to(max(self.last_ts, eng.end_ts))
            guard = 0
            while self._heap and guard < 1_000_000:
                t, _p, _s, target, ev = heapq.heappop(self._heap)
                target.handle(ev)
                self.events += 1
                guard += 1
                if not self._heap:
                    # let timers resolve any remaining timeouts, then drain again
                    hb = max(t, eng.now) + eng.cfg.exchange.order_ack_timeout_ms
                    eng.handle(TimerEvent(ts=hb, name="heartbeat"))
            return eng.finalize()


@dataclass
class ReplayHandles:
    engine: DarwinEngine
    driver: ReplayDriver
    shadow: SimExchange
    challenge: SimExchange


def build_replay(
    cfg: ChallengeConfig,
    market: Iterable[MarketDataEvent] | Iterable[Event],
    start_ts: int,
    store: AuditStore | None = None,
    run_id: str = "replay",
    chaos: Chaos | None = None,
    timer_ms: int = 5_000,
) -> ReplayHandles:
    engine = DarwinEngine(cfg, run_id=run_id, start_ts=start_ts, store=store)
    driver = ReplayDriver(engine, market, [], timer_ms=timer_ms)
    instruments = {s: cfg.instrument(s) for s in cfg.challenge.symbols}
    common = {
        "symbols": cfg.challenge.symbols,
        "fees": cfg.exchange.fees,
        "instruments": instruments,
        "sim": cfg.sim,
        "scheduler": driver,
        "engine": engine,
        "maintenance_margin_rate": cfg.risk.maintenance_margin_rate,
    }
    shadow = SimExchange(SHADOW_VENUE, seed_offset=1, **common)  # type: ignore[arg-type]
    challenge = SimExchange(CHALLENGE_VENUE, seed_offset=2, chaos=chaos, **common)  # type: ignore[arg-type]
    driver.venues = [shadow, challenge]
    engine.attach_gateway(SHADOW_VENUE, shadow)
    engine.attach_gateway(CHALLENGE_VENUE, challenge)
    engine.start()
    return ReplayHandles(engine=engine, driver=driver, shadow=shadow, challenge=challenge)
