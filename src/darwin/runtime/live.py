"""Asynchronous live driver: WebSocket feeds / paced synthetic data -> the same DarwinEngine.

The engine stays single-threaded and synchronous. This driver owns the event loop and:

* collects events from feeds (Bybit public/private streams, a paced synthetic market) through
  one queue, re-stamping them so engine time is monotonic across sockets;
* schedules simulated-venue events (paper/shadow fills, latency) on a timer heap;
* fires bar-boundary and heartbeat timers from the clock;
* flushes the audit store off the event loop thread.

A :class:`Clock` abstracts wall time so ``Mode.SIM`` can run the synthetic market accelerated
(``speed`` x real time) with exactly the same code path as paper/live trading.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from darwin.core.events import (
    BookDelta,
    BookSnapshot,
    Event,
    FeedStatus,
    FundingSettlement,
    IntelligenceSignal,
    LiquidationEvent,
    TickerEvent,
    TimerEvent,
    TradeEvent,
)
from darwin.exchange.scheduler import EventTarget
from darwin.exchange.sim.venue import SimExchange
from darwin.replay.recorder import ParquetRecorder
from darwin.runtime.engine import DarwinEngine

log = logging.getLogger(__name__)

_MARKET_TYPES = (TradeEvent, BookSnapshot, BookDelta, TickerEvent, LiquidationEvent, FundingSettlement)


@dataclass
class Clock:
    """Maps wall time to engine time. speed=1, origin=None => real time."""

    speed: float = 1.0
    origin_ms: int | None = None
    _wall0: float = field(default_factory=time.monotonic)

    def now_ms(self) -> int:
        if self.origin_ms is None:
            return int(time.time() * 1000)
        return self.origin_ms + int((time.monotonic() - self._wall0) * 1000 * self.speed)

    def seconds_until(self, ts: int) -> float:
        return max(ts - self.now_ms(), 0) / 1000 / self.speed


class LiveDriver:
    def __init__(
        self,
        engine: DarwinEngine,
        venues: list[SimExchange],
        clock: Clock,
        heartbeat_ms: int = 1_000,
        flush_interval_s: float = 2.0,
        drain_timeout_ms: int = 60_000,
    ) -> None:
        self.engine = engine
        self.venues = venues
        self.clock = clock
        self.heartbeat_ms = heartbeat_ms
        self.flush_interval_s = flush_interval_s
        self.drain_timeout_ms = drain_timeout_ms
        self.queue: asyncio.Queue[tuple[Event, EventTarget | None]] = asyncio.Queue()
        self._heap: list[tuple[int, int, EventTarget, Event]] = []
        self._seq = itertools.count()
        self.stop = asyncio.Event()
        self.events = 0
        self.restamped = 0
        self.recorder: ParquetRecorder | None = None

    # Scheduler protocol (simulated venues) ------------------------------------
    def schedule(self, ts: int, event: Event, target: EventTarget) -> None:
        heapq.heappush(self._heap, (ts, next(self._seq), target, event))
        self.queue.put_nowait((TimerEvent(ts=ts, name="wake"), None))  # wake the consumer

    # Feed entry point ------------------------------------------------------------
    def push(self, event: Event) -> None:
        """Thread-unsafe by design: call from the event loop (feeds run on it)."""
        self.queue.put_nowait((event, None))

    def _restamp(self, ev: Event) -> Event:
        if ev.ts < self.engine.now:
            self.restamped += 1
            return ev.model_copy(update={"ts": self.engine.now})
        return ev

    def _dispatch(self, ev: Event) -> None:
        ev = self._restamp(ev)
        if self.recorder is not None:
            self.recorder.record(ev)
        if isinstance(ev, _MARKET_TYPES):
            for v in self.venues:
                v.on_market(ev)
        self.engine.handle(ev)
        self.events += 1

    def _run_due(self, now: int) -> None:
        eng = self.engine
        while True:
            t_heap = self._heap[0][0] if self._heap else None
            t_bar = eng.next_bar_ts if eng.next_bar_ts <= now else None
            if t_heap is not None and t_heap > now:
                t_heap = None
            if t_heap is None and t_bar is None:
                return
            if t_bar is not None and (t_heap is None or t_bar <= t_heap):
                eng.handle(TimerEvent(ts=t_bar, name="bar"))
            else:
                _, _s, target, ev = heapq.heappop(self._heap)
                ev = self._restamp(ev)
                if self.recorder is not None and isinstance(ev, IntelligenceSignal):
                    self.recorder.record(ev)
                target.handle(ev)
                self.events += 1

    async def _flusher(self) -> None:
        store = self.engine.store
        if store is None:
            return
        while not self.stop.is_set():
            await asyncio.sleep(self.flush_interval_s)
            if not store.pending():
                continue
            try:
                await asyncio.to_thread(store.flush)  # a failed batch is requeued by the store
            except Exception as e:  # keep flushing; the engine halts new risk meanwhile
                with self.engine.lock:
                    self.engine.audit_flush_failed(repr(e))
            else:
                with self.engine.lock:
                    self.engine.audit_flush_ok()

    async def run(self) -> float:
        eng = self.engine
        flusher = asyncio.create_task(self._flusher())
        next_hb = self.clock.now_ms() + self.heartbeat_ms
        ended_at: int | None = None
        try:
            while not self.stop.is_set():
                now = self.clock.now_ms()
                deadlines = [next_hb, eng.next_bar_ts]
                if self._heap:
                    deadlines.append(self._heap[0][0])
                timeout = self.clock.seconds_until(min(deadlines))
                item: tuple[Event, EventTarget | None] | None = None
                with contextlib.suppress(TimeoutError):
                    item = await asyncio.wait_for(self.queue.get(), timeout=min(timeout, 1.0))
                now = self.clock.now_ms()
                with eng.lock:
                    self._run_due(now)
                    if item is not None and not (isinstance(item[0], TimerEvent) and item[0].name == "wake"):
                        self._dispatch(item[0])
                    if now >= next_hb:
                        eng.handle(TimerEvent(ts=max(now, eng.now), name="heartbeat"))
                        next_hb = now + self.heartbeat_ms
                if eng.ended:
                    ended_at = ended_at or now
                    with eng.lock:
                        in_flight = (
                            bool(eng.execution.open_orders()) or bool(self._heap) or eng.flatten_pending()
                        )
                    if not in_flight:
                        break
                    if now - ended_at > self.drain_timeout_ms:
                        with eng.lock:
                            eng.flatten_incomplete()
                        break
        finally:
            self.stop.set()
            if self.recorder is not None:
                self.recorder.flush()
            flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flusher
        with eng.lock:
            return eng.finalize()


class SyntheticFeed:
    """Pushes a synthetic event stream into the driver, paced by the driver's clock."""

    def __init__(self, driver: LiveDriver, events: Iterable[Event]) -> None:
        self.driver = driver
        self.events = events

    async def run(self) -> None:
        self.driver.push(
            FeedStatus(ts=self.driver.clock.now_ms(), feed="public:synthetic", status="connected")
        )
        for ev in self.events:
            if self.driver.stop.is_set():
                return
            wait = self.driver.clock.seconds_until(ev.ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self.driver.push(ev)
