from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterator
from typing import Any

import pytest

from darwin.config.challenge import ChallengeConfig, load_config
from darwin.core.events import BookSnapshot, Event, TickerEvent, TradeEvent
from darwin.core.types import Side
from darwin.exchange.scheduler import EventTarget

T0 = 1_700_000_000_000


def make_config(**overrides: Any) -> ChallengeConfig:
    base: dict[str, Any] = {
        "challenge": {"duration_hours": 12, "symbols": ["BTCUSDT", "ETHUSDT"], "bar_ms": 60_000},
        "risk": {"kill_switch_file": None},
        "evolution": {"population_size": 12, "generation_bars": 120, "min_trades": 3},
    }

    def merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    return load_config(None, merge(base, overrides))


@pytest.fixture
def cfg() -> ChallengeConfig:
    return make_config()


def book(
    symbol: str,
    ts: int,
    mid: float,
    update_id: int = 1,
    spread: float = 1.0,
    levels: int = 10,
    qty: float = 1.0,
    step: float = 1.0,
) -> BookSnapshot:
    bids = tuple((mid - spread / 2 - i * step, qty) for i in range(levels))
    asks = tuple((mid + spread / 2 + i * step, qty) for i in range(levels))
    return BookSnapshot(ts=ts, symbol=symbol, bids=bids, asks=asks, update_id=update_id)


def trade(
    symbol: str, ts: int, price: float, qty: float = 0.1, side: Side = Side.BUY, tid: str | None = None
) -> TradeEvent:
    return TradeEvent(
        ts=ts, symbol=symbol, price=price, qty=qty, taker_side=side, trade_id=tid or f"{symbol}-{ts}-{price}"
    )


def ticker(symbol: str, ts: int, mark: float, funding: float = 0.0001, oi: float = 1000.0) -> TickerEvent:
    return TickerEvent(
        ts=ts,
        symbol=symbol,
        mark_price=mark,
        last_price=mark,
        index_price=mark,
        funding_rate=funding,
        open_interest=oi,
    )


class ManualScheduler:
    """Minimal deterministic scheduler for unit tests."""

    def __init__(self) -> None:
        self.heap: list[tuple[int, int, EventTarget, Event]] = []
        self.seq = itertools.count()
        self.now = 0

    def schedule(self, ts: int, event: Event, target: EventTarget) -> None:
        heapq.heappush(self.heap, (ts, next(self.seq), target, event))

    def run_until(self, ts: int) -> None:
        while self.heap and self.heap[0][0] <= ts:
            t, _s, target, ev = heapq.heappop(self.heap)
            self.now = t
            target.handle(ev)

    def drain(self) -> None:
        self.run_until(2**62)


class Collector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def handle(self, event: Event) -> None:
        self.events.append(event)

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


def iter_events(evs: list[Event]) -> Iterator[Event]:
    yield from sorted(evs, key=lambda e: e.ts)
