"""Local L2 order book with snapshot/delta sequencing.

Integrity rules
---------------
* A snapshot fully replaces the book and (re)establishes the sequence.
* A delta whose ``update_id`` is <= the last applied one is a duplicate/stale replay: ignored.
* A delta that skips ahead (gap) invalidates the book until the next snapshot. An invalid book is
  reported as stale, so the Risk Governor rejects new risk on that symbol until resync.
* A crossed book after an update (best bid >= best ask) is treated as corruption: invalid.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from darwin.core.events import BookDelta, BookSnapshot, Level


@dataclass
class OrderBook:
    symbol: str
    strict_sequence: bool = True
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    update_id: int | None = None
    valid: bool = False
    last_ts: int = 0
    gaps: int = 0
    duplicates: int = 0
    invalid_reason: str = "no snapshot"

    # ---------------------------------------------------------------- updates
    def apply_snapshot(self, ev: BookSnapshot) -> None:
        self.bids = {p: q for p, q in ev.bids if q > 0}
        self.asks = {p: q for p, q in ev.asks if q > 0}
        self.update_id = ev.update_id
        self.last_ts = ev.ts
        self.valid = True
        self.invalid_reason = ""
        self._check_crossed()

    def apply_delta(self, ev: BookDelta) -> bool:
        """Apply a delta. Returns False if the book is (now) invalid and needs a resync."""
        if self.update_id is None:
            self.invalid_reason = "delta before snapshot"
            self.valid = False
            return False
        if ev.update_id <= self.update_id:
            self.duplicates += 1
            return self.valid
        if self.strict_sequence and ev.update_id != self.update_id + 1:
            self.gaps += 1
            self.valid = False
            self.invalid_reason = f"sequence gap {self.update_id} -> {ev.update_id}"
            self.update_id = ev.update_id
            return False
        if not self.valid:
            # we are waiting for a snapshot; keep tracking ids but don't trust contents
            self.update_id = ev.update_id
            return False
        _apply_levels(self.bids, ev.bids)
        _apply_levels(self.asks, ev.asks)
        self.update_id = ev.update_id
        self.last_ts = ev.ts
        self._check_crossed()
        return self.valid

    def invalidate(self, reason: str) -> None:
        self.valid = False
        self.invalid_reason = reason

    def _check_crossed(self) -> None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None and bb >= ba:
            self.valid = False
            self.invalid_reason = f"crossed book {bb} >= {ba}"

    # ---------------------------------------------------------------- queries
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def mid(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread_bps(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (ba - bb) / ((ba + bb) / 2.0) * 1e4

    def top_bids(self, n: int) -> list[Level]:
        return [(p, self.bids[p]) for p in heapq.nlargest(n, self.bids)]

    def top_asks(self, n: int) -> list[Level]:
        return [(p, self.asks[p]) for p in heapq.nsmallest(n, self.asks)]

    def imbalance(self, n: int = 5) -> float | None:
        """(bid qty - ask qty) / (bid qty + ask qty) over the top ``n`` levels, in [-1, 1]."""
        b = sum(q for _, q in self.top_bids(n))
        a = sum(q for _, q in self.top_asks(n))
        if b + a <= 0:
            return None
        return (b - a) / (b + a)


def _apply_levels(side: dict[float, float], levels: tuple[Level, ...]) -> None:
    for price, qty in levels:
        if qty <= 0:
            side.pop(price, None)
        else:
            side[price] = qty
