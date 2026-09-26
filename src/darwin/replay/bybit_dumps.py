"""Loader for Bybit's public historical trade dumps (``public.bybit.com/trading/<SYM>/``).

CSV columns: ``timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,
foreignNotional`` with ``timestamp`` in (fractional) seconds. These dumps contain trades only, so
for replay we synthesise a minimal book and mark around the last trade (``spread_bps``,
``depth_notional``) at a fixed cadence. That is an explicit approximation: slippage estimates
from such replays are optimistic for size and should be cross-checked against recorded
paper-mode sessions (which contain the real order book).
"""

from __future__ import annotations

import csv
import gzip
import heapq
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from darwin.core.events import BookSnapshot, Event, TickerEvent, TradeEvent
from darwin.core.types import Side


def read_trade_dump(path: str | Path) -> Iterator[TradeEvent]:
    p = Path(path)
    raw = gzip.open(p, "rt") if p.suffix == ".gz" else open(p, newline="")  # noqa: SIM115
    with raw as f:
        for row in csv.DictReader(f):
            ts = round(float(row["timestamp"]) * 1000)
            yield TradeEvent(
                ts=ts,
                symbol=row["symbol"],
                price=float(row["price"]),
                qty=float(row["size"]),
                taker_side=Side(row["side"]),
                trade_id=row.get("trdMatchID") or f"{row['symbol']}-{ts}",
                exch_ts=ts,
            )


def with_synthetic_book(
    trades: Iterable[TradeEvent],
    tick: float,
    spread_bps: float = 1.0,
    depth_notional: float = 100_000.0,
    levels: int = 20,
    every_ms: int = 1_000,
) -> Iterator[Event]:
    """Interleave a synthetic book + ticker snapshot (at most every ``every_ms``) before trades."""
    last_emit: dict[str, int] = {}
    uid = 0
    for tr in trades:
        if tr.ts - last_emit.get(tr.symbol, -(10**15)) >= every_ms:
            last_emit[tr.symbol] = tr.ts
            uid += 1
            mid = tr.price
            half = max(tick, mid * spread_bps / 2e4)
            bb = math.floor((mid - half) / tick) * tick
            ba = max(math.ceil((mid + half) / tick) * tick, bb + tick)
            q = depth_notional / mid / levels
            yield TickerEvent(ts=tr.ts, symbol=tr.symbol, last_price=mid, mark_price=mid, index_price=mid)
            yield BookSnapshot(
                ts=tr.ts,
                symbol=tr.symbol,
                bids=tuple((round(bb - i * tick, 10), q) for i in range(levels)),
                asks=tuple((round(ba + i * tick, 10), q) for i in range(levels)),
                update_id=uid,
            )
        yield tr


def merge_streams(*streams: Iterable[Event]) -> Iterator[Event]:
    """Merge per-symbol streams into one time-ordered stream (stable within equal timestamps)."""
    return heapq.merge(*streams, key=lambda e: e.ts)
