"""High-volume event recording (Parquet) and loading (DuckDB) for replay and research.

Layout: ``<dir>/<run_id>/part-00000.parquet`` with a deliberately simple, schema-stable table:

    ts (int64) · seq (int64) · kind (string) · symbol (string, nullable) · payload (string, JSON)

``payload`` is the event's own JSON, so the loader can reconstruct the exact typed event. DuckDB
filters by time/kind/symbol without loading everything and returns rows in ``(ts, seq)`` order,
which is precisely the order the live engine processed them — a recorded paper/live session
replays decision-for-decision.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, TypeAdapter

from darwin.core.events import (
    BookDelta,
    BookSnapshot,
    Event,
    FundingSettlement,
    IntelligenceSignal,
    LiquidationEvent,
    TickerEvent,
    TradeEvent,
)

REPLAYABLE = Annotated[
    TradeEvent
    | BookSnapshot
    | BookDelta
    | TickerEvent
    | LiquidationEvent
    | FundingSettlement
    | IntelligenceSignal,
    Field(discriminator="kind"),
]
_ADAPTER: TypeAdapter[Any] = TypeAdapter(REPLAYABLE)
_SCHEMA = pa.schema(
    [
        ("ts", pa.int64()),
        ("seq", pa.int64()),
        ("kind", pa.string()),
        ("symbol", pa.string()),
        ("payload", pa.string()),
    ]
)


class ParquetRecorder:
    def __init__(self, directory: str | Path, run_id: str, flush_every: int = 50_000) -> None:
        self.dir = Path(directory) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._rows: list[tuple[int, int, str, str | None, str]] = []
        self._seq = 0
        self._part = len(list(self.dir.glob("part-*.parquet")))
        self.recorded = 0

    def record(self, ev: Event) -> None:
        if not isinstance(
            ev,
            (
                TradeEvent,
                BookSnapshot,
                BookDelta,
                TickerEvent,
                LiquidationEvent,
                FundingSettlement,
                IntelligenceSignal,
            ),
        ):
            return
        self._seq += 1
        symbol = getattr(ev, "symbol", None)
        self._rows.append((ev.ts, self._seq, ev.kind, symbol, ev.model_dump_json()))
        self.recorded += 1
        if len(self._rows) >= self.flush_every:
            self.flush()

    def record_all(self, events: Iterable[Event]) -> None:
        for e in events:
            self.record(e)
        self.flush()

    def flush(self) -> Path | None:
        if not self._rows:
            return None
        cols = list(zip(*self._rows, strict=True))
        table = pa.Table.from_arrays(
            [pa.array(c, type=f.type) for c, f in zip(cols, _SCHEMA, strict=True)], schema=_SCHEMA
        )
        path = self.dir / f"part-{self._part:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self._part += 1
        self._rows = []
        return path


def load_events(
    path: str | Path,
    kinds: tuple[str, ...] | None = None,
    symbols: tuple[str, ...] | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    batch: int = 20_000,
) -> Iterator[Event]:
    """Stream recorded events in processing order. ``path`` is a run directory or a glob."""
    p = Path(path)
    pattern = str(p / "*.parquet") if p.is_dir() else str(p)
    where = ["1=1"]
    params: list[Any] = []
    if kinds:
        where.append(f"kind IN ({','.join('?' * len(kinds))})")
        params += list(kinds)
    if symbols:
        where.append(f"(symbol IS NULL OR symbol IN ({','.join('?' * len(symbols))}))")
        params += list(symbols)
    if start_ts is not None:
        where.append("ts >= ?")
        params.append(start_ts)
    if end_ts is not None:
        where.append("ts < ?")
        params.append(end_ts)
    con = duckdb.connect()
    try:
        cur = con.execute(
            f"SELECT payload FROM read_parquet(?) WHERE {' AND '.join(where)} ORDER BY ts, seq",
            [pattern, *params],
        )
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for (payload,) in rows:
                yield _ADAPTER.validate_python(json.loads(payload))
    finally:
        con.close()
