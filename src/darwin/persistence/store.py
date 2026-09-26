"""Buffered audit store.

The engine's hot path never blocks on the database: rows are appended to in-memory buffers and
written in batches by :meth:`AuditStore.flush` (called by the driver between events in replay,
and from a background thread in live mode). Upserted tables (orders, agents) keep only the
latest row per key between flushes.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

from sqlalchemy import Engine, Table, create_engine, event, insert, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.pool import StaticPool

from darwin.persistence import schema

log = logging.getLogger(__name__)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
        eng = create_engine(url, **kwargs)

        @event.listens_for(eng, "connect")
        def _pragma(dbapi_conn: Any, _rec: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        return eng
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return create_engine(url, pool_pre_ping=True)


class AuditStore:
    def __init__(self, url: str, run_id: str) -> None:
        self.url = url
        self.run_id = run_id
        self.engine = make_engine(url)
        schema.metadata.create_all(self.engine)
        self._rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._upserts: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
        self._lock = threading.Lock()
        self.rows_written = 0

    # ------------------------------------------------------------------ writes
    def add(self, table: str, row: dict[str, Any]) -> None:
        row = {"run_id": self.run_id, **row}  # copy: callers may keep references (e.g. read models)
        with self._lock:
            self._rows[table].append(row)

    def upsert(self, table: str, row: dict[str, Any]) -> None:
        if table != "genomes":
            row = {"run_id": self.run_id, **row}
        key = tuple(row[k] for k in schema.UPSERT_KEYS[table])
        with self._lock:
            self._upserts[table][key] = row

    def pending(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._rows.values()) + sum(len(v) for v in self._upserts.values())

    def flush(self) -> int:
        with self._lock:
            rows, self._rows = self._rows, defaultdict(list)
            ups, self._upserts = self._upserts, defaultdict(dict)
        n = 0
        with self.engine.begin() as conn:
            for name, batch in rows.items():
                if batch:
                    tbl = schema.metadata.tables[name]
                    conn.execute(insert(tbl), batch)
                    n += len(batch)
            for name, by_key in ups.items():
                if by_key:
                    tbl = schema.metadata.tables[name]
                    for row in by_key.values():
                        conn.execute(self._upsert_stmt(tbl, row))
                    n += len(by_key)
        self.rows_written += n
        return n

    def _upsert_stmt(self, tbl: Table, row: dict[str, Any]) -> Any:
        keys = schema.UPSERT_KEYS[tbl.name]
        dialect = self.engine.dialect.name
        mod: Any = postgresql if dialect == "postgresql" else sqlite
        stmt = mod.insert(tbl).values(**row)
        update = {k: stmt.excluded[k] for k in row if k not in keys}
        return stmt.on_conflict_do_update(index_elements=list(keys), set_=update)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.engine.dispose()

    # ------------------------------------------------------------------ reads
    def query(self, table: str, **where: Any) -> list[dict[str, Any]]:
        tbl = schema.metadata.tables[table]
        stmt = select(tbl)
        for k, v in where.items():
            stmt = stmt.where(tbl.c[k] == v)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def one(self, table: str, **where: Any) -> dict[str, Any] | None:
        rows = self.query(table, **where)
        return rows[0] if rows else None
