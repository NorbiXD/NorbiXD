"""Buffered audit store.

The engine's hot path never blocks on the database: rows are appended to in-memory buffers and
written in batches by :meth:`AuditStore.flush` (called by the driver between events in replay,
and from a background thread in live mode). Upserted tables (orders, agents) keep only the
latest row per key between flushes.

Failure policy:

* rows are sanitized on the way in (NaN/Inf -> NULL, numpy scalars -> Python, integers beyond
  BIGINT -> NULL or a string inside JSON), so one bad float cannot poison a batch;
* a **transient** failure (connection lost, database locked/unavailable) puts the whole batch
  back and raises, so the engine halts new risk until the store recovers;
* any other failure (a row the database refuses) retries the batch row by row: good rows are
  written, refused rows go to ``dead_letters`` with the error, and trading continues.
"""

from __future__ import annotations

import logging
import math
import secrets
import threading
import time
from collections import defaultdict
from typing import Any

from sqlalchemy import Engine, Table, create_engine, event, exc, insert, select, text
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


_INT64 = 2**63 - 1
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    exc.OperationalError,
    exc.InterfaceError,
    exc.DisconnectionError,
    exc.TimeoutError,
)


class RunExistsError(RuntimeError):
    pass


def new_run_id(prefix: str) -> str:
    """Unique default run id: two runs started in the same second must not share a trail."""
    return f"{prefix}-{int(time.time())}-{secrets.token_hex(3)}"


def _clean(v: Any, nested: bool) -> Any:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if hasattr(v, "item") and not isinstance(v, (int, float)):  # numpy scalar
        try:
            v = v.item()
        except (TypeError, ValueError):
            return repr(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, int):
        if -_INT64 <= v <= _INT64:
            return v
        return str(v) if nested else None
    if isinstance(v, dict):
        return {str(k): _clean(x, True) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x, True) for x in v]
    return v


def sanitize(row: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``row`` that every supported database accepts value-wise."""
    return {k: _clean(v, False) for k, v in row.items()}


class AuditStore:
    def __init__(self, url: str, run_id: str) -> None:
        self.url = url
        self.run_id = run_id
        self.engine = make_engine(url)
        schema.metadata.create_all(self.engine)
        self._rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._upserts: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self.rows_written = 0
        self.failures = 0
        self.dead_lettered = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------ writes
    def assert_new_run(self) -> None:
        """Refuse to start a run whose id already has an audit trail (it would be overwritten)."""
        if self.one("runs", run_id=self.run_id) is not None:
            raise RunExistsError(f"run_id {self.run_id!r} already exists in {self.url}; pick a new one")

    def add(self, table: str, row: dict[str, Any]) -> None:
        row = sanitize({"run_id": self.run_id, **row})  # a copy: callers may keep references
        with self._lock:
            self._rows[table].append(row)

    def upsert(self, table: str, row: dict[str, Any]) -> None:
        row = sanitize(row if table == "genomes" else {"run_id": self.run_id, **row})
        key = tuple(row[k] for k in schema.UPSERT_KEYS[table])
        with self._lock:
            self._upserts[table][key] = row

    def pending(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._rows.values()) + sum(len(v) for v in self._upserts.values())

    def flush(self) -> int:
        """Write buffered rows in one transaction. On failure the batch is put back (nothing is
        silently dropped) and the error propagates so the caller can halt new risk."""
        with self._flush_lock:
            with self._lock:
                rows, self._rows = self._rows, defaultdict(list)
                ups, self._upserts = self._upserts, defaultdict(dict)
            n = 0
            try:
                with self.engine.begin() as conn:
                    for name, batch in rows.items():
                        if batch:
                            conn.execute(insert(schema.metadata.tables[name]), batch)
                            n += len(batch)
                    for name, by_key in ups.items():
                        if by_key:
                            tbl = schema.metadata.tables[name]
                            for row in by_key.values():
                                conn.execute(self._upsert_stmt(tbl, row))
                            n += len(by_key)
            except TRANSIENT_ERRORS as e:
                self._requeue(rows, ups)
                self.failures += 1
                self.last_error = repr(e)
                raise
            except Exception as e:
                self.last_error = repr(e)
                if not self._alive():
                    # the database itself is unreachable, whatever the exception type
                    self._requeue(rows, ups)
                    self.failures += 1
                    raise
                # the database refused some row: isolate it instead of halting on it forever
                log.warning("audit batch refused (%r); retrying row by row", e)
                n = self._flush_rows(rows, ups)
            self.rows_written += n
            return n

    def _alive(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def _requeue(
        self, rows: dict[str, list[dict[str, Any]]], ups: dict[str, dict[tuple[Any, ...], dict[str, Any]]]
    ) -> None:
        with self._lock:  # ahead of anything buffered meanwhile
            for name, batch in rows.items():
                self._rows[name] = batch + self._rows[name]
            for name, by_key in ups.items():
                merged = dict(by_key)
                merged.update(self._upserts[name])
                self._upserts[name] = merged

    def _flush_rows(
        self, rows: dict[str, list[dict[str, Any]]], ups: dict[str, dict[tuple[Any, ...], dict[str, Any]]]
    ) -> int:
        """One transaction per row; refused rows are dead-lettered. A transient error midway
        requeues everything not yet written and raises."""
        work: list[tuple[str, bool, tuple[Any, ...] | None, dict[str, Any]]] = []
        for name, batch in rows.items():
            work += [(name, False, None, r) for r in batch]
        for name, by_key in ups.items():
            work += [(name, True, k, r) for k, r in by_key.items()]
        n = 0
        for i, (name, is_up, _key, row) in enumerate(work):
            tbl = schema.metadata.tables[name]
            try:
                with self.engine.begin() as conn:
                    conn.execute(self._upsert_stmt(tbl, row) if is_up else insert(tbl).values(**row))
                n += 1
            except Exception as e:
                if not isinstance(e, TRANSIENT_ERRORS) and self._dead_letter(name, row, e):
                    continue
                # transient, or not even the dead letter could be written: the store is down
                left_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
                left_ups: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
                for name2, up2, key2, row2 in work[i:]:
                    if up2:
                        assert key2 is not None
                        left_ups[name2][key2] = row2
                    else:
                        left_rows[name2].append(row2)
                self._requeue(left_rows, left_ups)
                self.failures += 1
                self.last_error = repr(e)
                self.rows_written += n
                raise
        return n

    def _dead_letter(self, table: str, row: dict[str, Any], error: BaseException) -> bool:
        log.error("audit row refused by %s, dead-lettering: %r", table, error)
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(schema.dead_letters).values(
                        run_id=self.run_id,
                        table_name=table,
                        error=repr(error)[:2000],
                        payload=repr(row)[:20000],
                    )
                )
        except Exception:
            log.exception("could not write dead letter for %s", table)
            return False
        self.dead_lettered += 1
        return True

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
