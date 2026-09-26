"""FastAPI control & observability API (and the dashboard's backend).

Reads live state from the in-memory engine (under its lock) and history from the audit store.
The only write endpoint is the operator kill switch, protected by ``DARWIN_OPERATOR_TOKEN``
(disabled when the variable is unset). Nothing here can modify risk limits.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from darwin.attribution.analytics import signal_information, species_regime_performance
from darwin.attribution.explain import explain, find_intent
from darwin.runtime.engine import CHALLENGE, DarwinEngine, shadow_account
from darwin.signals.external import WebhookRejected, WebhookSignalFeed

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


class KillSwitchRequest(BaseModel):
    engage: bool


def _jsonable(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def create_app(engine: DarwinEngine, webhooks: dict[str, WebhookSignalFeed] | None = None) -> FastAPI:
    app = FastAPI(title="DARWIN", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    store = engine.store
    feeds = webhooks or {}

    def locked(fn: Any) -> Any:
        with engine.lock:
            return _jsonable(fn())

    def lineage_tree() -> list[dict[str, Any]]:
        evals = engine.population.last_evaluations
        out = []
        for a in engine.population.agents.values():
            e = evals.get(a.agent_id)
            out.append(
                {
                    "agent_id": a.agent_id,
                    "species": a.species,
                    "status": a.status.value,
                    "generation": a.generation,
                    "parents": list(a.parent_ids),
                    "born_ts": a.born_ts,
                    "died_ts": a.died_ts,
                    "death_reason": a.death_reason,
                    "fitness": e.adjusted_fitness if e else None,
                    "champion": a.agent_id == engine.population.champion_id,
                    "weight": engine.weights.get(a.agent_id, 0.0),
                }
            )
        return out

    def allocations() -> list[dict[str, Any]]:
        eq = engine.ledger[CHALLENGE].last_equity
        return [
            {
                "agent_id": aid,
                "weight": w,
                "capital": w * eq,
                "species": engine.population.agents[aid].species if aid in engine.population.agents else "?",
            }
            for aid, w in sorted(engine.weights.items(), key=lambda kv: -kv[1])
        ]

    def agent_detail(agent_id: str) -> dict[str, Any]:
        a = engine.population.agents.get(agent_id)
        if a is None:
            raise HTTPException(404, "unknown agent")
        e = engine.population.last_evaluations.get(agent_id)
        book = engine.population.books[agent_id]
        sacct = engine.ledger.accounts.get(shadow_account(agent_id))
        return {
            "agent_id": agent_id,
            "species": a.species,
            "status": a.status.value,
            "generation": a.generation,
            "parents": list(a.parent_ids),
            "strikes": a.strikes,
            "genome_id": a.genome.genome_id,
            "genome": a.genome.model_dump(mode="json"),
            "evaluation": e.to_record() if e else None,
            "shadow_equity_curve": list(zip(book.ts[-500:], book.equity[-500:], strict=True)),
            "shadow_equity": sacct.last_equity if sacct else None,
            "trades": len(book.trades),
            "lineage": [r for r in engine.recent_lineage if r["agent_id"] == agent_id],
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "now": engine.now, "ended": engine.ended}

    @app.get("/api/state")
    def state() -> Any:
        return locked(engine.state)

    @app.get("/api/leaderboard")
    def leaderboard() -> Any:
        return locked(engine.leaderboard)

    @app.get("/api/positions")
    def positions() -> Any:
        return locked(engine.positions)

    @app.get("/api/allocations")
    def get_allocations() -> Any:
        return locked(allocations)

    @app.get("/api/equity")
    def equity(limit: int = Query(2_000, le=20_000)) -> Any:
        return locked(lambda: list(engine.equity_curve)[-limit:])

    @app.get("/api/decisions")
    def decisions(limit: int = Query(100, le=300)) -> Any:
        return locked(lambda: list(engine.recent_decisions)[-limit:][::-1])

    @app.get("/api/lineage/recent")
    def lineage_recent(limit: int = Query(100, le=300)) -> Any:
        return locked(lambda: list(engine.recent_lineage)[-limit:][::-1])

    @app.get("/api/lineage/tree")
    def get_lineage_tree() -> Any:
        return locked(lineage_tree)

    @app.get("/api/agents/{agent_id}")
    def agent(agent_id: str) -> Any:
        return locked(lambda: agent_detail(agent_id))

    @app.get("/api/trades")
    def trades(limit: int = Query(100, le=1_000), account: str = "challenge") -> Any:
        if store is None:
            return []
        store.flush()
        rows = store.query("trades", account_id=account) if account else store.query("trades")
        rows.sort(key=lambda r: r["exit_ts"] or 0, reverse=True)
        return _jsonable(rows[:limit])

    @app.get("/api/explain/{intent_id}")
    def get_explain(intent_id: str) -> Any:
        if store is None:
            raise HTTPException(503, "no audit store configured")
        store.flush()
        ex = explain(store, intent_id, run_id=engine.run_id)
        if ex is None:
            raise HTTPException(404, "unknown intent")
        return _jsonable(ex)

    @app.get("/api/why")
    def why(agent: str, at: int, symbol: str | None = None) -> Any:
        """ "Why did <agent> trade <symbol> at <ts ms>?" -> the explanation of the latest intent."""
        if store is None:
            raise HTTPException(503, "no audit store configured")
        store.flush()
        iid = find_intent(store, agent, symbol, at, run_id=engine.run_id)
        if iid is None:
            raise HTTPException(404, "no decision found")
        return _jsonable(explain(store, iid, run_id=engine.run_id))

    @app.get("/api/signals/recent")
    def signals_recent(limit: int = Query(50, le=200)) -> Any:
        return locked(lambda: list(engine.recent_signals)[-limit:][::-1])

    @app.post("/api/signals/{source}", status_code=202)
    async def ingest_signal(source: str, request: Request) -> Any:
        """Authenticated external signal webhook (HMAC-SHA256 over the raw body)."""
        feed = feeds.get(source)
        if feed is None:
            raise HTTPException(404, "unknown or disabled signal source")
        body = await request.body()
        try:
            sig = feed.ingest(body, request.headers.get("X-Darwin-Signature"))
        except WebhookRejected as e:
            raise HTTPException(e.status, e.reason) from e
        return {"accepted": sig.signal_id, "available_at": sig.ts}

    @app.get("/api/attribution/species")
    def attribution_species() -> Any:
        if store is None:
            return []
        store.flush()
        return _jsonable(species_regime_performance(store, engine.run_id))

    @app.get("/api/attribution/signals")
    def attribution_signals() -> Any:
        if store is None:
            return []
        store.flush()
        return _jsonable(signal_information(store, engine.run_id))

    @app.post("/api/kill-switch")
    def kill_switch(req: KillSwitchRequest, authorization: str | None = Header(default=None)) -> Any:
        token = os.environ.get("DARWIN_OPERATOR_TOKEN")
        if not token:
            raise HTTPException(403, "operator endpoint disabled (DARWIN_OPERATOR_TOKEN unset)")
        if authorization != f"Bearer {token}":
            raise HTTPException(401, "bad token")
        with engine.lock:
            if req.engage:
                engine.governor.engage_kill_switch()
            else:
                engine.governor.release_kill_switch()
            return {"kill_switch": engine.governor.kill_switch_active()}

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                payload = locked(
                    lambda: {
                        "state": engine.state(),
                        "leaderboard": engine.leaderboard(),
                        "allocations": allocations(),
                        "decisions": list(engine.recent_decisions)[-25:][::-1],
                        "lineage": list(engine.recent_lineage)[-25:][::-1],
                        "equity": list(engine.equity_curve)[-600:],
                        "positions": engine.positions(),
                        "signals": list(engine.recent_signals)[-15:][::-1],
                    }
                )
                await ws.send_json(payload)
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            with contextlib.suppress(Exception):
                await ws.close()

    @app.get("/")
    def index() -> Any:
        f = DASHBOARD_DIR / "index.html"
        if f.exists():
            return FileResponse(f)
        return JSONResponse({"message": "DARWIN API", "docs": "/api/docs"})

    return app
