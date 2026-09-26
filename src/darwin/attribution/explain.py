"""Decision reconstruction: "Why did Agent 184 long SOL at 14:03:21?"

Everything needed is persisted at decision time; this module only joins it back together:

    intent (features seen, signals seen, component scores, regime, model response)
      -> market snapshot the agent saw
      -> genome + lineage of the agent
      -> risk decisions (shadow + challenge) with every clip/reject reason
      -> orders -> fills (price, fee, slippage)
      -> round trip (entry/exit, MFE, MAE, PnL) and its share of the agent's fitness
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import and_, select

from darwin.persistence import schema
from darwin.persistence.store import AuditStore


def _rows(store: AuditStore, stmt: Any) -> list[dict[str, Any]]:
    with store.engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt)]


def find_intent(
    store: AuditStore, agent_id: str, symbol: str | None, at_ts: int, run_id: str | None = None
) -> str | None:
    """The latest intent by ``agent_id`` (on ``symbol``) at or before ``at_ts``."""
    t = schema.intents
    conds = [t.c.agent_id == agent_id, t.c.ts <= at_ts]
    if symbol:
        conds.append(t.c.symbol == symbol)
    if run_id:
        conds.append(t.c.run_id == run_id)
    stmt = select(t.c.intent_id).where(and_(*conds)).order_by(t.c.ts.desc(), t.c.intent_id.desc()).limit(1)
    rows = _rows(store, stmt)
    return rows[0]["intent_id"] if rows else None


def explain(store: AuditStore, intent_id: str) -> dict[str, Any] | None:
    intent = store.one("intents", intent_id=intent_id)
    if intent is None:
        return None
    run_id = intent["run_id"]
    snapshot = store.one("snapshots", snapshot_id=intent["snapshot_ref"]) if intent["snapshot_ref"] else None
    genome = store.one("genomes", genome_id=intent["genome_id"])
    agent = store.one("agents", run_id=run_id, agent_id=intent["agent_id"])
    lineage = _rows(
        store,
        select(schema.lineage)
        .where(and_(schema.lineage.c.run_id == run_id, schema.lineage.c.agent_id == intent["agent_id"]))
        .order_by(schema.lineage.c.ts),
    )
    decisions = store.query("risk_decisions", intent_id=intent_id)
    orders = store.query("orders", intent_id=intent_id)
    fills: list[dict[str, Any]] = []
    for o in orders:
        fills += store.query("fills", client_order_id=o["client_order_id"])
    t = schema.trades
    trades = _rows(
        store,
        select(t).where(
            and_(t.c.run_id == run_id, (t.c.entry_intent_id == intent_id) | (t.c.exit_intent_id == intent_id))
        ),
    )
    fitness_rows = _rows(
        store,
        select(schema.fitness)
        .where(
            and_(
                schema.fitness.c.run_id == run_id,
                schema.fitness.c.agent_id == intent["agent_id"],
                schema.fitness.c.ts >= intent["ts"],
            )
        )
        .order_by(schema.fitness.c.ts)
        .limit(1),
    )
    contribution = None
    if fitness_rows and trades:
        # share of the agent's windowed net return explained by the trade(s) this intent opened
        shadow = [tr for tr in trades if str(tr["account_id"]).startswith("shadow:")]
        fr = fitness_rows[0]
        net_ret = fr["metrics"].get("net_return") if isinstance(fr["metrics"], dict) else None
        if shadow and net_ret:
            contribution = {
                "trade_return_on_equity": sum(tr["return_on_equity"] for tr in shadow),
                "window_net_return": net_ret,
                "fitness_after": fr["fitness"],
                "generation": fr["generation"],
            }
    return {
        "question": (
            f"Why did {intent['agent_id']} go {intent['direction']} {intent['symbol']} at {intent['ts']}?"
        ),
        "intent": intent,
        "knew": {
            "snapshot": snapshot,
            "features_seen": intent["features"],
            "signals_seen": intent["signals"],
            "components": intent["components"],
            "regime": intent["regime"],
            "provider": intent["provider"],
            "model_response": intent["model_response"],
        },
        "agent": agent,
        "genome": genome,
        "lineage": lineage,
        "risk_decisions": decisions,
        "orders": orders,
        "fills": fills,
        "trades": trades,
        "fitness_contribution": contribution,
    }


def lineage_tree(store: AuditStore, run_id: str) -> list[dict[str, Any]]:
    """Agents with parents/status for rendering a family tree."""
    rows = store.query("agents", run_id=run_id)
    return sorted(rows, key=lambda r: r["agent_id"])
