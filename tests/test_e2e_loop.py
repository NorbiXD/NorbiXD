"""End-to-end: the definition of done, as executable assertions.

multiple agents receive identical market state, decide independently, execute through the
same controlled gateway, get measurable outcomes, are ranked; underperformers lose capital
and die; winners reproduce and mutate; a new generation competes; and the lineage and
reasoning trail are reconstructable from the database.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

from darwin.attribution.explain import explain, find_intent
from darwin.market.synthetic import SyntheticMarket
from darwin.persistence.store import AuditStore
from darwin.runtime.replay import ReplayHandles, build_replay
from tests.conftest import T0, make_config


def _cfg() -> Any:
    return make_config(
        challenge={"duration_hours": 24, "symbols": ["BTCUSDT", "ETHUSDT"]},
        sim={"seed": 3},
        evolution={"population_size": 12, "generation_bars": 120, "min_trades": 3, "eval_generations": 2},
        allocator={"rebalance_bars": 30},
    )


def _run(tmp: Path, name: str = "e2e") -> tuple[ReplayHandles, AuditStore, float]:
    cfg = _cfg()
    store = AuditStore(f"sqlite:///{tmp / (name + '.db')}", run_id=name)
    market = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=T0,
        duration_ms=cfg.duration_ms + 120_000,
        step_ms=5_000,
        seed=3,
    )
    h = build_replay(cfg, market.events(), T0, store=store, run_id=name)
    final = h.driver.run()
    store.flush()
    return h, store, final


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[ReplayHandles, AuditStore, float]:
    return _run(tmp_path_factory.mktemp("e2e"))


def test_agents_share_identical_market_state(run) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = run
    intents = [i for i in store.query("intents") if i["reason"] in ("entry", "exit", "flip")]
    by_bar: dict[tuple[int, str], set[str]] = defaultdict(set)
    agents_at_bar: dict[tuple[int, str], set[str]] = defaultdict(set)
    for i in intents:
        by_bar[(i["ts"], i["symbol"])].add(i["snapshot_ref"])
        agents_at_bar[(i["ts"], i["symbol"])].add(i["agent_id"])
    assert all(len(refs) == 1 for refs in by_bar.values()), "same bar/symbol => same snapshot for every agent"
    assert any(len(a) > 1 for a in agents_at_bar.values()), "several agents decided on the same bar"
    snap = store.one("snapshots", snapshot_id=intents[0]["snapshot_ref"])
    assert snap is not None and "bar" in snap["data"]


def test_independent_decisions(run) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = run
    intents = [i for i in store.query("intents") if i["reason"] == "entry"]
    assert len({i["agent_id"] for i in intents}) >= 6
    per_bar: dict[tuple[int, str], set[str]] = defaultdict(set)
    for i in intents:
        per_bar[(i["ts"], i["symbol"])].add(i["direction"])
    # agents looking at the same data sometimes disagree
    assert any(len(d) > 1 for d in per_bar.values())


def test_every_order_went_through_the_risk_governor(run) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = run
    decisions = {d["decision_id"]: d for d in store.query("risk_decisions")}
    intents = {i["intent_id"] for i in store.query("intents")}
    orders = store.query("orders")
    assert len(orders) > 100
    for o in orders:
        d = decisions[o["decision_id"]]
        assert d["approved"] and d["intent_id"] == o["intent_id"] and o["intent_id"] in intents
        assert abs(d["order_qty"]) == pytest.approx(o["qty"])
        assert d["limits_fingerprint"]


def test_measurable_outcomes(run) -> None:  # type: ignore[no-untyped-def]
    h, store, final = run
    trades = store.query("trades")
    assert len(trades) > 50
    t = trades[0]
    for k in ("net_pnl", "fees", "mfe_pct", "mae_pct", "entry_intent_id", "exit_reason"):
        assert k in t
    assert any(tr["exit_reason"] in ("stop_loss", "take_profit", "max_hold") for tr in trades)
    fills = store.query("fills")
    assert all(f["fee"] > 0 for f in fills if not f["is_liquidation"])
    assert final > 0 and h.engine.finalized
    run_row = store.one("runs", run_id="e2e")
    assert run_row is not None and run_row["final_equity"] == pytest.approx(final)


def test_ranking_death_reproduction_and_new_generation(run) -> None:  # type: ignore[no-untyped-def]
    h, store, _ = run
    fitness = store.query("fitness")
    gens = {f["generation"] for f in fitness}
    assert len(gens) >= 8
    lineage = store.query("lineage")
    kinds = Counter(r["kind"] for r in lineage)
    assert kinds["killed"] > 0, "underperformers must die"
    assert kinds["mutated"] + kinds["crossover"] > 0, "winners must reproduce"
    assert kinds["promoted"] > 0
    agents = {a["agent_id"]: a for a in store.query("agents")}
    children = [a for a in agents.values() if a["parent_ids"]]
    assert children
    # a new generation actually competes: children made decisions after birth
    child_ids = {c["agent_id"] for c in children}
    child_intents = [i for i in store.query("intents") if i["agent_id"] in child_ids]
    assert child_intents
    # population size is maintained
    alive = [a for a in h.engine.population.agents.values() if a.alive]
    assert len(alive) == h.engine.cfg.evolution.population_size
    # capital moved: allocations changed over time, and dead agents hold no capital
    allocs = store.query("allocations")
    assert len({a["agent_id"] for a in allocs}) >= 2
    dead = {a["agent_id"] for a in agents.values() if a["status"] == "dead"}
    assert not (dead & set(h.engine.weights))


def test_reasoning_trail_is_reconstructable(run) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = run
    # pick an entry of a *child* agent that produced a completed trade
    agents = {a["agent_id"]: a for a in store.query("agents")}
    trades = [t for t in store.query("trades") if t["account_id"].startswith("shadow:")]
    child_trades = [t for t in trades if agents[t["agent_id"]]["parent_ids"]]
    tr = (child_trades or trades)[0]
    iid = find_intent(store, tr["agent_id"], tr["symbol"], tr["entry_ts"])
    assert iid is not None
    ex = explain(store, tr["entry_intent_id"])
    assert ex is not None
    assert ex["knew"]["features_seen"], "what the agent saw is persisted"
    assert ex["knew"]["components"]
    assert ex["knew"]["snapshot"] is not None
    assert ex["genome"]["genome"]["terms"]
    assert ex["lineage"] and ex["lineage"][0]["kind"] in ("born", "mutated", "crossover")
    assert ex["risk_decisions"] and ex["orders"] and ex["fills"]
    assert any(t["trade_id"] == tr["trade_id"] for t in ex["trades"])
    if agents[tr["agent_id"]]["parent_ids"]:
        born = ex["lineage"][0]
        assert born["parent_ids"] and born["details"].get("mutations") is not None


def test_replay_is_deterministic(run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _h1, store1, final1 = run
    _h2, store2, final2 = _run(tmp_path, "e2e_repeat")
    assert final1 == final2
    i1 = [(i["intent_id"], i["agent_id"], i["target_exposure"]) for i in store1.query("intents")]
    i2 = [(i["intent_id"], i["agent_id"], i["target_exposure"]) for i in store2.query("intents")]
    assert i1 == i2
    l1 = [(r["agent_id"], r["kind"]) for r in store1.query("lineage")]
    l2 = [(r["agent_id"], r["kind"]) for r in store2.query("lineage")]
    assert l1 == l2


def test_causality_orders_after_intents_fills_after_orders(run) -> None:  # type: ignore[no-untyped-def]
    """Regression: orders must never be stamped before the decision that created them."""
    h, store, _ = run
    intents = {i["intent_id"]: i for i in store.query("intents")}
    orders = {o["client_order_id"]: o for o in store.query("orders")}
    min_latency = h.engine.cfg.sim.latency.min_ms
    for o in orders.values():
        assert o["created_ts"] >= intents[o["intent_id"]]["ts"]
    for f in store.query("fills"):
        o = orders.get(f["client_order_id"])
        if o is not None:
            assert f["ts"] >= o["created_ts"] + min_latency
    for t in store.query("trades"):
        if t["entry_intent_id"]:
            assert t["entry_ts"] > intents[t["entry_intent_id"]]["ts"]
