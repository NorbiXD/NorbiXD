"""Regression tests for the Quality Manager's iteration-2 findings (flatten: tests/test_flatten.py)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from darwin.agents.genome import GeneTerm, Genome, RiskGenes
from darwin.agents.params import ParamSpec
from darwin.agents.primitives import PRIMITIVES, Primitive, register_primitive
from darwin.core.types import LineageEventKind
from darwin.features.engine import FeatureView
from darwin.market.synthetic import SyntheticMarket
from darwin.runtime.engine import DarwinEngine
from darwin.runtime.replay import build_replay
from tests.conftest import T0, make_config

# ----------------------------------------------------------------------------- species isolation


@pytest.fixture
def exploding_primitive() -> Iterator[str]:
    calls = {"n": 0}

    def boom(v: FeatureView, p: Mapping[str, float]) -> float:
        calls["n"] += 1
        if calls["n"] > 30:
            raise TypeError("species bug: unsupported operand")
        return 0.0

    name = "test_exploding"
    register_primitive(
        Primitive(name, boom, {"k": ParamSpec(1.0, 2.0)}, "raises after a while", origin="sandbox:test"),
        replace=True,
    )
    yield name
    PRIMITIVES.pop(name, None)


def _genome(primitive: str, params: dict[str, float]) -> Genome:
    return Genome(
        terms=(GeneTerm(primitive=primitive, params=params, weight=1.0),),
        symbols=("BTCUSDT",),
        entry_threshold=0.3,
        exit_threshold=0.0,
        risk=RiskGenes(exposure=1.0, stop_loss_pct=0.02, take_profit_pct=0.05, max_hold_bars=60),
    )


def _short_market(hours: float, seed: int = 3) -> Any:
    return SyntheticMarket(
        symbols=("BTCUSDT", "ETHUSDT"),
        start_ts=T0,
        duration_ms=int(hours * 3_600_000) + 60_000,
        step_ms=5_000,
        seed=seed,
    ).events()


def test_a_raising_species_is_quarantined_and_the_run_continues(exploding_primitive: str) -> None:
    cfg = make_config(challenge={"duration_hours": 2})
    bad = _genome(exploding_primitive, {"k": 1.5})
    h = build_replay(cfg, _short_market(2), T0, seed_genomes=[(bad, "sandbox:test")])
    eng = h.engine
    (agent,) = [a for a in eng.population.agents.values() if a.genome.genome_id == bad.genome_id]
    h.driver.run()
    assert not agent.alive and agent.death_reason == "runtime_error"
    assert eng.stats["quarantined"] == 1 and eng.stats["sys:agent_runtime_error"] == 1
    killed = [r for r in eng.recent_lineage if r["kind"] == LineageEventKind.KILLED.value]
    assert any(r["agent_id"] == agent.agent_id and r["details"]["reason"] == "runtime_error" for r in killed)
    # the rest of the population kept deciding through the end of the run
    assert eng.stats["intents"] > 20 and eng.ended and eng.bar_index >= 119


def test_a_failing_observer_is_disabled_without_stopping_trading() -> None:
    cfg = make_config(challenge={"duration_hours": 1})
    h = build_replay(cfg, _short_market(1), T0)
    seen = {"n": 0}

    def flaky(eng: DarwinEngine, ts: int, views: dict[str, FeatureView]) -> None:
        seen["n"] += 1
        raise RuntimeError("provider exploded")

    h.engine.observers.append(flaky)
    h.driver.run()
    eng = h.engine
    assert seen["n"] == 3 and flaky not in eng.observers
    assert eng.stats["sys:observer_error"] == 3 and eng.stats["sys:observer_disabled"] == 1
    assert eng.bar_index >= 59 and eng.ended


# ----------------------------------------------------------------------------- clones


def _cand(aid: str, fit: float, r: Any) -> Any:
    import numpy as np

    from darwin.core.types import AgentStatus
    from darwin.evolution.allocator import AllocationCandidate

    r = np.asarray(r, dtype=float)
    return AllocationCandidate(aid, AgentStatus.ALIVE, True, fit, 0.0, 0.01, r, tuple(["range"] * r.size))


def test_allocator_gives_no_capital_to_clones_of_a_better_agent() -> None:
    import numpy as np

    from darwin.config.challenge import AllocatorSettings
    from darwin.evolution.allocator import (
        EqualWeightAllocator,
        FitnessWeightedAllocator,
        ThompsonAllocator,
        diversify,
        return_correlation,
    )

    rng = np.random.default_rng(0)
    base = 0.0004 + 0.001 * rng.standard_normal(400)
    clones = [base + 0.0001 * rng.standard_normal(400) for _ in range(4)]  # rho ~ 0.99
    other = 0.0003 + 0.001 * rng.standard_normal(400)  # independent edge
    cands = [_cand("LEADER", 3.0, base)]
    cands += [_cand(f"CLONE{i}", 2.9 - 0.1 * i, c) for i, c in enumerate(clones)]
    cands += [_cand("OTHER", 1.0, other)]
    cfg = AllocatorSettings(exploration_budget=0.0, max_weight=0.6)
    cluster = {"LEADER", *(f"CLONE{i}" for i in range(4))}
    for alloc in (FitnessWeightedAllocator(cfg), EqualWeightAllocator(cfg), ThompsonAllocator(cfg)):
        w = alloc.allocate(cands, "range", np.random.default_rng(1))
        assert len(cluster & set(w)) == 1, (alloc.name, w)  # one member of the clone cluster
        if alloc.name != "thompson":  # ranked by fitness: the best member is the one kept
            assert "LEADER" in w
    kept, dropped = diversify(sorted(cands, key=lambda c: -c.adjusted_fitness), cfg)
    assert [c.agent_id for c in kept] == ["LEADER", "OTHER"] and set(dropped.values()) == {"LEADER"}
    # without the constraint (max_pair_correlation=1.0) the clones take most of the book
    loose = FitnessWeightedAllocator(AllocatorSettings(exploration_budget=0.0, max_pair_correlation=1.0))
    w = loose.allocate(cands, "range", np.random.default_rng(1))
    assert sum(v for k, v in w.items() if k.startswith("CLONE")) > 0.5
    assert return_correlation(base[:10], other[:10]) is None  # too little overlap: no verdict


def test_offspring_per_parent_are_capped_per_generation() -> None:
    from collections import Counter

    from tests.test_population import feed, make_pop

    pop = make_pop(population_size=16, max_offspring_per_parent=2, immigrant_rate=0.0, kill_fraction=0.5)
    pop.seed(0)
    ids = [a.agent_id for a in pop.alive]
    drift = {ids[0]: 0.001}  # one dominant parent, everyone else flat
    ts = feed(pop, 0, 100, drift, seed=1)
    pop.evolve(ts, {a: 1000.0 for a in ids})
    ts = feed(pop, ts, 100, drift, seed=2)
    eq = {a.agent_id: 1000.0 for a in pop.alive}
    for victim in ids[6:14]:  # free 8 slots at once
        eq[victim] = 100.0
    res = pop.evolve(ts, eq)
    per_parent = Counter(p for a in res.born for p in a.parent_ids)
    assert res.born and max(per_parent.values()) <= 2


def test_exploration_slot_is_not_spent_on_a_clone_of_a_funded_agent() -> None:
    import numpy as np

    from darwin.config.challenge import AllocatorSettings
    from darwin.core.types import AgentStatus
    from darwin.evolution.allocator import AllocationCandidate, FitnessWeightedAllocator

    rng = np.random.default_rng(3)
    base = 0.0004 + 0.001 * rng.standard_normal(300)
    clone = base + 0.0001 * rng.standard_normal(300)
    fresh = 0.0002 + 0.001 * rng.standard_normal(300)
    labels = tuple(["range"] * 300)
    funded = _cand("LEADER", 2.0, base)
    # unproven challengers: the clone has the better record, the fresh idea is independent
    clone_c = AllocationCandidate("CLONE", AgentStatus.ALIVE, False, 0.0, 0.0, 0.05, clone, labels)
    fresh_c = AllocationCandidate("FRESH", AgentStatus.ALIVE, False, 0.0, 0.0, 0.01, fresh, labels)
    w = FitnessWeightedAllocator(AllocatorSettings(exploration_budget=0.1)).allocate(
        [funded, clone_c, fresh_c], "range", np.random.default_rng(0)
    )
    assert "CLONE" not in w and w.get("FRESH") == pytest.approx(0.1) and w["LEADER"] > 0


# ----------------------------------------------------------------------------- audit store


def test_values_are_sanitized_so_one_bad_float_cannot_poison_a_batch(tmp_path: Any) -> None:
    import numpy as np

    from darwin.persistence.store import AuditStore

    store = AuditStore(f"sqlite:///{tmp_path / 's.db'}", "r")
    store.add(
        "system_events",
        {
            "ts": T0,
            "kind": "x",
            "detail": {"nan": float("nan"), "huge": 2**70, "np": np.float64(1.5), "ni": np.int64(3)},
        },
    )
    store.add(
        "equity", {"ts": T0, "account_id": "a", "equity": float("inf"), "cash": 1.0, "gross_notional": 0.0}
    )
    store.add("equity", {"ts": 2**70, "account_id": "b", "equity": 1.0, "cash": 1.0, "gross_notional": 0.0})
    assert store.flush() == 3
    (ev,) = store.query("system_events", run_id="r")
    assert ev["detail"] == {"nan": None, "huge": str(2**70), "np": 1.5, "ni": 3}
    eq = {r["account_id"]: r for r in store.query("equity", run_id="r")}
    assert eq["a"]["equity"] is None and eq["b"]["ts"] is None


def test_refused_rows_are_dead_lettered_and_the_rest_of_the_batch_is_written(tmp_path: Any) -> None:
    from darwin.persistence.store import AuditStore

    store = AuditStore(f"sqlite:///{tmp_path / 'd.db'}", "r")
    row = {"intent_id": "I1", "ts": T0, "agent_id": "A", "genome_id": "G", "symbol": "BTCUSDT"}
    store.add("intents", row)
    store.add("intents", dict(row))  # duplicate primary key: the database refuses it
    store.add("system_events", {"ts": T0, "kind": "fine", "detail": {}})
    assert store.flush() == 2  # did not raise: trading continues
    assert store.dead_lettered == 1 and store.pending() == 0
    (dl,) = store.query("dead_letters", run_id="r")
    assert dl["table_name"] == "intents" and "I1" in dl["payload"] and "UNIQUE" in dl["error"].upper()
    assert len(store.query("intents", run_id="r")) == 1 and store.query("system_events", run_id="r")


def test_transient_database_errors_requeue_and_raise(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy.exc import OperationalError

    from darwin.persistence.store import AuditStore

    store = AuditStore(f"sqlite:///{tmp_path / 't.db'}", "r")
    store.add("system_events", {"ts": T0, "kind": "x", "detail": {}})

    def locked() -> Any:
        raise OperationalError("INSERT", {}, Exception("database is locked"))

    real = store.engine.begin
    monkeypatch.setattr(store.engine, "begin", locked)
    with pytest.raises(OperationalError):
        store.flush()
    assert store.pending() == 1 and store.dead_lettered == 0 and store.failures == 1
    monkeypatch.setattr(store.engine, "begin", real)
    assert store.flush() == 1


def test_an_existing_run_id_is_refused_and_default_ids_are_unique(tmp_path: Any) -> None:
    from darwin.persistence.store import AuditStore, RunExistsError, new_run_id

    url = f"sqlite:///{tmp_path / 'r.db'}"
    store = AuditStore(url, "same")
    build_replay(make_config(), iter([]), T0, store=store, run_id="same")
    store.flush()
    with pytest.raises(RunExistsError):
        build_replay(make_config(), iter([]), T0, store=AuditStore(url, "same"), run_id="same")
    ids = {new_run_id("sim") for _ in range(50)}
    assert len(ids) == 50 and all(i.startswith("sim-") for i in ids)
