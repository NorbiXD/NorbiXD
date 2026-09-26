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


# ----------------------------------------------------------------------------- minors


def test_business_rejects_do_not_degrade_venue_health() -> None:
    from darwin.execution.engine import is_business_reject

    for r in ("EC_PostOnlyWillTakeLiquidity", "bybit:110007:insufficient balance", "bybit:10001:params"):
        assert is_business_reject(r), r
    for r in ("chaos_reject", "bybit:10003:invalid api key", "bybit:10006:rate limit", "timeout"):
        assert not is_business_reject(r), r


def test_position_mismatch_needs_two_snapshots_and_ignores_fresh_fills() -> None:
    from darwin.core.events import PositionSnapshot
    from darwin.core.types import Side
    from darwin.runtime.engine import CHALLENGE, CHALLENGE_VENUE

    h = build_replay(make_config(), iter([]), T0)
    eng = h.engine
    eng.ledger[CHALLENGE].pos("A0001", "BTCUSDT").apply_fill(Side.BUY, 0.01, 100.0)
    eng.now = T0 + 10_000
    eng._last_fill_ts[(CHALLENGE, "BTCUSDT")] = T0 + 8_000  # a fill 2s ago: snapshot may predate it
    snap = PositionSnapshot(ts=eng.now, account=CHALLENGE, symbol="BTCUSDT", qty=0.0)
    eng.handle(snap)
    assert eng.health[CHALLENGE_VENUE].reconcile_ok
    eng.now = T0 + 60_000
    eng.handle(snap.model_copy(update={"ts": eng.now}))
    assert eng.health[CHALLENGE_VENUE].reconcile_ok  # first real mismatch: not yet
    eng.now = T0 + 120_000
    eng.handle(snap.model_copy(update={"ts": eng.now}))
    assert not eng.health[CHALLENGE_VENUE].reconcile_ok  # persisted: now it blocks new risk
    eng.handle(snap.model_copy(update={"ts": eng.now + 60_000, "qty": 0.01}))
    assert eng.health[CHALLENGE_VENUE].reconcile_ok


def test_registry_refuses_records_without_a_passing_report(tmp_path: Any) -> None:
    import json

    from darwin.research.sandbox import SpeciesRegistry
    from tests.test_research import GOOD

    reg = SpeciesRegistry(tmp_path)
    tampered = {
        "proposal_id": "P1",
        "primitive": "evo_tampered",
        "source": GOOD,
        "report": {"passed": True, "stages": [{"stage": "static", "passed": False}]},
    }
    (tmp_path / "P1.json").write_text(json.dumps(tampered))
    assert reg.load(register=True) == [] and "evo_tampered" not in PRIMITIVES


def test_grok_model_picker_skips_non_text_models_and_parses_versions() -> None:
    from darwin.intelligence.providers.grok import pick_grok_model

    ids = ["grok-3", "grok-4-0709", "grok-code-fast-1", "grok-2-vision-1212", "grok-4.1", "grok-4.1-fast"]
    assert pick_grok_model(ids) == "grok-4.1"
    assert pick_grok_model(["grok-3-mini", "grok-3"]) == "grok-3"
    assert pick_grok_model(["grok-2-image-1212", "gpt-x"]) is None


def test_cohort_median_includes_agents_that_died_inside_the_window() -> None:
    from tests.test_population import feed, make_pop

    pop = make_pop(immigrant_rate=0.0)
    pop.seed(0)
    ids = [a.agent_id for a in pop.alive]
    ts = feed(pop, 0, 100, {ids[0]: 0.0004}, seed=1)
    base = pop.evaluate(ts)[ids[0]].cohort_median
    for aid in ids[1:5]:  # four losers die mid-window
        a = pop.agents[aid]
        a.status, a.died_ts = type(a.status).DEAD, ts
        pop.books[aid].equity[-1] *= 0.5
    ts2 = feed(pop, ts, 20, {ids[0]: 0.0004}, seed=2)
    after = pop.evaluate(ts2)[ids[0]]
    assert after.cohort_median < base  # the departed losers still count: no survivorship bias
    assert pop.prune_books(ts2) == []  # died inside the window: evidence kept
    far = ts2 + 10 * pop.cfg.eval_generations * pop.cfg.generation_bars * pop.bar_ms
    assert set(pop.prune_books(far)) == set(ids[1:5]) and pop.trade_count(ids[1]) > 0


def test_default_synthetic_stream_uses_deltas_with_gaps() -> None:
    from darwin.core.events import BookDelta
    from darwin.runtime.app import synthetic_stream

    cfg = make_config(challenge={"duration_hours": 2}, sim={"synthetic_step_ms": 1_000})
    evs = list(synthetic_stream(cfg, T0))
    deltas = [e for e in evs if isinstance(e, BookDelta)]
    assert len(deltas) > 100
    h = build_replay(cfg, iter(evs), T0)
    h.driver.run()
    assert h.engine.stats["book_invalid"] > 0 and h.engine.stats["orders"] > 0  # gaps detected, recovered


# ----------------------------------------------------------------------------- warm start


async def test_bybit_candle_backfill_warms_indicators_without_touching_evidence() -> None:
    """Paper/testnet/live start with no history: without a warm-up the fast AI path (needs 60
    bars) and long-lookback agents stayed idle for up to hours."""
    import httpx

    from darwin.exchange.bybit.rest import BybitRest
    from darwin.runtime.app import backfill_from_bybit

    forming = (T0 // 60_000) * 60_000  # Bybit candles are minute-aligned
    start = forming + 37_000  # mid-bar: the candle starting at `forming` is still open
    h = build_replay(make_config(), iter([]), start)
    eng = h.engine
    t_first = forming - 200 * 60_000

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v5/market/kline":
            rows = [
                [str(t), "100", "101", "99", str(100 + (t - t_first) / 6e6), "5", "500"]
                for t in range(t_first, forming + 60_000, 60_000)  # includes the forming candle
            ][::-1]  # Bybit: newest first
            return httpx.Response(200, json={"retCode": 0, "result": {"list": rows}})
        if req.url.path == "/v5/market/tickers":
            body = {"list": [{"fundingRate": "0.0001", "openInterest": "12345"}]}
            return httpx.Response(200, json={"retCode": 0, "result": body})
        return httpx.Response(404)

    rest = BybitRest("https://bybit.test", None, transport=httpx.MockTransport(handler))
    added = await backfill_from_bybit(eng, rest, 800)
    await rest.close()
    assert added == {"BTCUSDT": 200, "ETHUSDT": 200}  # the forming candle was dropped
    view = eng.features.view("BTCUSDT", start)
    assert view.ready(60) and view.bars[-1].end_ts == forming  # warm now, nothing from the future
    assert eng.bar_index == 0 and not eng.population.books[next(iter(eng.population.books))].equity
    assert eng.ledger["challenge"].cash == eng.cfg.challenge.starting_capital


# ----------------------------------------------------------------------------- QM iteration 3 minors


def test_webhook_recursion_bomb_is_rejected_not_a_server_error() -> None:
    from darwin.signals.external import WebhookRejected, WebhookSignalFeed

    secret = "s3cret-s3cret-s3cret"
    feed = WebhookSignalFeed(
        "alpha", secret, lambda s: None, lambda: T0, max_body_bytes=100_000, wall_ms=lambda: 5
    )
    head = b'{"id": "deep", "value": 0.1, "confidence": 0.1, "payload": {"x": '
    body = head + b"[" * 20_000 + b"]" * 20_000 + b"}}"  # 40 KB, signed, deeper than the parser allows
    with pytest.raises(WebhookRejected) as e:
        feed.ingest(body, WebhookSignalFeed.sign(secret, body, 5), "5")
    assert e.value.status == 422  # RecursionError -> 422, never an unhandled 500


def test_registry_refuses_a_passing_report_attached_to_different_source(tmp_path: Any) -> None:
    import json

    from darwin.research.sandbox import Proposal, SpeciesRegistry
    from tests.test_research import GOOD

    good = Proposal(source=GOOD)
    evil = GOOD.replace('math.tanh(z / p["s"])', '-math.tanh(z / p["s"])')
    rec = {
        "proposal_id": good.proposal_id,  # the id and report the sandbox issued for GOOD...
        "primitive": good.primitive_name,
        "source": evil,  # ...re-used for code the sandbox never saw
        "report": {"passed": True, "stages": [{"stage": s, "passed": True} for s in ("static", "unit")]},
    }
    (tmp_path / f"{good.proposal_id}.json").write_text(json.dumps(rec))
    assert SpeciesRegistry(tmp_path).load(register=True) == []
    assert good.primitive_name not in PRIMITIVES


def test_replay_survives_a_failing_audit_store_and_recovers(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from darwin.persistence.store import AuditStore
    from darwin.runtime.engine import CHALLENGE_VENUE

    store = AuditStore(f"sqlite:///{tmp_path / 'r.db'}", "r")
    real = store.flush
    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("database is locked")
        return real()

    monkeypatch.setattr(store, "flush", flaky)
    cfg = make_config(challenge={"duration_hours": 1})
    h = build_replay(cfg, _short_market(1), T0, store=store, run_id="r")
    h.driver.flush_every = 500
    h.driver.run()  # used to abort on the first failing flush
    eng = h.engine
    assert eng.ended and eng.stats["audit_flush_failures"] >= 1
    assert eng.health[CHALLENGE_VENUE].halted == ""  # recovered once the store came back
    st = eng.state()
    assert st["flatten"]["challenge_flat"] and "venue_positions" in st


def test_store_reports_the_store_error_when_even_dead_lettering_fails(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import OperationalError

    from darwin.persistence.store import AuditStore

    store = AuditStore(f"sqlite:///{tmp_path / 'l.db'}", "r")
    row = {"intent_id": "I1", "ts": T0, "agent_id": "A", "genome_id": "G", "symbol": "BTCUSDT"}
    store.add("intents", row)
    store.add("intents", dict(row))  # a refused row...
    locked = OperationalError("INSERT", {}, Exception("database is locked"))
    monkeypatch.setattr(store, "_dead_letter", lambda table, r, err: locked)  # ...while the DB locks up
    with pytest.raises(OperationalError):
        store.flush()
    assert "locked" in (store.last_error or "") and store.pending() >= 1
