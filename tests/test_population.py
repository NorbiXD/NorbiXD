from __future__ import annotations

import numpy as np

from darwin.agents.genome import random_genome
from darwin.config.challenge import EvolutionSettings
from darwin.core.types import AgentStatus, LineageEventKind
from darwin.evolution.fitness import TradeSample
from darwin.evolution.population import Population

BAR = 60_000
SYMS = ("BTCUSDT",)


def make_pop(**kw: object) -> Population:
    base: dict[str, object] = {
        "population_size": 10,
        "generation_bars": 100,
        "min_trades": 5,
        "min_age_generations": 1,
        "kill_fraction": 0.3,
        "kill_t_stat": 1.0,
        "max_strikes": 2,
        "immigrant_rate": 0.1,
    }
    cfg = EvolutionSettings(**{**base, **kw})  # type: ignore[arg-type]
    return Population(cfg, SYMS, BAR, horizon_days=7, seed=1)


def feed(
    pop: Population,
    t0: int,
    bars: int,
    drift: dict[str, float],
    noise: float = 0.0005,
    trades_every: int = 10,
    seed: int = 0,
) -> int:
    """Simulate shadow equity paths + trades for all alive agents over ``bars`` bars."""
    rng = np.random.default_rng(seed)
    eq = {
        a.agent_id: pop.books[a.agent_id].equity[-1] if pop.books[a.agent_id].equity else 1000.0
        for a in pop.alive
    }
    ts = t0
    for i in range(bars):
        ts += BAR
        for aid in eq:
            r = drift.get(aid, 0.0) + noise * rng.standard_normal()
            eq[aid] *= 1 + r
            if i % trades_every == 0:
                pop.record_trade(
                    aid,
                    TradeSample(
                        ret=r * trades_every, confidence=0.5, slippage_bps=1, entry_ts=ts - BAR, exit_ts=ts
                    ),
                )
        pop.record_bar(ts, dict(eq), "range")
    return ts


def test_seed_population_covers_species() -> None:
    pop = make_pop()
    born = pop.seed(0)
    assert len(born) == 10
    species = {a.species for a in born}
    assert len(species) >= 6
    assert all(e.kind is LineageEventKind.BORN for e in pop.lineage)


def test_no_death_without_minimum_evidence() -> None:
    pop = make_pop()
    pop.seed(0)
    worst = pop.alive[0].agent_id
    ts = feed(pop, 0, 100, {worst: -0.01}, trades_every=1000)  # terrible, but only 1 trade
    res = pop.evolve(ts, {a.agent_id: 900.0 for a in pop.alive})
    assert not res.killed and not res.demoted


def test_persistent_inferiority_strikes_then_death() -> None:
    pop = make_pop()
    pop.seed(0)
    loser = pop.alive[0].agent_id
    drift = {loser: -0.001}
    ts = feed(pop, 0, 100, drift, seed=1)
    res1 = pop.evolve(ts, {a.agent_id: 1000.0 for a in pop.alive})
    # first generation: nobody old enough yet (min_age_generations=1 measured in time => age 1 ok)
    ts = feed(pop, ts, 100, drift, seed=2)
    res2 = pop.evolve(ts, {a.agent_id: 1000.0 for a in pop.alive})
    demoted = {a.agent_id for a in res1.demoted + res2.demoted}
    assert loser in demoted
    assert pop.agents[loser].status in (AgentStatus.PROBATION, AgentStatus.DEAD)
    ts = feed(pop, ts, 100, drift, seed=3)
    pop.evolve(ts, {a.agent_id: 1000.0 for a in pop.alive})
    assert pop.agents[loser].status is AgentStatus.DEAD
    assert pop.agents[loser].death_reason == "persistent_inferiority"


def test_ruin_is_instant_death() -> None:
    pop = make_pop()
    pop.seed(0)
    victim = pop.alive[3].agent_id
    ts = feed(pop, 0, 10, {})
    eq = {a.agent_id: 1000.0 for a in pop.alive}
    eq[victim] = 100.0
    res = pop.evolve(ts, eq)
    assert (pop.agents[victim], "ruined") in res.killed


def test_winners_reproduce_and_slots_refill() -> None:
    pop = make_pop()
    pop.seed(0)
    ids = [a.agent_id for a in pop.alive]
    winner, loser = ids[0], ids[1]
    drift = {winner: 0.0008, loser: -0.001}
    ts = 0
    for g in range(4):
        ts = feed(pop, ts, 100, drift, seed=10 + g)
        res = pop.evolve(ts, {a.agent_id: 1000.0 for a in pop.alive})
    assert len(pop.alive) == 10  # population size maintained
    children = [a for a in pop.agents.values() if a.parent_ids]
    assert children, "winners must reproduce"
    assert any(winner in a.parent_ids for a in children)
    assert all(loser not in a.parent_ids for a in children)
    kinds = {e.kind for e in pop.lineage} | {e.kind for e in res.lineage}
    assert LineageEventKind.MUTATED in kinds or LineageEventKind.CROSSOVER in kinds
    assert pop.champion_id == winner


def test_correlated_clones_are_penalised() -> None:
    pop = make_pop(correlation_threshold=0.5, correlation_penalty=1.0)
    pop.seed(0)
    ids = [a.agent_id for a in pop.alive]
    a_id, b_id = ids[0], ids[1]
    rng = np.random.default_rng(5)
    ts = 0
    eq = {aid: 1000.0 for aid in ids}
    for _i in range(200):
        ts += BAR
        common = 0.0005 + 0.001 * rng.standard_normal()
        for aid in ids:
            r = common if aid in (a_id, b_id) else 0.001 * rng.standard_normal()
            eq[aid] *= 1 + r
        if _i % 5 == 0:
            for aid in ids:
                pop.record_trade(aid, TradeSample(0.002, 0.5, 1, ts - BAR, ts))
        pop.record_bar(ts, dict(eq), "range")
    evals = pop.evaluate(ts)
    ea, eb = evals[a_id], evals[b_id]
    clone = ea if ea.max_corr > eb.max_corr else eb
    assert clone.max_corr > 0.99
    assert clone.adjusted_fitness < clone.report.fitness


def test_champion_requires_significant_improvement() -> None:
    pop = make_pop(champion_t_stat=3.0)
    pop.seed(0)
    ids = [a.agent_id for a in pop.alive]
    ts = feed(pop, 0, 100, {ids[0]: 0.0006}, seed=1)
    pop.evolve(ts, {a: 1000.0 for a in ids})
    ts = feed(pop, ts, 100, {ids[0]: 0.0006}, seed=2)
    pop.evolve(ts, {a.agent_id: 1000.0 for a in pop.alive})
    champ = pop.champion_id
    assert champ == ids[0]
    # a challenger that is only marginally better must not dethrone the champion
    ts = feed(pop, ts, 100, {ids[0]: 0.0006, ids[1]: 0.00062}, seed=3)
    pop.evolve(ts, {a.agent_id: 1000.0 for a in pop.alive})
    assert pop.champion_id == champ


def test_species_cap_and_inject() -> None:
    pop = make_pop(max_species_share=0.3)
    pop.seed(0)
    g = random_genome(np.random.default_rng(9), SYMS, primitive="momentum")
    a = pop.inject(g, 0, origin="sandbox:P1")
    assert a.agent_id in pop.agents
    assert pop.lineage[-1].kind is LineageEventKind.PROPOSED
