from __future__ import annotations

import numpy as np

from darwin.agents.genome import SCALAR_SPACE, Genome, crossover, genome_distance, mutate, random_genome
from darwin.agents.primitives import PRIMITIVES

SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _in_bounds(g: Genome) -> None:
    for path, spec in SCALAR_SPACE.items():
        v = g.scalar(path)
        assert spec.low - 1e-9 <= v <= spec.high + 1e-9, (path, v)
    for t in g.terms:
        assert -1 <= t.weight <= 1
        for k, spec in PRIMITIVES[t.primitive].params.items():
            v = t.params[k]
            if spec.kind == "choice":
                assert 0 <= v < len(spec.choices)
            else:
                assert spec.low - 1e-9 <= v <= spec.high + 1e-9, (t.primitive, k, v)


def test_random_genomes_are_valid_and_hash_stably() -> None:
    rng = np.random.default_rng(1)
    for prim in sorted(PRIMITIVES):
        g = random_genome(rng, SYMS, primitive=prim)
        _in_bounds(g)
        assert g.species == prim or g.species.startswith("anti-") or g.species.startswith("hybrid")
        clone = Genome.model_validate(g.model_dump(mode="json"))
        assert clone.genome_id == g.genome_id  # content-addressed and round-trippable


def test_mutation_stays_in_bounds_and_records_diffs() -> None:
    rng = np.random.default_rng(2)
    g = random_genome(rng, SYMS, primitive="momentum")
    for _ in range(200):
        child, muts = mutate(g, rng, sigma=0.3, universe=SYMS, structural_rate=0.5, max_terms=3)
        _in_bounds(child)
        for m in muts:
            assert m.path
        g = child
    assert 1 <= len(g.terms) <= 3


def test_crossover_of_different_species_produces_hybrid() -> None:
    rng = np.random.default_rng(3)
    a = random_genome(rng, SYMS, primitive="momentum")
    b = random_genome(rng, SYMS, primitive="funding_oi")
    seen_hybrid = False
    for _ in range(20):
        c = crossover(a, b, rng, max_terms=3)
        _in_bounds(c)
        prims = {t.primitive for t in c.terms}
        if prims == {"momentum", "funding_oi"}:
            seen_hybrid = True
            assert c.species.startswith("hybrid:")
    assert seen_hybrid


def test_same_structure_crossover_mixes_parameters() -> None:
    rng = np.random.default_rng(4)
    a = random_genome(rng, ("BTCUSDT",), primitive="mean_reversion")
    b = random_genome(rng, ("BTCUSDT",), primitive="mean_reversion")
    c = crossover(a, b, rng)
    for k, v in c.terms[0].params.items():
        assert v in (a.terms[0].params[k], b.terms[0].params[k])


def test_distance() -> None:
    rng = np.random.default_rng(5)
    a = random_genome(rng, ("BTCUSDT",), primitive="momentum")
    assert genome_distance(a, a) == 0.0
    b = random_genome(rng, ("BTCUSDT",), primitive="breakout")
    assert genome_distance(a, b) == 1.0
