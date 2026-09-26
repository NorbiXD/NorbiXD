from __future__ import annotations

import numpy as np
import pytest

from darwin.config.challenge import AllocatorSettings
from darwin.core.types import AgentStatus
from darwin.evolution.allocator import (
    AllocationCandidate,
    EqualWeightAllocator,
    FitnessWeightedAllocator,
    ThompsonAllocator,
)


def cand(
    aid: str,
    fit: float,
    mean: float = 0.0,
    eligible: bool = True,
    status: AgentStatus = AgentStatus.ALIVE,
    net: float = 0.0,
    n: int = 300,
    seed: int | None = None,
) -> AllocationCandidate:
    # distinct noise per agent by default: identical series would be (correctly) de-cloned
    rng = np.random.default_rng(sum(map(ord, aid)) if seed is None else seed)
    r = mean + 0.001 * rng.standard_normal(n)
    return AllocationCandidate(aid, status, eligible, fit, 0.0, net, r, tuple(["range"] * n))


def test_fitness_weighted_caps_and_topk() -> None:
    cfg = AllocatorSettings(max_weight=0.4, max_funded_agents=3, cash_buffer=0.1, exploration_budget=0.0)
    w = FitnessWeightedAllocator(cfg).allocate(
        [cand("A", 5), cand("B", 3), cand("C", 2), cand("D", 1), cand("E", -1)],
        "range",
        np.random.default_rng(0),
    )
    assert set(w) <= {"A", "B", "C"} and "E" not in w
    assert all(v <= 0.4 + 1e-9 for v in w.values())
    assert sum(w.values()) <= 0.9 + 1e-9


def test_no_capital_when_nothing_has_edge() -> None:
    cfg = AllocatorSettings(exploration_budget=0.0)
    w = FitnessWeightedAllocator(cfg).allocate(
        [cand("A", -1), cand("B", -0.1)], "range", np.random.default_rng(0)
    )
    assert w == {}


def test_probation_and_unproven_excluded_but_exploration_funds_challenger() -> None:
    cfg = AllocatorSettings(exploration_budget=0.1)
    cands = [cand("A", 2), cand("P", 9, status=AgentStatus.PROBATION), cand("N", 0, eligible=False, net=0.05)]
    w = FitnessWeightedAllocator(cfg).allocate(cands, "range", np.random.default_rng(0))
    assert "P" not in w and w.get("N") == pytest.approx(0.1) and w["A"] > 0


def test_thompson_prefers_better_agent_on_average() -> None:
    cfg = AllocatorSettings(max_weight=1.0, max_funded_agents=5, cash_buffer=0.0)
    al = ThompsonAllocator(cfg)
    good, bad = cand("G", 1, mean=0.0003, seed=1), cand("B", 1, mean=-0.0003, seed=2)
    rng = np.random.default_rng(0)
    wins = sum(al.allocate([good, bad], "range", rng).get("G", 0) > 0 for _ in range(200))
    losses = sum(al.allocate([good, bad], "range", rng).get("B", 0) > 0 for _ in range(200))
    assert wins > 150 and losses < 50


def test_thompson_uses_regime_specific_evidence() -> None:
    cfg = AllocatorSettings(max_weight=1.0, cash_buffer=0.0)
    al = ThompsonAllocator(cfg, min_regime_bars=50)
    n = 400
    rng = np.random.default_rng(3)
    r = 0.0005 * rng.standard_normal(n)
    labels = ["trend_up" if i % 2 else "range" for i in range(n)]
    r = np.array([x + (0.001 if lbl == "trend_up" else -0.001) for x, lbl in zip(r, labels, strict=True)])
    c = AllocationCandidate("S", AgentStatus.ALIVE, True, 1.0, 0.0, 0.0, r, tuple(labels))
    m_up, _, _ = al.posterior(c, "trend_up")
    m_rng, _, _ = al.posterior(c, "range")
    assert m_up > 0 > m_rng


def test_equal_weight_baseline() -> None:
    cfg = AllocatorSettings(max_weight=1.0, max_funded_agents=4, cash_buffer=0.2, exploration_budget=0.0)
    w = EqualWeightAllocator(cfg).allocate([cand("A", 1), cand("B", 2)], "range", np.random.default_rng(0))
    assert w == {"A": pytest.approx(0.4), "B": pytest.approx(0.4)}
