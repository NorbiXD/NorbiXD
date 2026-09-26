"""Capital allocator (meta-agent): which forms of intelligence deserve real capital right now.

It does not predict trades. It decides weights over agents given their shadow evidence.

Three interchangeable policies, benchmarked against each other in ``darwin bench-allocators``:

* ``equal``            — equal weight over eligible agents (the baseline any policy must beat).
* ``fitness_weighted`` — softmax over correlation-adjusted fitness, capped, top-K.
* ``thompson``         — contextual Thompson sampling: a Normal posterior over each agent's mean
  bar return *in the current regime* (falling back to all regimes when regime evidence is thin);
  sample, fund agents whose sampled mean is positive with Kelly-like weights ∝ μ̃/σ². Exploration
  comes from posterior uncertainty rather than a fixed ε.

All policies share the same post-processing: at most ``max_funded_agents``, per-agent cap
``max_weight``, a cash buffer, and an explicit exploration slice for unproven challengers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from darwin.config.challenge import AllocatorSettings
from darwin.core.types import AgentStatus


@dataclass(frozen=True)
class AllocationCandidate:
    agent_id: str
    status: AgentStatus
    eligible: bool
    adjusted_fitness: float
    ruin_prob: float
    net_return: float
    bar_returns: np.ndarray
    regime_labels: tuple[str, ...]  # regime label per bar return (same length)
    #: own-window absolute fitness; ``adjusted_fitness`` is relative to the cohort on the same
    #: bars. Capital requires both: beating a losing crowd is not an edge.
    absolute_fitness: float | None = None

    @property
    def has_edge(self) -> bool:
        absolute = self.adjusted_fitness if self.absolute_fitness is None else self.absolute_fitness
        return self.adjusted_fitness > 0 and absolute > 0


class Allocator(Protocol):
    name: str

    def allocate(
        self, cands: list[AllocationCandidate], regime: str, rng: np.random.Generator
    ) -> dict[str, float]: ...


def _finalise(raw: dict[str, float], cfg: AllocatorSettings, budget: float) -> dict[str, float]:
    """Top-K, per-agent cap (with redistribution), scale to budget."""
    items = sorted(((k, v) for k, v in raw.items() if v > 0), key=lambda kv: -kv[1])[: cfg.max_funded_agents]
    if not items:
        return {}
    w = {k: v for k, v in items}
    total = sum(w.values())
    w = {k: v / total * budget for k, v in w.items()}
    cap = cfg.max_weight
    for _ in range(10):  # iterative capping; excess stays in cash if everyone is capped
        over = {k: v for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for k in over:
            w[k] = cap
        free = {k: v for k, v in w.items() if v < cap - 1e-12}
        if not free:
            break
        ft = sum(free.values())
        for k, v in free.items():
            w[k] = min(cap, v + excess * v / ft)
    return {k: round(v, 6) for k, v in w.items() if v > 1e-6}


def _explore(
    cands: list[AllocationCandidate], taken: dict[str, float], cfg: AllocatorSettings
) -> dict[str, float]:
    """Give the exploration slice to the most promising unproven challenger."""
    if cfg.exploration_budget <= 0:
        return {}
    unproven = [
        c
        for c in cands
        if not c.eligible and c.status is AgentStatus.ALIVE and c.agent_id not in taken and c.net_return > 0
    ]
    if not unproven:
        return {}
    best = max(unproven, key=lambda c: c.net_return)
    return {best.agent_id: min(cfg.exploration_budget, cfg.max_weight)}


class EqualWeightAllocator:
    name = "equal"

    def __init__(self, cfg: AllocatorSettings) -> None:
        self.cfg = cfg

    def allocate(
        self, cands: list[AllocationCandidate], regime: str, rng: np.random.Generator
    ) -> dict[str, float]:
        cfg = self.cfg
        ok = [c for c in cands if c.eligible and c.status is AgentStatus.ALIVE and c.has_edge]
        ok.sort(key=lambda c: -c.adjusted_fitness)
        explore = _explore(cands, {}, cfg)
        budget = 1 - cfg.cash_buffer - sum(explore.values())
        out = _finalise({c.agent_id: 1.0 for c in ok}, cfg, budget)
        return {**out, **explore}


class FitnessWeightedAllocator:
    name = "fitness_weighted"

    def __init__(self, cfg: AllocatorSettings) -> None:
        self.cfg = cfg

    def allocate(
        self, cands: list[AllocationCandidate], regime: str, rng: np.random.Generator
    ) -> dict[str, float]:
        cfg = self.cfg
        ok = [
            c
            for c in cands
            if c.eligible and c.status is AgentStatus.ALIVE and c.has_edge and c.ruin_prob <= 0.5
        ]
        explore = _explore(cands, {}, cfg)
        budget = 1 - cfg.cash_buffer - sum(explore.values())
        if not ok:
            return explore
        f = np.array([c.adjusted_fitness for c in ok])
        z = (f - f.max()) / cfg.temperature
        p = np.exp(z)
        raw = {c.agent_id: float(pi) for c, pi in zip(ok, p, strict=True)}
        return {**_finalise(raw, cfg, budget), **explore}


class ThompsonAllocator:
    name = "thompson"

    def __init__(self, cfg: AllocatorSettings, min_regime_bars: int = 60) -> None:
        self.cfg = cfg
        self.min_regime_bars = min_regime_bars

    def posterior(self, c: AllocationCandidate, regime: str) -> tuple[float, float, float]:
        """(posterior mean, posterior sd, sample variance) of the agent's mean bar return."""
        r = c.bar_returns
        if c.regime_labels and len(c.regime_labels) == r.size:
            mask = np.array([lbl == regime for lbl in c.regime_labels])
            if mask.sum() >= self.min_regime_bars:
                r = r[mask]
        n = r.size
        tau2 = self.cfg.thompson_prior_sd**2
        if n < 2:
            return 0.0, math.sqrt(tau2), tau2
        s2 = max(float(np.var(r, ddof=1)), 1e-12)
        prec = n / s2 + 1 / tau2
        mean = (n * float(np.mean(r)) / s2) / prec
        return mean, math.sqrt(1 / prec), s2

    def allocate(
        self, cands: list[AllocationCandidate], regime: str, rng: np.random.Generator
    ) -> dict[str, float]:
        cfg = self.cfg
        ok = [
            c
            for c in cands
            if c.status is AgentStatus.ALIVE and c.ruin_prob <= 0.5 and c.bar_returns.size >= 30
        ]
        raw: dict[str, float] = {}
        for c in ok:
            m, sd, s2 = self.posterior(c, regime)
            draw = float(rng.normal(m, sd))
            if draw > 0:
                raw[c.agent_id] = draw / s2
        budget = 1 - cfg.cash_buffer
        return _finalise(raw, cfg, budget)


def make_allocator(cfg: AllocatorSettings) -> Allocator:
    if cfg.kind == "thompson":
        return ThompsonAllocator(cfg)
    if cfg.kind == "equal":
        return EqualWeightAllocator(cfg)
    return FitnessWeightedAllocator(cfg)
