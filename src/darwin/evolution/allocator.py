"""Capital allocator (meta-agent): which forms of intelligence deserve real capital right now.

It does not predict trades. It decides weights over agents given their shadow evidence.

Three interchangeable policies, benchmarked against each other in ``darwin bench-allocators``:

* ``equal``            — equal weight over eligible agents (the baseline any policy must beat).
* ``fitness_weighted`` — softmax over correlation-adjusted fitness, capped, top-K.
* ``thompson``         — contextual Thompson sampling: a Normal posterior over each agent's mean
  bar return *in the current regime* (falling back to all regimes when regime evidence is thin);
  sample, fund agents whose sampled mean is positive with Kelly-like weights ∝ μ̃/σ². Exploration
  comes from posterior uncertainty rather than a fixed ε.

All policies share the same post-processing: diversification (a candidate correlated above
``max_pair_correlation`` with a better-ranked one is dropped, so clones cannot hold the book), at
most ``max_funded_agents``, per-agent cap ``max_weight``, a cash buffer, and an explicit
exploration slice for unproven challengers.
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


def return_correlation(a: np.ndarray, b: np.ndarray, min_overlap: int = 30) -> float | None:
    """Correlation of two agents' bar returns over their common (most recent) bars.

    Every alive agent is marked on every bar, so the tails of two return series are aligned.
    ``None`` when the overlap is too short or either series is flat (no evidence either way).
    """
    n = min(a.size, b.size)
    if n < min_overlap:
        return None
    x, y = a[-n:], b[-n:]
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def diversify(
    ranked: list[AllocationCandidate], cfg: AllocatorSettings
) -> tuple[list[AllocationCandidate], dict[str, str]]:
    """Greedy de-cloning: walk candidates best-first, drop any whose returns correlate above
    ``max_pair_correlation`` with one already kept. Returns (kept, {dropped: kept_twin})."""
    if cfg.max_pair_correlation >= 1.0:
        return ranked, {}
    kept: list[AllocationCandidate] = []
    dropped: dict[str, str] = {}
    for c in ranked:
        twin = None
        for k in kept:
            rho = return_correlation(c.bar_returns, k.bar_returns, cfg.min_overlap_bars)
            if rho is not None and rho > cfg.max_pair_correlation:
                twin = k.agent_id
                break
        if twin is None:
            kept.append(c)
        else:
            dropped[c.agent_id] = twin
    return kept, dropped


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
    cands: list[AllocationCandidate], funded: list[AllocationCandidate], cfg: AllocatorSettings
) -> dict[str, float]:
    """Give the exploration slice to the most promising unproven challenger that is not a clone
    of an agent already funded (exploration buys new information, not more of the same bet)."""
    if cfg.exploration_budget <= 0:
        return {}
    taken = {c.agent_id for c in funded}
    unproven = sorted(
        (
            c
            for c in cands
            if not c.eligible
            and c.status is AgentStatus.ALIVE
            and c.agent_id not in taken
            and c.net_return > 0
        ),
        key=lambda c: -c.net_return,
    )
    for c in unproven:
        if cfg.max_pair_correlation < 1.0 and any(
            (rho := return_correlation(c.bar_returns, f.bar_returns, cfg.min_overlap_bars)) is not None
            and rho > cfg.max_pair_correlation
            for f in funded
        ):
            continue
        return {c.agent_id: min(cfg.exploration_budget, cfg.max_weight)}
    return {}


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
        ok, _ = diversify(ok, cfg)
        top = ok[: cfg.max_funded_agents]
        explore = _explore(cands, top, cfg)
        budget = 1 - cfg.cash_buffer - sum(explore.values())
        out = _finalise({c.agent_id: 1.0 for c in top}, cfg, budget)
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
        ok.sort(key=lambda c: -c.adjusted_fitness)
        ok, _ = diversify(ok, cfg)
        ok = ok[: cfg.max_funded_agents]
        explore = _explore(cands, ok, cfg)
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
        scored: list[tuple[float, AllocationCandidate]] = []
        for c in ok:
            m, sd, s2 = self.posterior(c, regime)
            draw = float(rng.normal(m, sd))
            if draw > 0:
                scored.append((draw / s2, c))
        scored.sort(key=lambda t: -t[0])
        kept, _ = diversify([c for _, c in scored], cfg)
        keep = {c.agent_id for c in kept}
        raw = {c.agent_id: w for w, c in scored if c.agent_id in keep}
        budget = 1 - cfg.cash_buffer
        return _finalise(raw, cfg, budget)


def make_allocator(cfg: AllocatorSettings) -> Allocator:
    if cfg.kind == "thompson":
        return ThompsonAllocator(cfg)
    if cfg.kind == "equal":
        return EqualWeightAllocator(cfg)
    return FitnessWeightedAllocator(cfg)
