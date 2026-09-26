"""Population management: evaluation, selection, death, reproduction, lineage, champion.

Statistical hygiene
-------------------
* **Comparable windows.** Births only happen at generation boundaries and every alive agent is
  marked-to-market on every bar, so any two agents share an identical, contiguous set of bars
  (the younger agent's life). Relative performance is measured as a *paired* statistic against
  the cohort median on exactly those bars, which removes the market's common component.
* **Minimum evidence.** Nobody is ranked for death or reproduction before ``min_trades`` closed
  trades and ``min_age_generations``. One losing trade kills nobody.
* **Strikes, not executions.** Persistent inferiority (bottom quantile *and* significantly below
  the cohort) earns a strike and a demotion to probation (no capital, still evaluated). Only
  repeated strikes kill. Recovery removes strikes.
* **Diversity.** Correlated clones share fitness (penalised for correlation with better-ranked
  agents), species share is capped, and immigrants are injected every generation.
* **Only ruin is instant.** An evaluation book below ``ruin_fraction`` is dead immediately.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from darwin.agents.agent import Agent
from darwin.agents.genome import Genome, Mutation, crossover, genome_distance, mutate, random_genome
from darwin.agents.primitives import PRIMITIVES
from darwin.config.challenge import EvolutionSettings
from darwin.core.ids import Sequence
from darwin.core.types import AgentStatus, LineageEventKind
from darwin.evolution.fitness import FitnessReport, TradeSample, compute_fitness, paired_t_stat


@dataclass
class AgentBook:
    """Per-agent evaluation evidence from its shadow account."""

    ts: list[int] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    regimes: list[str] = field(default_factory=list)
    trades: list[TradeSample] = field(default_factory=list)


@dataclass
class Evaluation:
    agent_id: str
    generation: int
    ts: int
    report: FitnessReport  # own window: [max(now - eval window, born), now]
    adjusted_fitness: float  # window-matched relative fitness, correlation-penalised (selection metric)
    eligible: bool
    excess_t: float  # paired t-stat of bar returns vs cohort median on identical bars
    max_corr: float
    age_generations: int
    rank: int = 0
    relative_fitness: float = 0.0  # fitness - cohort median fitness on the *same* window
    cohort_median: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "ts": self.ts,
            "fitness": self.report.fitness,
            "adjusted_fitness": self.adjusted_fitness,
            "relative_fitness": self.relative_fitness,
            "cohort_median": self.cohort_median,
            "eligible": self.eligible,
            "excess_t": self.excess_t,
            "max_corr": self.max_corr,
            "age_generations": self.age_generations,
            "rank": self.rank,
            "metrics": self.report.to_record(),
        }


@dataclass
class LineageEvent:
    ts: int
    generation: int
    agent_id: str
    kind: LineageEventKind
    genome_id: str
    parent_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "generation": self.generation,
            "agent_id": self.agent_id,
            "kind": self.kind.value,
            "genome_id": self.genome_id,
            "parent_ids": list(self.parent_ids),
            "details": self.details,
        }


@dataclass
class EvolutionResult:
    generation: int
    evaluations: dict[str, Evaluation]
    killed: list[tuple[Agent, str]]
    born: list[Agent]
    demoted: list[Agent]
    reinstated: list[Agent]
    champion_id: str | None
    champion_changed: bool
    lineage: list[LineageEvent]


class Population:
    def __init__(
        self,
        cfg: EvolutionSettings,
        symbols: tuple[str, ...],
        bar_ms: int,
        horizon_days: float,
        seed: int = 7,
    ) -> None:
        self.cfg = cfg
        self.symbols = symbols
        self.bar_ms = bar_ms
        self.horizon_days = horizon_days
        self.rng = np.random.default_rng(seed + 424_242)
        self.agents: dict[str, Agent] = {}
        self.books: dict[str, AgentBook] = {}
        self.generation = 0
        self.generation_start_ts = 0
        self.champion_id: str | None = None
        self.ids = Sequence("A", width=4)
        self.lineage: list[LineageEvent] = []  # drained by the engine for persistence
        self.last_evaluations: dict[str, Evaluation] = {}
        self.genome_origin: dict[str, str] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def alive(self) -> list[Agent]:
        return [a for a in self.agents.values() if a.alive]

    def _new_agent(
        self,
        genome: Genome,
        ts: int,
        kind: LineageEventKind,
        parents: tuple[str, ...] = (),
        details: dict[str, Any] | None = None,
    ) -> Agent:
        agent = Agent(
            agent_id=self.ids.next(),
            genome=genome,
            generation=self.generation,
            parent_ids=parents,
            born_ts=ts,
        )
        self.agents[agent.agent_id] = agent
        self.books[agent.agent_id] = AgentBook()
        self.lineage.append(
            LineageEvent(
                ts=ts,
                generation=self.generation,
                agent_id=agent.agent_id,
                kind=kind,
                genome_id=genome.genome_id,
                parent_ids=parents,
                details={"species": genome.species, **(details or {})},
            )
        )
        return agent

    def quarantine(self, agent_id: str, now: int, details: dict[str, Any]) -> Agent | None:
        """Kill an agent whose code failed at runtime (outside selection: no evidence needed)."""
        a = self.agents.get(agent_id)
        if a is None or not a.alive:
            return None
        a.status = AgentStatus.DEAD
        a.died_ts = now
        a.death_reason = "runtime_error"
        if self.champion_id == agent_id:
            self.champion_id = None
        self.lineage.append(
            LineageEvent(
                now,
                self.generation,
                a.agent_id,
                LineageEventKind.KILLED,
                a.genome.genome_id,
                a.parent_ids,
                {"reason": "runtime_error", **details},
            )
        )
        return a

    def drain_lineage(self) -> list[LineageEvent]:
        out, self.lineage = self.lineage, []
        return out

    # ------------------------------------------------------------------ seeding
    def seed(
        self, ts: int, genomes: list[tuple[Genome, str]] | None = None, fill: bool = True
    ) -> list[Agent]:
        """Initial population.

        ``genomes`` (genome, origin) are placed first — e.g. a champion set from offline
        evolution (``darwin evolve``) or sandbox-validated species. With ``fill`` the rest of
        the population is random, covering every seed species.
        """
        self.generation_start_ts = ts
        born: list[Agent] = []
        for g, origin in genomes or []:
            if any(a.genome.genome_id == g.genome_id for a in born):
                continue
            self.genome_origin[g.genome_id] = origin
            born.append(self._new_agent(g, ts, LineageEventKind.BORN, details={"origin": origin}))
        if not fill:
            return born
        species = [s for s in self.cfg.seed_species if s in PRIMITIVES]
        i = 0
        while len(born) < self.cfg.population_size:
            prim = species[i % len(species)] if species and i < 2 * len(species) else None
            g = random_genome(self.rng, self.symbols, primitive=prim, max_terms=1 if prim else 2)
            if any(a.genome.genome_id == g.genome_id for a in born):
                continue
            born.append(self._new_agent(g, ts, LineageEventKind.BORN, details={"origin": "seed"}))
            i += 1
        return born

    def inject(self, genome: Genome, ts: int, origin: str, details: dict[str, Any] | None = None) -> Agent:
        """Add an externally proposed genome (e.g. a sandbox-validated Level-2 species)."""
        self.genome_origin[genome.genome_id] = origin
        return self._new_agent(
            genome, ts, LineageEventKind.PROPOSED, details={"origin": origin, **(details or {})}
        )

    # ------------------------------------------------------------------ evidence
    def record_bar(self, ts: int, equities: dict[str, float], regime: str) -> None:
        for aid, eq in equities.items():
            book = self.books.get(aid)
            if book is None:
                continue
            book.ts.append(ts)
            book.equity.append(eq)
            book.regimes.append(regime)

    def record_trade(self, agent_id: str, sample: TradeSample) -> None:
        book = self.books.get(agent_id)
        if book is not None:
            book.trades.append(sample)

    def returns(self, agent_id: str, since_ts: int) -> tuple[np.ndarray, np.ndarray]:
        """(timestamps, bar log-returns) of an agent's shadow equity since ``since_ts``."""
        b = self.books[agent_id]
        if len(b.ts) < 2:
            return np.array([], dtype=np.int64), np.array([], dtype=float)
        ts = np.asarray(b.ts, dtype=np.int64)
        eq = np.asarray(b.equity, dtype=float)
        mask = ts >= since_ts
        ts, eq = ts[mask], eq[mask]
        if eq.size < 2:
            return np.array([], dtype=np.int64), np.array([], dtype=float)
        return ts[1:], np.diff(np.log(np.maximum(eq, 1e-9)))

    # ------------------------------------------------------------------ evaluation
    def _window_fitness(self, agent_id: str, start: int, now: int, seed: str) -> FitnessReport:
        b = self.books[agent_id]
        ts = np.asarray(b.ts, dtype=np.int64)
        eq = np.asarray(b.equity, dtype=float)
        m = ts >= start
        trades = [t for t in b.trades if t.exit_ts >= start]
        return compute_fitness(
            eq[m],
            trades,
            window_ms=max(now - start, self.bar_ms),
            settings=self.cfg.fitness,
            horizon_days=self.horizon_days,
            seed_key=seed,
        )

    def evaluate(self, now: int) -> dict[str, Evaluation]:
        """Evaluate every alive agent.

        Absolute fitness is measured on each agent's own window. Because agents born in
        different generations have windows of different length (and therefore different
        regimes), *selection* uses window-matched relative fitness: an agent's fitness minus the
        median fitness of every agent that was alive over exactly the same bars. Births happen
        only at generation boundaries, so there are at most ``eval_generations + 1`` distinct
        windows and the benchmark cost is O(windows x agents).
        """
        cfg = self.cfg
        window_ms = cfg.eval_generations * cfg.generation_bars * self.bar_ms
        since = now - window_ms
        alive = self.alive
        starts = {a.agent_id: max(since, a.born_ts) for a in alive}
        rets = {a.agent_id: self.returns(a.agent_id, starts[a.agent_id]) for a in alive}
        # fitness of every agent on every distinct window it fully covers
        by_window: dict[int, dict[str, FitnessReport]] = {}
        for w_start in sorted(set(starts.values())):
            cohort = [a for a in alive if a.born_ts <= w_start]
            by_window[w_start] = {
                a.agent_id: self._window_fitness(
                    a.agent_id, w_start, now, f"{a.agent_id}:{self.generation}:{w_start}"
                )
                for a in cohort
            }
        # cohort median return per bar (paired, window-matched comparison)
        by_ts: dict[int, list[float]] = {}
        for ts_arr, r_arr in rets.values():
            for t, r in zip(ts_arr.tolist(), r_arr.tolist(), strict=True):
                by_ts.setdefault(t, []).append(r)
        median = {t: float(np.median(v)) for t, v in by_ts.items()}

        evals: dict[str, Evaluation] = {}
        for a in alive:
            w = by_window[starts[a.agent_id]]
            rep = w[a.agent_id]
            cohort_med = float(np.median([r.fitness for r in w.values()]))
            ts_arr, r_arr = rets[a.agent_id]
            med = np.array([median[t] for t in ts_arr.tolist()], dtype=float)
            excess_t = paired_t_stat(r_arr, med) if r_arr.size else 0.0
            # completed generations lived, measured in time so mid-generation calls agree
            age = int((now - a.born_ts) // (cfg.generation_bars * self.bar_ms))
            eligible = rep.n_trades >= cfg.min_trades and age >= cfg.min_age_generations
            rel = rep.fitness - cohort_med
            evals[a.agent_id] = Evaluation(
                agent_id=a.agent_id,
                generation=self.generation,
                ts=now,
                report=rep,
                adjusted_fitness=rel,
                eligible=eligible,
                excess_t=excess_t,
                max_corr=0.0,
                age_generations=age,
                relative_fitness=rel,
                cohort_median=cohort_med,
            )

        # fitness sharing: penalise correlation with better-ranked eligible agents
        ranked = sorted((e for e in evals.values() if e.eligible), key=lambda e: -e.relative_fitness)
        kept: list[Evaluation] = []
        for e in ranked:
            mc = 0.0
            for better in kept:
                mc = max(mc, self._corr(rets[e.agent_id], rets[better.agent_id]))
            e.max_corr = mc
            thr = cfg.correlation_threshold
            pen = cfg.correlation_penalty * max(0.0, (mc - thr) / (1 - thr))
            e.adjusted_fitness = e.relative_fitness - abs(e.relative_fitness) * pen
            kept.append(e)
        order = sorted(evals.values(), key=lambda e: (not e.eligible, -e.adjusted_fitness))
        for i, e in enumerate(order, 1):
            e.rank = i
        self.last_evaluations = evals
        return evals

    @staticmethod
    def _corr(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
        ta, ra = a
        tb, rb = b
        if ra.size < 30 or rb.size < 30:
            return 0.0
        common, ia, ib = np.intersect1d(ta, tb, assume_unique=True, return_indices=True)
        if common.size < 30:
            return 0.0
        x, y = ra[ia], rb[ib]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    # ------------------------------------------------------------------ selection
    def evolve(self, now: int, shadow_equity: dict[str, float]) -> EvolutionResult:
        cfg = self.cfg
        evals = self.evaluate(now)
        killed: list[tuple[Agent, str]] = []
        demoted: list[Agent] = []
        reinstated: list[Agent] = []

        def kill(a: Agent, reason: str, details: dict[str, Any]) -> None:
            a.status = AgentStatus.DEAD
            a.died_ts = now
            a.death_reason = reason
            killed.append((a, reason))
            self.lineage.append(
                LineageEvent(
                    now,
                    self.generation,
                    a.agent_id,
                    LineageEventKind.KILLED,
                    a.genome.genome_id,
                    a.parent_ids,
                    {"reason": reason, **details},
                )
            )

        # 1) ruin: instant death
        for a in self.alive:
            eq = shadow_equity.get(a.agent_id, cfg.eval_capital)
            if eq <= cfg.fitness.ruin_fraction * cfg.eval_capital:
                kill(a, "ruined", {"equity": round(eq, 4)})

        # 2) persistent inferiority (eligible only), 3) inactivity
        eligible = [evals[a.agent_id] for a in self.alive if evals[a.agent_id].eligible]
        eligible.sort(key=lambda e: e.adjusted_fitness)
        n_bottom = math.floor(len(eligible) * cfg.kill_fraction)
        bottom = {e.agent_id for e in eligible[:n_bottom]}
        for e in eligible:
            a = self.agents[e.agent_id]
            if not a.alive:
                continue
            inferior = e.agent_id in bottom and e.excess_t < -cfg.kill_t_stat
            if inferior:  # the champion is not immune: a degraded champion is demoted too
                a.strikes += 1
                details = {
                    "strikes": a.strikes,
                    "fitness": round(e.adjusted_fitness, 4),
                    "excess_t": round(e.excess_t, 3),
                }
                if a.strikes >= cfg.max_strikes:
                    kill(a, "persistent_inferiority", details)
                elif a.status is not AgentStatus.PROBATION:
                    a.status = AgentStatus.PROBATION
                    demoted.append(a)
                    self.lineage.append(
                        LineageEvent(
                            now,
                            self.generation,
                            a.agent_id,
                            LineageEventKind.DEMOTED,
                            a.genome.genome_id,
                            a.parent_ids,
                            details,
                        )
                    )
            else:
                if a.strikes > 0:
                    a.strikes -= 1
                if a.status is AgentStatus.PROBATION and a.strikes == 0:
                    a.status = AgentStatus.ALIVE
                    reinstated.append(a)
                    self.lineage.append(
                        LineageEvent(
                            now,
                            self.generation,
                            a.agent_id,
                            LineageEventKind.REINSTATED,
                            a.genome.genome_id,
                            a.parent_ids,
                            {"fitness": round(e.adjusted_fitness, 4)},
                        )
                    )
        for a in self.alive:
            e = evals[a.agent_id]
            lifetime_trades = len(self.books[a.agent_id].trades)
            if e.age_generations >= cfg.max_inactive_generations and lifetime_trades < cfg.min_trades:
                kill(a, "inactive", {"trades": lifetime_trades, "age_generations": e.age_generations})

        # 4) champion / challenger
        champion_changed = self._update_champion(evals, now)

        # 5) reproduction into free slots
        self.generation += 1
        self.generation_start_ts = now
        born = self._reproduce(evals, now)

        return EvolutionResult(
            generation=self.generation,
            evaluations=evals,
            killed=killed,
            born=born,
            demoted=demoted,
            reinstated=reinstated,
            champion_id=self.champion_id,
            champion_changed=champion_changed,
            lineage=list(self.lineage),
        )

    def _qualified(self, e: Evaluation) -> bool:
        """Parent / champion / capital eligible: beats the cohort on identical bars (relative),
        has an absolute edge after risk penalties, and an acceptable ruin probability."""
        return (
            e.eligible
            and self.agents[e.agent_id].status is AgentStatus.ALIVE
            and e.adjusted_fitness > 0
            and e.report.fitness > 0
            and e.report.ruin_prob <= self.cfg.max_ruin_prob_parent
        )

    def _update_champion(self, evals: dict[str, Evaluation], now: int) -> bool:
        cfg = self.cfg
        cur = self.champion_id
        cur_ok = cur is not None and self.agents[cur].status is AgentStatus.ALIVE and cur in evals
        cands = [e for e in evals.values() if self._qualified(e)]
        if not cands:
            if cur is not None and not cur_ok:
                self.champion_id = None  # dead or demoted champions lose the title
                return True
            return False
        best = max(cands, key=lambda e: e.adjusted_fitness)
        if cur_ok and cur != best.agent_id:
            assert cur is not None
            ce = evals[cur]
            since = max(self.agents[cur].born_ts, self.agents[best.agent_id].born_ts)
            _, rc = self.returns(cur, since)
            _, rb = self.returns(best.agent_id, since)
            t = paired_t_stat(rb, rc)
            if not (
                best.adjusted_fitness > ce.adjusted_fitness + cfg.champion_margin and t > cfg.champion_t_stat
            ):
                return False
        if cur == best.agent_id:
            return False
        self.champion_id = best.agent_id
        a = self.agents[best.agent_id]
        self.lineage.append(
            LineageEvent(
                now,
                self.generation,
                a.agent_id,
                LineageEventKind.PROMOTED,
                a.genome.genome_id,
                a.parent_ids,
                {"fitness": round(best.adjusted_fitness, 4), "previous": cur},
            )
        )
        return True

    def _tournament(self, pool: list[Evaluation]) -> Evaluation:
        k = min(self.cfg.tournament_size, len(pool))
        idx = self.rng.choice(len(pool), size=k, replace=False)
        return max((pool[int(i)] for i in idx), key=lambda e: e.adjusted_fitness)

    def _reproduce(self, evals: dict[str, Evaluation], now: int) -> list[Agent]:
        cfg = self.cfg
        slots = cfg.population_size - len(self.alive)
        if slots <= 0:
            return []
        born: list[Agent] = []
        pool = [e for e in evals.values() if self._qualified(e)]
        pool.sort(key=lambda e: -e.adjusted_fitness)
        n_elite = max(1, math.ceil(len(pool) * cfg.elite_fraction)) if pool else 0
        parents = pool[: max(n_elite, min(len(pool), cfg.tournament_size))]
        if parents:
            # exploration is a fraction of the free slots, never all of them: when parents
            # qualify at least one slot always goes to offspring
            n_imm = min(slots - 1, max(1 if slots >= 3 else 0, round(cfg.immigrant_rate * slots)))
        else:
            n_imm = slots  # nobody has earned the right to reproduce: explore instead
        existing = {a.genome.genome_id for a in self.alive}
        species_count = Counter(a.species for a in self.alive)
        cap = max(1, math.floor(cfg.max_species_share * cfg.population_size))

        offspring: Counter[str] = Counter()
        attempts = 0
        while len(born) < slots - n_imm and attempts < 20 * slots:
            attempts += 1
            open_parents = [p for p in parents if offspring[p.agent_id] < cfg.max_offspring_per_parent]
            if not open_parents:
                break  # every parent has its quota: the rest of the slots go to immigrants
            pa = self._tournament(open_parents)
            a_agent = self.agents[pa.agent_id]
            muts: list[Mutation]
            if len(open_parents) >= 2 and self.rng.random() < cfg.crossover_rate:
                others = [p for p in open_parents if p.agent_id != pa.agent_id]
                pb = self._tournament(others)
                b_agent = self.agents[pb.agent_id]
                child = crossover(a_agent.genome, b_agent.genome, self.rng, cfg.max_terms)
                child, muts = mutate(
                    child,
                    self.rng,
                    cfg.mutation_sigma / 2,
                    self.symbols,
                    cfg.structural_mutation_rate / 2,
                    cfg.max_terms,
                )
                kind = LineageEventKind.CROSSOVER
                parent_ids: tuple[str, ...] = (pa.agent_id, pb.agent_id)
            else:
                child, muts = mutate(
                    a_agent.genome,
                    self.rng,
                    cfg.mutation_sigma,
                    self.symbols,
                    cfg.structural_mutation_rate,
                    cfg.max_terms,
                )
                kind = LineageEventKind.MUTATED
                parent_ids = (pa.agent_id,)
            if child.genome_id in existing:
                continue
            if kind is LineageEventKind.MUTATED and genome_distance(child, a_agent.genome) < 0.01:
                continue
            if species_count[child.species] >= cap:
                continue
            existing.add(child.genome_id)
            species_count[child.species] += 1
            offspring.update(parent_ids)
            born.append(
                self._new_agent(
                    child,
                    now,
                    kind,
                    parent_ids,
                    details={
                        "mutations": [m.model_dump() for m in muts],
                        "parent_fitness": {p: round(evals[p].adjusted_fitness, 4) for p in parent_ids},
                    },
                )
            )

        # immigrants (and any unfilled offspring slots): favour under-represented primitives
        while len(born) < slots:
            counts = Counter(t.primitive for a in self.alive for t in a.genome.terms)
            names = sorted(PRIMITIVES, key=lambda n: (counts.get(n, 0), self.rng.random()))
            g = random_genome(self.rng, self.symbols, primitive=names[0], max_terms=cfg.max_terms)
            if g.genome_id in existing:
                continue
            existing.add(g.genome_id)
            born.append(self._new_agent(g, now, LineageEventKind.BORN, details={"origin": "immigrant"}))
        return born
