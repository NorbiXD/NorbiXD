"""Offline accelerated evolution: train → holdout → champion set.

    historical/synthetic data ──► large population, short generations (train window)
                               ──► top-K eligible genomes by correlation-adjusted fitness
                               ──► frozen re-evaluation on a *later, unseen* holdout window
                               ──► champion set (JSON) that seeds the live population

The holdout replay starts ``warmup`` before the holdout window so indicators are primed, but only
bars/trades inside the holdout window count. Train and holdout never overlap in time, and the
holdout agents are frozen (no evolution) — so the holdout numbers are genuinely out-of-sample
for the selection step. A large train→holdout degradation is reported as overfitting.

``baseline_k`` random genomes (no selection at all) are evaluated alongside the champions on the
*same* holdout bars. Selection has found an edge only if champions beat that baseline; beating
zero is not enough when the whole market drifted (tests/test_evolution_science.py).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from darwin.agents.genome import Genome, random_genome
from darwin.config.challenge import ChallengeConfig
from darwin.core.events import Event
from darwin.evolution.fitness import FitnessReport, compute_fitness
from darwin.runtime.replay import build_replay


@dataclass(frozen=True)
class ChampionRecord:
    genome_id: str
    species: str
    genome: dict[str, Any]
    train: dict[str, float]
    holdout: dict[str, float]


@dataclass(frozen=True)
class TournamentResult:
    champions: list[ChampionRecord]
    train_generations: int
    train_population_born: int
    train_window: tuple[int, int]
    holdout_window: tuple[int, int]
    config_fingerprint: str
    elapsed_s: float
    #: random, unselected genomes evaluated on the same holdout bars (the null of selection)
    baselines: list[ChampionRecord] = field(default_factory=list)

    def excess_over_baseline(self, metric: str = "net_return") -> float:
        """Mean holdout ``metric`` of champions minus that of the random baseline."""
        if not self.champions or not self.baselines:
            return 0.0
        return float(
            np.mean([c.holdout[metric] for c in self.champions])
            - np.mean([b.holdout[metric] for b in self.baselines])
        )

    def degradation(self) -> float:
        """Mean(train fitness) - mean(holdout fitness) of the selected genomes."""
        if not self.champions:
            return 0.0
        return float(
            np.mean([c.train["fitness"] for c in self.champions])
            - np.mean([c.holdout["fitness"] for c in self.champions])
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))


def _summary(rep: FitnessReport) -> dict[str, float]:
    return {
        "fitness": round(rep.fitness, 6),
        "net_return": round(rep.net_return, 6),
        "trades": float(rep.n_trades),
        "ruin_prob": round(rep.ruin_prob, 4),
        "max_drawdown": round(rep.max_drawdown, 6),
        "evidence_weight": round(rep.evidence_weight, 4),
    }


def run_tournament(
    cfg: ChallengeConfig,
    events: Sequence[Event],
    start_ts: int,
    train_ms: int,
    holdout_ms: int,
    top_k: int = 8,
    population_size: int = 48,
    generation_bars: int = 120,
    warmup_ms: int = 6 * 3_600_000,
    baseline_k: int = 0,
    baseline_seed: int = 0,
) -> TournamentResult:
    t_start = time.time()
    train_end = start_ts + train_ms
    holdout_end = train_end + holdout_ms
    bar_ms = cfg.challenge.bar_ms

    # ---- train: evolution on the training window only
    train_cfg = cfg.model_copy(
        update={
            "challenge": cfg.challenge.model_copy(update={"duration_hours": train_ms / 3_600_000}),
            "evolution": cfg.evolution.model_copy(
                update={"population_size": population_size, "generation_bars": generation_bars}
            ),
        }
    )
    train = build_replay(
        train_cfg, (e for e in events if e.ts < train_end), start_ts, run_id="tournament-train"
    )
    train.driver.run()
    pop = train.engine.population
    evals = pop.evaluate(train_end)
    # same bar as reproduction: absolute edge (own window) AND beating the cohort on identical bars
    ranked = sorted(
        (e for e in evals.values() if e.eligible and pop.agents[e.agent_id].alive and e.report.fitness > 0),
        key=lambda e: -e.adjusted_fitness,
    )[:top_k]
    selected = [(pop.agents[e.agent_id].genome, f"tournament:{e.agent_id}") for e in ranked]
    train_metrics = {pop.agents[e.agent_id].genome.genome_id: _summary(e.report) for e in ranked}
    if not selected:
        return TournamentResult(
            [],
            pop.generation,
            len(pop.agents),
            (start_ts, train_end),
            (train_end, holdout_end),
            cfg.fingerprint(),
            time.time() - t_start,
        )

    # ---- holdout: frozen re-evaluation on unseen, later data (plus the random baseline)
    brng = np.random.default_rng(baseline_seed)
    chosen = {g.genome_id for g, _ in selected}
    baselines: list[tuple[Genome, str]] = []
    while len(baselines) < baseline_k:
        g = random_genome(brng, tuple(cfg.challenge.symbols), max_terms=cfg.evolution.max_terms)
        if g.genome_id not in chosen:
            chosen.add(g.genome_id)
            baselines.append((g, "baseline:random"))
    baseline_ids = {g.genome_id for g, _ in baselines}
    hold_start = train_end - warmup_ms
    total_bars = (holdout_end - hold_start) // bar_ms
    hold_cfg = cfg.model_copy(
        update={
            "challenge": cfg.challenge.model_copy(
                update={"duration_hours": (holdout_end - hold_start) / 3_600_000}
            ),
            "evolution": cfg.evolution.model_copy(
                update={
                    "population_size": max(4, len(selected) + len(baselines)),
                    "generation_bars": int(total_bars) + 10,
                }
            ),
        }
    )
    hold = build_replay(
        hold_cfg,
        (e for e in events if hold_start <= e.ts < holdout_end),
        hold_start,
        run_id="tournament-holdout",
        seed_genomes=selected + baselines,
        fill=False,
    )
    hold.driver.run()
    hpop = hold.engine.population
    champions: list[ChampionRecord] = []
    base_records: list[ChampionRecord] = []
    for agent in hpop.agents.values():
        book = hpop.books[agent.agent_id]
        ts = np.asarray(book.ts, dtype=np.int64)
        eq = np.asarray(book.equity, dtype=float)
        mask = ts >= train_end
        trades = [t for t in book.trades if t.entry_ts >= train_end]
        rep = compute_fitness(
            eq[mask],
            trades,
            holdout_ms,
            cfg.evolution.fitness,
            cfg.challenge.duration_hours / 24,
            seed_key=f"holdout:{agent.agent_id}",
        )
        g = agent.genome
        rec = ChampionRecord(
            genome_id=g.genome_id,
            species=g.species,
            genome=g.model_dump(mode="json"),
            train=train_metrics.get(g.genome_id, {}),
            holdout=_summary(rep),
        )
        (base_records if g.genome_id in baseline_ids else champions).append(rec)
    champions.sort(key=lambda c: -c.holdout["fitness"])
    return TournamentResult(
        champions=champions,
        train_generations=pop.generation,
        train_population_born=len(pop.agents),
        train_window=(start_ts, train_end),
        holdout_window=(train_end, holdout_end),
        config_fingerprint=cfg.fingerprint(),
        elapsed_s=time.time() - t_start,
        baselines=base_records,
    )


def load_champions(
    path: str | Path, min_holdout_fitness: float | None = 0.0, limit: int | None = None
) -> list[tuple[Genome, str]]:
    """Genomes from a saved tournament, optionally only those that held up out-of-sample."""
    data = json.loads(Path(path).read_text())
    out: list[tuple[Genome, str]] = []
    for c in data["champions"]:
        if min_holdout_fitness is not None and c["holdout"].get("fitness", -1e9) < min_holdout_fitness:
            continue
        out.append((Genome.model_validate(c["genome"]), f"champion-set:{c['genome_id']}"))
        if limit is not None and len(out) >= limit:
            break
    return out
