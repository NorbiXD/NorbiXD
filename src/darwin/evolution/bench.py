"""Allocator benchmark: does Thompson sampling beat fitness weighting (and equal weight)?

Design: for each seed, one market stream is generated and replayed once per allocator. Because
agents are evaluated on their shadow books, the *population, its decisions and its evolution are
identical across allocators* for a given seed — only the capital weights differ. The comparison is
therefore perfectly paired: per-seed differences in terminal log-equity isolate the allocator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from darwin.config.challenge import ChallengeConfig
from darwin.evolution.fitness import max_drawdown
from darwin.market.synthetic import SyntheticMarket
from darwin.runtime.replay import build_replay


@dataclass(frozen=True)
class BenchRow:
    seed: int
    allocator: str
    final_equity: float
    log_growth: float
    max_drawdown: float
    orders: int


@dataclass(frozen=True)
class BenchSummary:
    allocator: str
    runs: int
    mean_final: float
    median_final: float
    mean_log_growth: float
    mean_max_drawdown: float
    paired_vs_baseline_mean: float  # mean per-seed log-growth difference vs the baseline allocator
    paired_vs_baseline_t: float
    wins_vs_baseline: int


def run_bench(
    cfg: ChallengeConfig,
    seeds: Sequence[int],
    hours: float,
    kinds: Sequence[str] = ("equal", "fitness_weighted", "thompson"),
    baseline: str = "equal",
    step_ms: int = 5_000,
    planted: bool = True,
    start_ts: int = 1_700_000_000_000,
) -> tuple[list[BenchRow], list[BenchSummary]]:
    rows: list[BenchRow] = []
    base = cfg.model_copy(update={"challenge": cfg.challenge.model_copy(update={"duration_hours": hours})})
    for seed in seeds:
        market = SyntheticMarket(
            symbols=base.challenge.symbols,
            start_ts=start_ts,
            duration_ms=base.duration_ms + 2 * base.challenge.bar_ms,
            seed=seed,
            step_ms=step_ms,
            planted_edges=planted,
        )
        events = list(market.events())
        for kind in kinds:
            c = base.model_copy(
                update={
                    "allocator": base.allocator.model_copy(update={"kind": kind}),
                    "sim": base.sim.model_copy(update={"seed": seed}),
                }
            )
            h = build_replay(c, iter(events), start_ts, run_id=f"bench-{kind}-{seed}")
            final = h.driver.run()
            eq = np.array([e for _, e in h.engine.equity_curve], dtype=float)
            start = c.challenge.starting_capital
            rows.append(
                BenchRow(
                    seed,
                    kind,
                    final,
                    math.log(max(final, 1e-9) / start),
                    max_drawdown(eq) if eq.size else 0.0,
                    h.engine.stats.get("orders", 0),
                )
            )
    summaries = []
    for kind in kinds:
        mine = {r.seed: r for r in rows if r.allocator == kind}
        ref = {r.seed: r for r in rows if r.allocator == baseline}
        diffs = np.array([mine[s].log_growth - ref[s].log_growth for s in mine if s in ref])
        sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
        t = float(diffs.mean() / (sd / math.sqrt(diffs.size))) if sd > 0 else 0.0
        finals = np.array([r.final_equity for r in mine.values()])
        summaries.append(
            BenchSummary(
                allocator=kind,
                runs=len(mine),
                mean_final=float(finals.mean()),
                median_final=float(np.median(finals)),
                mean_log_growth=float(np.mean([r.log_growth for r in mine.values()])),
                mean_max_drawdown=float(np.mean([r.max_drawdown for r in mine.values()])),
                paired_vs_baseline_mean=float(diffs.mean()) if diffs.size else 0.0,
                paired_vs_baseline_t=t,
                wins_vs_baseline=int((diffs > 1e-12).sum()),
            )
        )
    return rows, summaries
