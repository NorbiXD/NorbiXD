"""Known-answer tests for the evolutionary machinery as a whole (train -> select -> holdout).

Every claim is made across seeds, against a baseline, with a statistical criterion — a single
seed proved nothing (QM iteration 2). Per seed: evolve 48h on one path, freeze the top genomes,
and evaluate them on the next, unseen 24h together with 12 *random, unselected* genomes on the
same bars.

What these tests prove — and, as importantly, what they do not:

* **Selection beats random deployment, on both markets.** Selected genomes lose much less than
  random genomes out of sample on the planted *and* on the null market. On noise this can only
  be cost/risk avoidance (random genomes overtrade and size badly), so this is NOT evidence that
  evolution discovers the planted edge (QM iteration 3, M5). The test is named for what it shows.
* **No false discovery:** on the null market selection produces no positive out-of-sample
  return, and train winners degrade.
* **Direction of the null control only:** the advantage over random genomes is larger on the
  planted market than on the null one, but NOT significantly so at this budget (48h training:
  Welch t ~ 1.3; 96h: t ~ 2.0 with only 3 null seeds producing champions). Absolute holdout
  returns do not separate the markets (t ~ 0.5). Edge discovery is therefore an open problem,
  not a claim (progress.md has the per-seed tables).
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pytest

HOUR = 3_600_000
PLANTED_SEEDS = (101, 102, 103, 104, 105, 106)
NULL_SEEDS = (201, 202, 203, 204, 205, 206)


def _seed_summary(planted: bool, seed: int, train_h: int = 48, hold_h: int = 24) -> dict[str, Any]:
    from darwin.evolution.tournament import run_tournament
    from darwin.market.synthetic import SyntheticMarket
    from tests.conftest import T0, make_config

    cfg = make_config(
        challenge={"duration_hours": train_h, "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
        evolution={"eval_generations": 6, "min_trades": 6},
        sim={"seed": seed},
    )
    m = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=T0,
        duration_ms=(train_h + hold_h) * HOUR,
        step_ms=5_000,
        seed=seed,
        planted_edges=planted,
    )
    r = run_tournament(
        cfg,
        list(m.events()),
        T0,
        train_h * HOUR,
        hold_h * HOUR,
        top_k=8,
        population_size=32,
        generation_bars=240,
        baseline_k=12,
        baseline_seed=seed + 1000,
    )
    return {
        "seed": seed,
        "champions": [c.holdout["net_return"] for c in r.champions],
        "baseline": [b.holdout["net_return"] for b in r.baselines],
        "excess": r.excess_over_baseline() if r.champions else None,
        "degradation": r.degradation() if r.champions else None,
        "species": [c.species for c in r.champions],
    }


@pytest.fixture(scope="module")
def results() -> dict[str, list[dict[str, Any]]]:
    jobs = [(True, s) for s in PLANTED_SEEDS] + [(False, s) for s in NULL_SEEDS]
    ctx = multiprocessing.get_context("spawn")  # never fork a process that has threads running
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1), mp_context=ctx) as ex:
        out = list(ex.map(_seed_summary, *zip(*jobs, strict=True)))
    n = len(PLANTED_SEEDS)
    return {"planted": out[:n], "null": out[n:]}


def _t(x: np.ndarray) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(x.size))) if x.size > 1 and x.std() > 0 else 0.0


@pytest.mark.slow
def test_selection_avoids_the_losses_of_random_deployment_on_both_markets(
    results: dict[str, list[dict[str, Any]]],
) -> None:
    rows = [r for r in results["planted"] if r["excess"] is not None]
    assert len(rows) >= 5, "evolution must produce eligible champions on most planted seeds"
    excess = np.array([r["excess"] for r in rows])
    assert excess.mean() > 0.0 and _t(excess) > 2.0, excess
    assert (excess > 0).sum() >= len(rows) - 2, excess
    # the same holds on noise, which is exactly why this is not evidence of edge discovery
    null_excess = np.array([r["excess"] for r in results["null"] if r["excess"] is not None])
    assert null_excess.size == 0 or null_excess.mean() > 0.0, null_excess


@pytest.mark.slow
def test_null_control_direction_planted_advantage_exceeds_null_advantage(
    results: dict[str, list[dict[str, Any]]],
) -> None:
    """Direction only (not significant at this budget; see module docstring): a regression guard
    against evolution doing *better* on noise than on a market with a real edge."""
    planted = np.array([r["excess"] for r in results["planted"] if r["excess"] is not None])
    null = np.array([r["excess"] for r in results["null"] if r["excess"] is not None])
    assert null.size == 0 or planted.mean() > null.mean(), (planted, null)


@pytest.mark.slow
def test_null_market_produces_no_out_of_sample_edge_and_winners_degrade(
    results: dict[str, list[dict[str, Any]]],
) -> None:
    rows = [r for r in results["null"] if r["champions"]]
    seed_means = np.array([np.mean(r["champions"]) for r in rows])
    # selection on noise may find nothing at all (no eligible, positive agents): also a pass
    assert seed_means.size == 0 or seed_means.mean() < 0.005, seed_means
    assert (seed_means < 0.02).all(), seed_means
    degr = np.array([r["degradation"] for r in rows])
    assert degr.size == 0 or degr.mean() > 0, "train winners on noise must degrade out of sample"
