"""Known-answer tests for the evolutionary machinery as a whole (train -> select -> holdout).

* Planted market: the synthetic market contains a capturable trend edge (hand-built momentum /
  breakout genomes earn double-digit returns at 1x). Evolution, starting from random genomes,
  must converge on trend-family species that keep working, on average, on a later unseen
  holdout window. (Per-champion holdout results vary a lot with the holdout path; see
  progress.md for multi-seed numbers.)
* Null market: same microstructure surface, no structure. Evolution must not produce champions
  that look good out of sample — train winners should degrade (overfitting is exposed).
"""

from __future__ import annotations

import numpy as np
import pytest

from darwin.evolution.tournament import TournamentResult, run_tournament
from darwin.market.synthetic import SyntheticMarket
from tests.conftest import T0, make_config

HOUR = 3_600_000
TREND_FAMILY = ("momentum", "breakout")


def _tournament(planted: bool, seed: int, train_h: int = 48, hold_h: int = 24) -> TournamentResult:
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
    return run_tournament(
        cfg,
        list(m.events()),
        T0,
        train_h * HOUR,
        hold_h * HOUR,
        top_k=8,
        population_size=32,
        generation_bars=240,
    )


@pytest.fixture(scope="module")
def planted() -> TournamentResult:
    return _tournament(True, seed=12)


@pytest.fixture(scope="module")
def null() -> TournamentResult:
    return _tournament(False, seed=11)


def _dominant(species: str) -> str:
    return species.removeprefix("hybrid:").split("+")[0].removeprefix("anti-")


@pytest.mark.slow
def test_evolution_recovers_planted_trend_edge(planted: TournamentResult) -> None:
    assert planted.champions, "evolution must produce eligible champions"
    trend = [c for c in planted.champions if _dominant(c.species) in TREND_FAMILY]
    assert len(trend) >= 0.75 * len(planted.champions), [c.species for c in planted.champions]
    hold = np.array([c.holdout["net_return"] for c in planted.champions])
    # Out-of-sample returns of trend followers on a single 24h path are high-variance (they
    # depend on whether that day trends), so the claim is about the mean, not each champion.
    assert hold.mean() > 0.0, "selected genomes must keep their edge on unseen data on average"


@pytest.mark.slow
def test_null_market_produces_no_out_of_sample_edge(
    null: TournamentResult, planted: TournamentResult
) -> None:
    hold = np.array([c.holdout["net_return"] for c in null.champions])
    fit = np.array([c.holdout["fitness"] for c in null.champions])
    assert hold.mean() < 0.0025, "no false discovery on a random walk"
    assert (fit > 0).mean() <= 0.5
    assert null.degradation() > 0, "train winners on noise must degrade out of sample"
    planted_hold = np.mean([c.holdout["net_return"] for c in planted.champions])
    assert planted_hold > hold.mean() + 0.01
