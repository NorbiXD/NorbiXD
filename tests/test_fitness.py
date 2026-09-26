from __future__ import annotations

import math

import numpy as np
import pytest

from darwin.config.challenge import FitnessSettings
from darwin.evolution.fitness import (
    TradeSample,
    bootstrap_ruin,
    certainty_equivalent,
    compute_fitness,
    max_drawdown,
    paired_t_stat,
)

DAY = 86_400_000


def trades_from(rets: list[float], conf: float | list[float] = 0.5) -> list[TradeSample]:
    confs = conf if isinstance(conf, list) else [conf] * len(rets)
    return [
        TradeSample(ret=r, confidence=c, slippage_bps=1.0, entry_ts=i, exit_ts=i + 1)
        for i, (r, c) in enumerate(zip(rets, confs, strict=True))
    ]


def equity_from(rets: list[float], start: float = 1000.0) -> np.ndarray:
    return start * np.cumprod([1.0, *[1 + r for r in rets]])


def test_crra_special_cases() -> None:
    g = np.array([1.1, 0.95, 1.02])
    assert certainty_equivalent(g, 1.0) == pytest.approx(float(np.mean(np.log(g))))
    assert certainty_equivalent(g, 0.0) == pytest.approx(math.log(float(np.mean(g))))
    # more risk aversion => lower certainty equivalent for a risky prospect
    assert certainty_equivalent(g, 3.0) < certainty_equivalent(g, 1.0) < certainty_equivalent(g, 0.0)


def test_lucky_suicidal_bet_ranks_below_steady_edge() -> None:
    """The core property: selection must not breed ruin."""
    s = FitnessSettings()
    # agent A: one 5x-leveraged coin flip that happened to win +60%, plus tiny noise
    lucky = [0.6] + [0.0] * 2
    # agent B: 40 small trades with a genuine positive expectancy after costs
    rng = np.random.default_rng(0)
    steady = list(0.004 + 0.006 * rng.standard_normal(40))
    fa = compute_fitness(equity_from(lucky), trades_from(lucky), 2 * DAY, s, horizon_days=7)
    fb = compute_fitness(equity_from(steady), trades_from(steady), 2 * DAY, s, horizon_days=7)
    assert fa.net_return > fb.net_return  # A has the better *public* score so far...
    assert fb.fitness > fa.fitness  # ...but B has the better evidence of edge


def test_shrinkage_one_trade_cannot_dominate() -> None:
    s = FitnessSettings()
    one = compute_fitness(equity_from([0.05]), trades_from([0.05]), DAY, s, 7)
    assert one.evidence_weight < 0.5
    assert one.ce_growth_shrunk < 0.05 / 2  # pulled strongly toward the prior


def test_evidence_weight_grows_with_precision() -> None:
    s = FitnessSettings()
    rng = np.random.default_rng(3)
    noisy = list(0.002 + 0.05 * rng.standard_normal(30))
    precise = list(0.002 + 0.002 * rng.standard_normal(30))
    fn = compute_fitness(equity_from(noisy), trades_from(noisy), DAY, s, 7)
    fp = compute_fitness(equity_from(precise), trades_from(precise), DAY, s, 7)
    assert fp.evidence_weight > 0.9 > fn.evidence_weight


def test_winning_leveraged_bet_still_shows_ruin_risk_via_mae() -> None:
    s = FitnessSettings()
    rets = [0.3, 0.25, 0.35, 0.2]  # it won every time...
    trades = [
        TradeSample(ret=r, confidence=0.9, slippage_bps=1, entry_ts=i, exit_ts=i + 1, mae=-0.75)
        for i, r in enumerate(rets)
    ]  # ...after being 75% under water each time (ruin line: 70%)
    f = compute_fitness(equity_from(rets), trades, DAY, s, 7)
    assert f.ruin_prob > 0.9 and f.components["ruin"] < -5


def test_ruinous_trade_under_log_utility_is_catastrophic() -> None:
    s = FitnessSettings(gamma=1.0)
    rets = [0.05] * 10 + [-0.99]
    f = compute_fitness(equity_from(rets), trades_from(rets), DAY, s, 7)
    assert f.fitness < 0 and f.max_drawdown > 0.9


def test_bootstrap_ruin_monotone_in_risk() -> None:
    safe = np.log(1 + np.array([0.01, -0.01] * 20))
    risky = np.log(1 + np.array([0.3, -0.25] * 20))
    assert bootstrap_ruin(safe, 100, 0.3, 300, 1) == 0.0
    assert bootstrap_ruin(risky, 100, 0.3, 300, 1) > 0.2


def test_calibration_rewards_informative_confidence() -> None:
    s = FitnessSettings()
    rng = np.random.default_rng(1)
    rets = list(0.01 * rng.standard_normal(40))
    informative = [0.5 + 20 * r for r in rets]
    informative = [min(max(c, 0.0), 1.0) for c in informative]
    noise = list(rng.random(40))
    a = compute_fitness(equity_from(rets), trades_from(rets, informative), DAY, s, 7)
    b = compute_fitness(equity_from(rets), trades_from(rets, noise), DAY, s, 7)
    assert a.confidence_ic > 0.8 and a.components["calibration"] > b.components["calibration"]


def test_max_drawdown_and_paired_t() -> None:
    assert max_drawdown(np.array([100, 120, 90, 130])) == pytest.approx(0.25)
    rng = np.random.default_rng(2)
    base = rng.standard_normal(200) * 0.001
    better = base + 0.0005 + rng.standard_normal(200) * 0.0005
    assert paired_t_stat(better, base) > 10
    assert paired_t_stat(base, base) == 0.0
    assert paired_t_stat(base + 0.001, base) > 1e5  # constant positive gap


def test_no_trades_is_neutral_not_rewarded() -> None:
    f = compute_fitness(np.full(100, 1000.0), [], DAY, FitnessSettings(), 7)
    assert f.n_trades == 0 and f.fitness <= 0.0
