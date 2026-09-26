"""Internal fitness: separating genuine edge from a lucky suicidal bet.

The public score is terminal equity. Selecting directly on it is a mistake: over a short horizon
the population's "best" terminal equity is dominated by whoever took the largest leveraged bet
and got lucky, and breeding from that is breeding ruin.

The core term is the **CRRA certainty-equivalent growth rate** of the agent's per-trade returns:

    γ = 1   -> E[log R]                     (growth-optimal / Kelly criterion)
    γ < 1   -> more aggressive               (operator-configurable)
    γ = 0   -> E[R] - 1                      (risk-neutral: prefers ruinous leverage)

which is the principled bridge between "maximise final equity" and "don't die": maximising
expected log-wealth maximises the long-run growth rate of equity and assigns -inf to ruin.
It is shrunk toward a slightly negative prior with a normal-normal empirical-Bayes posterior
whose data weight is the *precision* of the evidence (n / σ²), not just the trade count: three
trades with one +60% outlier carry almost no weight, forty consistent small trades carry
nearly full weight. The same evidence weight scales the realised-terminal term. Ruin
probability is bootstrapped from (return, intra-trade MAE) pairs, so a leveraged bet that
happened to win still reveals how close it came to liquidation. Drawdown, consistency,
confidence-outcome information coefficient and slippage complete the picture. All weights
live in ``challenge.yaml``.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from darwin.config.challenge import FitnessSettings


@dataclass(frozen=True)
class TradeSample:
    ret: float  # net return on equity at entry
    confidence: float
    slippage_bps: float
    entry_ts: int
    exit_ts: int
    mae: float = 0.0  # worst intra-trade excursion as a fraction of equity (<= 0)


@dataclass(frozen=True)
class FitnessReport:
    fitness: float
    n_trades: int
    window_days: float
    ce_growth_per_trade: float
    ce_growth_shrunk: float
    evidence_weight: float
    growth_per_day: float
    terminal_log_growth: float
    net_return: float
    max_drawdown: float
    ruin_prob: float
    consistency: float
    confidence_ic: float
    brier: float
    win_rate: float
    avg_slippage_bps: float
    sharpe_per_bar: float
    components: dict[str, float]

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def certainty_equivalent(gross: np.ndarray, gamma: float) -> float:
    """Per-period certainty-equivalent log growth for gross returns ``gross`` under CRRA(γ)."""
    g = np.maximum(gross, 1e-9)
    if abs(gamma - 1.0) < 1e-9:
        return float(np.mean(np.log(g)))
    u = np.mean(g ** (1 - gamma))
    if u <= 0:
        return -math.inf
    ce = u ** (1 / (1 - gamma))
    return math.log(max(ce, 1e-12))


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = 1 - equity / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def bootstrap_ruin(
    log_rets: np.ndarray,
    n_trades: int,
    ruin_fraction: float,
    paths: int,
    seed: int,
    log_maes: np.ndarray | None = None,
) -> float:
    """P(equity touches ``ruin_fraction`` of start within ``n_trades`` resampled trades).

    If intra-trade adverse excursions are given, a path is also ruined when the equity *during*
    a trade (cumulative return before it + its MAE) crosses the threshold.
    """
    if log_rets.size == 0 or n_trades <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    n = min(n_trades, 2_000)
    idx = rng.integers(0, log_rets.size, size=(paths, n))
    cum = np.cumsum(log_rets[idx], axis=1)
    worst = cum.min(axis=1)
    if log_maes is not None and log_maes.size == log_rets.size:
        before = np.concatenate([np.zeros((paths, 1)), cum[:, :-1]], axis=1)
        worst = np.minimum(worst, (before + log_maes[idx]).min(axis=1))
    return float(np.mean(worst <= math.log(ruin_fraction)))


def evidence_weight(log_rets: np.ndarray, prior_sd: float, min_sd: float) -> float:
    """Normal-normal posterior weight on the sample mean: (n/s²) / (n/s² + 1/τ²)."""
    n = log_rets.size
    if n == 0:
        return 0.0
    s2 = float(np.var(log_rets, ddof=1)) if n >= 2 else 0.0
    s2 = max(s2, min_sd**2)
    data_prec = n / s2
    return data_prec / (data_prec + 1 / prior_sd**2)


def _ic(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    # rank correlation (Spearman) is robust to the fat tails of trade returns
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_fitness(
    equity: np.ndarray,
    trades: list[TradeSample],
    window_ms: int,
    settings: FitnessSettings,
    horizon_days: float,
    seed_key: str = "",
) -> FitnessReport:
    w = settings.weights
    window_days = max(window_ms / 86_400_000, 1e-6)
    n = len(trades)
    rets = np.array([t.ret for t in trades], dtype=float)
    gross = 1.0 + rets
    ce = certainty_equivalent(gross, settings.gamma) if n else 0.0
    if not math.isfinite(ce):
        ce = -1.0  # a ruinous trade under γ>=1
    log_rets = np.log(np.maximum(gross, 1e-9)) if n else np.array([])
    w_data = evidence_weight(log_rets, settings.prior_trade_sd, settings.min_trade_sd)
    ce_shrunk = w_data * ce + (1 - w_data) * settings.prior_trade_growth if n else 0.0
    trades_per_day = n / window_days
    growth_per_day = ce_shrunk * trades_per_day if n else 0.0

    eq = equity[np.isfinite(equity)] if equity.size else equity
    terminal = float(math.log(eq[-1] / eq[0])) if eq.size >= 2 and eq[0] > 0 and eq[-1] > 0 else 0.0
    net_return = float(eq[-1] / eq[0] - 1) if eq.size >= 2 and eq[0] > 0 else 0.0
    mdd = max_drawdown(eq) if eq.size else 0.0

    seed = zlib.crc32(seed_key.encode()) & 0xFFFFFFFF
    log_maes = np.log(np.maximum(1.0 + np.array([min(t.mae, 0.0) for t in trades]), 1e-9)) if n else None
    ruin = (
        bootstrap_ruin(
            log_rets,
            round(trades_per_day * horizon_days),
            settings.ruin_fraction,
            settings.ruin_bootstrap_paths,
            seed,
            log_maes,
        )
        if n >= 1
        else 0.0
    )

    kwin = settings.consistency_windows
    if eq.size >= kwin + 1:
        chunks = np.array_split(eq, kwin)
        pos = [c[-1] > c[0] for c in chunks if c.size >= 2]
        consistency = float(np.mean(pos)) if pos else 0.5
    else:
        consistency = 0.5

    conf = np.array([t.confidence for t in trades], dtype=float)
    ic = _ic(conf, rets) if n else 0.0
    wins = (rets > 0).astype(float)
    brier = float(np.mean((0.5 + conf / 2 - wins) ** 2)) if n else 0.25
    win_rate = float(wins.mean()) if n else 0.0
    slip = float(np.mean([t.slippage_bps for t in trades])) if n else 0.0

    if eq.size >= 3:
        br = np.diff(np.log(np.maximum(eq, 1e-9)))
        sd = float(np.std(br))
        sharpe = float(np.mean(br) / sd) if sd > 0 else 0.0
    else:
        sharpe = 0.0

    # Every component is expressed in "% equity growth per day" equivalents so weights are
    # comparable: growth terms are rates, drawdown is amortised over the window, ruin is a
    # large per-day-equivalent penalty, and the calibration/consistency terms are bounded
    # tie-breakers (at most ±weight) that cannot rescue a losing agent.
    comps = {
        "growth": w.growth * growth_per_day * 100,
        "terminal": w.terminal * w_data * terminal * 100 / max(window_days, 1.0),
        "drawdown": -w.drawdown * mdd * 100 / max(window_days, 1.0),
        "ruin": -w.ruin * ruin * 10,
        "consistency": w.consistency * (consistency - 0.5) * 2,
        "calibration": w.calibration * ic * min(1.0, n / 20),
        "slippage": -w.slippage * slip / 100,
    }
    fitness = float(sum(comps.values()))
    return FitnessReport(
        fitness=fitness,
        n_trades=n,
        window_days=window_days,
        ce_growth_per_trade=ce,
        ce_growth_shrunk=ce_shrunk,
        evidence_weight=w_data,
        growth_per_day=growth_per_day,
        terminal_log_growth=terminal,
        net_return=net_return,
        max_drawdown=mdd,
        ruin_prob=ruin,
        consistency=consistency,
        confidence_ic=ic,
        brier=brier,
        win_rate=win_rate,
        avg_slippage_bps=slip,
        sharpe_per_bar=sharpe,
        components={k2: round(v, 6) for k2, v in comps.items()},
    )


def paired_t_stat(a: np.ndarray, b: np.ndarray) -> float:
    """t-statistic of mean(a - b) for two return series on *identical* bars."""
    n = min(a.size, b.size)
    if n < 10:
        return 0.0
    d = a[-n:] - b[-n:]
    sd = float(np.std(d, ddof=1))
    m = float(np.mean(d))
    if sd < 1e-15:
        # identical series => no evidence either way; a constant non-zero gap => overwhelming
        return 0.0 if abs(m) < 1e-15 else math.copysign(1e6, m)
    return m / (sd / math.sqrt(n))
