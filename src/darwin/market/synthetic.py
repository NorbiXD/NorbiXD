"""Synthetic perpetual-futures market generator.

Purpose: a *known-answer* environment for the evolutionary machinery, plus an offline demo feed
when exchange access is unavailable. It is NOT evidence about real markets.

Two modes:

* ``planted_edges=True`` — a regime-switching market (trend / range / high-vol) with deliberately
  planted, *cost-aware-exploitable* structure: drift in trend regimes (momentum/breakout edge),
  Ornstein-Uhlenbeck reversion in range regimes (mean-reversion edge), liquidation cascades that
  overshoot and partially revert (liquidation-fade edge), and a weak order-book-imbalance drift
  (an edge that should *not* survive taker fees at 1-minute horizons).
* ``planted_edges=False`` — a null market: correlated geometric random walk with the same
  microstructure *surface* (books, funding, OI, liquidations) but no predictive structure. Any
  "edge" evolution claims here is a false discovery; tests assert it stays near cost-negative.

Everything is seeded and vectorised per chunk; the event stream is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from darwin.core.events import (
    BookSnapshot,
    FundingSettlement,
    LiquidationEvent,
    TickerEvent,
    TradeEvent,
)
from darwin.core.types import Side
from darwin.market.state import MarketDataEvent

FUNDING_INTERVAL_MS = 8 * 3_600_000


@dataclass(frozen=True)
class SymbolProfile:
    symbol: str
    price: float
    vol_1m: float  # std of 1-minute log returns
    tick: float
    trade_notional: float  # typical taker trade notional (USD)
    depth_notional: float  # notional per book level


DEFAULT_PROFILES: dict[str, SymbolProfile] = {
    "BTCUSDT": SymbolProfile("BTCUSDT", 60_000.0, 0.0006, 0.1, 40_000.0, 250_000.0),
    "ETHUSDT": SymbolProfile("ETHUSDT", 3_000.0, 0.0008, 0.01, 20_000.0, 120_000.0),
    "SOLUSDT": SymbolProfile("SOLUSDT", 150.0, 0.0012, 0.01, 8_000.0, 40_000.0),
    "DOGEUSDT": SymbolProfile("DOGEUSDT", 0.15, 0.0015, 0.00001, 4_000.0, 20_000.0),
    "XRPUSDT": SymbolProfile("XRPUSDT", 0.6, 0.0012, 0.0001, 5_000.0, 25_000.0),
}

_REG_TREND_UP, _REG_TREND_DOWN, _REG_RANGE, _REG_HIGH_VOL = 0, 1, 2, 3
REGIME_NAMES = ("trend_up", "trend_down", "range", "high_vol")


@dataclass
class SyntheticMarket:
    symbols: tuple[str, ...]
    start_ts: int
    duration_ms: int
    seed: int = 7
    step_ms: int = 1_000
    book_every: int = 5
    book_levels: int = 20
    planted_edges: bool = True
    regime_mean_minutes: float = 180.0
    correlation: float = 0.7
    #: trend drift as a multiple of the per-hour noise, per hour
    trend_strength: float = 1.2
    #: OU half-life (minutes) in range regimes
    range_half_life_min: float = 25.0
    #: drift per unit book imbalance, in per-step sigmas (weak by design)
    imbalance_kappa: float = 0.03
    jump_rate_per_hour: float = 0.4
    chunk_steps: int = 3_600
    profiles: dict[str, SymbolProfile] = field(default_factory=lambda: dict(DEFAULT_PROFILES))
    #: regime path actually generated, one entry per (ts, regime) change; for diagnostics/tests
    regime_log: list[tuple[int, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for s in self.symbols:
            if s not in self.profiles:
                self.profiles[s] = SymbolProfile(s, 100.0, 0.001, 0.01, 5_000.0, 25_000.0)

    @property
    def n_steps(self) -> int:
        return self.duration_ms // self.step_ms

    def events(self) -> Iterator[MarketDataEvent]:
        rng = np.random.default_rng(self.seed)
        syms = self.symbols
        k = len(syms)
        profs = [self.profiles[s] for s in syms]
        steps_per_min = 60_000 / self.step_ms
        sig = np.array([p.vol_1m / math.sqrt(steps_per_min) for p in profs])  # per-step sigma
        logp = np.log(np.array([p.price for p in profs]))
        anchor = logp.copy()
        imb = np.zeros(k)
        funding = np.full(k, 0.0001)
        oi = np.array([p.depth_notional * 400 / p.price for p in profs])
        # rolling window for liquidation detection
        win = max(int(5 * steps_per_min), 5)
        hist = [logp.copy()]
        regime = _REG_RANGE
        regime_left = 0
        cascade_left = np.zeros(k, dtype=int)
        cascade_dir = np.zeros(k)
        cascade_origin = logp.copy()
        cascade_peak = logp.copy()
        refractory = np.zeros(k, dtype=int)
        push_steps = max(int(2 * steps_per_min), 1)
        revert_steps = max(int(8 * steps_per_min), 1)
        # total push ~ 10 one-minute sigmas, spread over the push phase
        push_per_step = 10.0 * np.array([p.vol_1m for p in profs]) / push_steps
        revert_k = 3.0 / revert_steps
        next_funding = (self.start_ts // FUNDING_INTERVAL_MS + 1) * FUNDING_INTERVAL_MS
        trade_seq = 0
        update_id = 0
        ou_theta = math.log(2) / (self.range_half_life_min * steps_per_min)
        steps_per_hour = 60 * steps_per_min
        # trend drift per step so that over one hour drift = trend_strength * hourly noise
        trend_mu = self.trend_strength * sig * math.sqrt(steps_per_hour) / steps_per_hour
        jump_p = self.jump_rate_per_hour / steps_per_hour
        corr = self.correlation
        mean_regime_steps = self.regime_mean_minutes * steps_per_min

        n = self.n_steps
        step = 0
        while step < n:
            m = min(self.chunk_steps, n - step)
            common = rng.standard_normal(m)
            idio = rng.standard_normal((m, k))
            shocks = corr**0.5 * common[:, None] + (1 - corr) ** 0.5 * idio
            imb_noise = rng.standard_normal((m, k))
            u_jump = rng.random((m, k))
            jump_sz = rng.standard_normal((m, k))
            u_side = rng.random((m, k))
            qty_noise = rng.lognormal(0.0, 0.6, (m, k))
            regime_draw = rng.random(m)
            fund_noise = rng.standard_normal((m, k))
            for i in range(m):
                ts = self.start_ts + (step + i) * self.step_ms
                # ---------------- regime switching (market-wide)
                if self.planted_edges:
                    if regime_left <= 0:
                        r = regime_draw[i]
                        regime = (
                            _REG_TREND_UP
                            if r < 0.28
                            else _REG_TREND_DOWN
                            if r < 0.56
                            else _REG_RANGE
                            if r < 0.86
                            else _REG_HIGH_VOL
                        )
                        regime_left = int(rng.exponential(mean_regime_steps)) + int(20 * steps_per_min)
                        anchor = logp.copy()
                        self.regime_log.append((ts, REGIME_NAMES[regime]))
                    regime_left -= 1
                vol_mult = 2.2 if (self.planted_edges and regime == _REG_HIGH_VOL) else 1.0
                s = sig * vol_mult
                # ---------------- order book imbalance: AR(1)
                imb = 0.97 * imb + 0.12 * imb_noise[i]
                imb = np.clip(imb, -0.9, 0.9)
                drift = np.zeros(k)
                if self.planted_edges:
                    if regime == _REG_TREND_UP:
                        drift += trend_mu
                    elif regime == _REG_TREND_DOWN:
                        drift -= trend_mu
                    elif regime == _REG_RANGE:
                        drift += -ou_theta * (logp - anchor)
                    drift += self.imbalance_kappa * imb * s
                    # liquidation cascade: forced flow overshoots (push phase), then price
                    # retraces half of the move from the pre-cascade level (revert phase)
                    active = cascade_left > 0
                    if active.any():
                        phase_push = cascade_left > revert_steps
                        drift += np.where(active & phase_push, cascade_dir * push_per_step, 0.0)
                        revert = active & ~phase_push
                        target = cascade_origin + 0.5 * (cascade_peak - cascade_origin)
                        cascade_peak = np.where(active & phase_push, logp, cascade_peak)
                        drift += np.where(revert, -(logp - target) * revert_k, 0.0)
                        cascade_left = np.maximum(cascade_left - 1, 0)
                    refractory = np.maximum(refractory - 1, 0)
                jumps = np.where(u_jump[i] < jump_p * vol_mult, jump_sz[i] * s * 12.0, 0.0)
                dlogp = drift + s * shocks[i] + jumps
                logp = logp + dlogp
                hist.append(logp.copy())
                if len(hist) > win + 1:
                    hist.pop(0)
                price = np.exp(logp)

                # ---------------- funding / OI
                trend_signal = (hist[-1] - hist[0]) / (sig * math.sqrt(win) + 1e-12)
                if self.planted_edges:
                    funding = 0.995 * funding + 0.005 * (0.0001 + 0.00015 * np.tanh(trend_signal / 3))
                else:
                    funding = 0.995 * funding + 0.005 * 0.0001
                funding = funding + 0.000002 * fund_noise[i]
                oi = oi * (1 + 0.0002 * np.tanh(np.abs(trend_signal) / 3) - 0.00005) + 0.0
                oi = np.maximum(oi, 1.0)

                events: list[MarketDataEvent] = []
                emit_book = (step + i) % self.book_every == 0
                if emit_book:
                    for j, sym in enumerate(syms):
                        prof = profs[j]
                        update_id += 1
                        events.append(
                            TickerEvent(
                                ts=ts,
                                symbol=sym,
                                last_price=float(price[j]),
                                mark_price=float(price[j]),
                                index_price=float(price[j] * (1 + 0.0001 * imb[j])),
                                funding_rate=float(funding[j]),
                                next_funding_ts=next_funding,
                                open_interest=float(oi[j]),
                            )
                        )
                        events.append(
                            self._book(ts, sym, prof, float(price[j]), float(imb[j]), vol_mult, update_id)
                        )
                # ---------------- trades (one aggregated print per symbol per step)
                for j, sym in enumerate(syms):
                    prof = profs[j]
                    p_buy = 0.5 + 0.35 * imb[j] + (0.1 if dlogp[j] > 0 else -0.1)
                    side = Side.BUY if u_side[i, j] < p_buy else Side.SELL
                    qty = prof.trade_notional * qty_noise[i, j] / price[j]
                    trade_seq += 1
                    events.append(
                        TradeEvent(
                            ts=ts,
                            symbol=sym,
                            price=_round_to(float(price[j]), prof.tick),
                            qty=float(qty),
                            taker_side=side,
                            trade_id=f"T{trade_seq}",
                        )
                    )
                # ---------------- liquidations after sharp moves
                move = hist[-1] - hist[0]
                thresh = 3.5 * sig * math.sqrt(win)
                for j, sym in enumerate(syms):
                    if abs(move[j]) > thresh[j] and cascade_left[j] == 0 and refractory[j] == 0:
                        down = move[j] < 0
                        liq_side = Side.BUY if down else Side.SELL  # longs liquidated on a drop
                        notional = profs[j].trade_notional * (5 + 10 * min(abs(move[j]) / thresh[j], 4))
                        events.append(
                            LiquidationEvent(
                                ts=ts,
                                symbol=sym,
                                side=liq_side,
                                price=float(price[j]),
                                qty=float(notional / price[j]),
                            )
                        )
                        cascade_left[j] = push_steps + revert_steps if self.planted_edges else 0
                        refractory[j] = int(45 * steps_per_min)
                        cascade_dir[j] = -1.0 if down else 1.0
                        cascade_origin[j] = hist[0][j]
                        cascade_peak[j] = logp[j]
                        oi[j] *= 0.97
                # ---------------- funding settlement
                if ts >= next_funding:
                    for j, sym in enumerate(syms):
                        events.append(
                            FundingSettlement(
                                ts=ts, symbol=sym, rate=float(funding[j]), mark_price=float(price[j])
                            )
                        )
                    next_funding += FUNDING_INTERVAL_MS
                yield from events
            step += m

    def _book(
        self,
        ts: int,
        sym: str,
        prof: SymbolProfile,
        mid: float,
        imb: float,
        vol_mult: float,
        update_id: int,
    ) -> BookSnapshot:
        spacing = max(prof.tick, mid * 0.00005)
        half_spread = max(prof.tick, mid * 0.00003 * vol_mult)
        best_bid = _round_down(mid - half_spread, prof.tick)
        best_ask = max(_round_up(mid + half_spread, prof.tick), best_bid + prof.tick)
        base_qty = prof.depth_notional / mid / vol_mult
        bids = tuple(
            (_round_down(best_bid - i * spacing, prof.tick), base_qty * (1 + imb) * (1 + 0.15 * i))
            for i in range(self.book_levels)
        )
        asks = tuple(
            (_round_up(best_ask + i * spacing, prof.tick), base_qty * (1 - imb) * (1 + 0.15 * i))
            for i in range(self.book_levels)
        )
        return BookSnapshot(ts=ts, symbol=sym, bids=_dedupe(bids), asks=_dedupe(asks), update_id=update_id)


def _round_to(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 10)


def _round_down(x: float, tick: float) -> float:
    return round(math.floor(x / tick + 1e-9) * tick, 10)


def _round_up(x: float, tick: float) -> float:
    return round(math.ceil(x / tick - 1e-9) * tick, 10)


def _dedupe(levels: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    seen: dict[float, float] = {}
    for p, q in levels:
        if q > 0:
            seen[p] = seen.get(p, 0.0) + q
    return tuple(seen.items())
