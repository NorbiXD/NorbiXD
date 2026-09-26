"""Typed challenge configuration loaded from ``challenge.yaml``.

The ``RiskLimits`` block is frozen and is handed only to the Risk Governor. Nothing in the
agent/evolution/intelligence layers receives a reference to it, and its hash is persisted with
every run so any change is auditable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darwin.core.ids import content_hash
from darwin.core.types import Mode


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InstrumentSpec(_Frozen):
    symbol: str
    tick_size: float = Field(gt=0)
    qty_step: float = Field(gt=0)
    min_qty: float = Field(gt=0)
    min_notional: float = Field(default=5.0, ge=0)
    max_leverage: float = Field(default=50.0, gt=0)


class FeeSchedule(_Frozen):
    taker_bps: float = 5.5  # Bybit VIP0 linear perps
    maker_bps: float = 2.0


class ChallengeSettings(_Frozen):
    name: str = "DARWIN-001"
    starting_capital: float = Field(default=200.0, gt=0)
    duration_hours: float = Field(default=168.0, gt=0)
    mode: Mode = Mode.REPLAY
    live_enabled: bool = False
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    bar_ms: int = Field(default=60_000, ge=1_000)
    flatten_at_end: bool = True


class ExchangeSettings(_Frozen):
    name: Literal["bybit"] = "bybit"
    category: Literal["linear"] = "linear"
    fees: FeeSchedule = FeeSchedule()
    instruments: tuple[InstrumentSpec, ...] = ()
    orderbook_depth: int = 50
    recv_window_ms: int = 5_000
    order_ack_timeout_ms: int = 3_000
    reconcile_interval_ms: int = 15_000


class LatencyModel(_Frozen):
    min_ms: int = Field(default=20, ge=0)
    max_ms: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def _check(self) -> LatencyModel:
        if self.max_ms < self.min_ms:
            raise ValueError("latency max_ms < min_ms")
        return self


class SimSettings(_Frozen):
    seed: int = 7
    latency: LatencyModel = LatencyModel()
    ack_latency: LatencyModel = LatencyModel(min_ms=5, max_ms=40)
    #: extra impact per unit of (order notional / top-of-book notional), in bps
    impact_bps_per_book: float = 2.0
    liquidation_fee_rate: float = 0.005
    #: synthetic market generator
    synthetic_step_ms: int = Field(default=1_000, ge=10)
    synthetic_book_every: int = Field(default=5, ge=1)
    #: wall-clock acceleration for Mode.SIM
    speed: float = Field(default=60.0, gt=0)


class RiskLimits(_Frozen):
    """Operator-owned hard envelope. Immutable at runtime."""

    max_gross_leverage: float = Field(default=5.0, gt=0)
    max_symbol_leverage: float = Field(default=3.0, gt=0)
    max_agent_leverage: float = Field(default=5.0, gt=0)
    max_concurrent_positions: int = Field(default=8, ge=1)
    min_liquidation_distance_pct: float = Field(default=0.06, gt=0, lt=1)
    maintenance_margin_rate: float = Field(default=0.005, ge=0, lt=1)
    max_drawdown_pct: float = Field(default=0.5, gt=0, le=1)
    daily_loss_limit_pct: float = Field(default=0.35, gt=0, le=1)
    flatten_on_breaker: bool = True
    max_data_staleness_ms: int = Field(default=5_000, gt=0)
    dedup_window_ms: int = Field(default=1_500, ge=0)
    max_orders_per_minute: int = Field(default=120, ge=1)
    max_consecutive_api_errors: int = Field(default=5, ge=1)
    require_stop_loss: bool = True
    min_stop_loss_pct: float = Field(default=0.002, gt=0)
    max_stop_loss_pct: float = Field(default=0.10, gt=0)
    max_slippage_bps: float = Field(default=50.0, gt=0)
    allowed_symbols: tuple[str, ...] = ()
    kill_switch_file: str | None = "./KILL"

    @model_validator(mode="after")
    def _check(self) -> RiskLimits:
        if self.min_stop_loss_pct >= self.max_stop_loss_pct:
            raise ValueError("min_stop_loss_pct must be < max_stop_loss_pct")
        if self.max_symbol_leverage > self.max_gross_leverage:
            raise ValueError("max_symbol_leverage cannot exceed max_gross_leverage")
        return self

    def fingerprint(self) -> str:
        return content_hash(self.model_dump(mode="json"), length=16)


class FitnessWeights(_Frozen):
    """Weights of fitness components; every component is in %-growth-per-day equivalents."""

    growth: float = 1.0  # CRRA certainty-equivalent growth (shrunk by evidence), %/day
    terminal: float = 0.25  # realised log growth over the window, %/day
    drawdown: float = 0.25  # max drawdown amortised over the window, %/day
    ruin: float = 1.0  # x10: bootstrap probability of ruin over the horizon
    consistency: float = 0.2  # bounded tie-breaker: fraction of positive sub-windows
    calibration: float = 0.2  # bounded tie-breaker: confidence/outcome rank IC
    slippage: float = 0.05  # average slippage (bps/100)


class FitnessSettings(_Frozen):
    #: CRRA risk aversion. 1.0 = log utility (growth-optimal / Kelly); <1 more aggressive;
    #: 0 = risk-neutral (maximises expected terminal wealth => ruin-seeking).
    gamma: float = Field(default=1.0, ge=0.0, le=10.0)
    #: prior mean per-trade log growth (slightly negative: costs are real)
    prior_trade_growth: float = -0.0005
    #: prior dispersion of *true* per-trade edges across strategies (empirical-Bayes τ)
    prior_trade_sd: float = Field(default=0.004, gt=0)
    #: floor on the per-trade return s.d. used for the evidence weight (guards tiny samples)
    min_trade_sd: float = Field(default=0.005, gt=0)
    ruin_fraction: float = Field(default=0.3, gt=0, lt=1)
    ruin_bootstrap_paths: int = Field(default=200, ge=10)
    consistency_windows: int = Field(default=4, ge=2)
    weights: FitnessWeights = FitnessWeights()


class EvolutionSettings(_Frozen):
    population_size: int = Field(default=24, ge=4)
    generation_bars: int = Field(default=240, ge=10)
    #: fitness is measured over the most recent N generations of each agent's life
    eval_generations: int = Field(default=3, ge=1)
    max_ruin_prob_parent: float = Field(default=0.5, ge=0, le=1)
    eval_capital: float = Field(default=1_000.0, gt=0)
    min_trades: int = Field(default=6, ge=1)
    min_age_generations: int = Field(default=1, ge=0)
    kill_fraction: float = Field(default=0.25, gt=0, lt=1)
    kill_t_stat: float = Field(default=1.0, ge=0)
    max_strikes: int = Field(default=2, ge=1)
    max_inactive_generations: int = Field(default=3, ge=1)
    elite_fraction: float = Field(default=0.25, gt=0, le=1)
    tournament_size: int = Field(default=3, ge=2)
    crossover_rate: float = Field(default=0.3, ge=0, le=1)
    immigrant_rate: float = Field(default=0.15, ge=0, le=1)
    mutation_sigma: float = Field(default=0.15, gt=0, le=1)
    structural_mutation_rate: float = Field(default=0.15, ge=0, le=1)
    max_terms: int = Field(default=3, ge=1)
    max_species_share: float = Field(default=0.4, gt=0, le=1)
    correlation_threshold: float = Field(default=0.7, gt=0, lt=1)
    correlation_penalty: float = Field(default=0.5, ge=0, le=1)
    champion_margin: float = Field(default=0.0, ge=0)
    champion_t_stat: float = Field(default=1.0, ge=0)
    seed_species: tuple[str, ...] = (
        "momentum",
        "breakout",
        "mean_reversion",
        "order_flow",
        "funding_oi",
        "liquidation",
        "volatility",
        "contrarian",
    )
    fitness: FitnessSettings = FitnessSettings()


class AllocatorSettings(_Frozen):
    kind: Literal["fitness_weighted", "thompson", "equal"] = "fitness_weighted"
    temperature: float = Field(default=1.0, gt=0)
    max_weight: float = Field(default=0.4, gt=0, le=1)
    max_funded_agents: int = Field(default=5, ge=1)
    exploration_budget: float = Field(default=0.1, ge=0, le=1)
    cash_buffer: float = Field(default=0.1, ge=0, lt=1)
    rebalance_bars: int = Field(default=60, ge=1)
    thompson_prior_sd: float = Field(default=0.001, gt=0)


class PersistenceSettings(_Frozen):
    database_url: str = "sqlite:///./darwin.db"
    parquet_dir: str = "./data"
    flush_every_events: int = Field(default=2_000, ge=1)


class ApiSettings(_Frozen):
    host: str = "127.0.0.1"
    port: int = 8000


class ProviderSettings(_Frozen):
    enabled: bool = False
    base_url: str = ""
    model: str | None = None  # None => discover via the provider's model listing
    timeout_s: float = 10.0
    interval_s: float = 600.0
    extra: dict[str, Any] = Field(default_factory=dict)


class IntelligenceSettings(_Frozen):
    jev: ProviderSettings = ProviderSettings(base_url="https://api.typesafe.ai")
    grok: ProviderSettings = ProviderSettings(base_url="https://api.x.ai", interval_s=900.0)
    mock: ProviderSettings = ProviderSettings(enabled=True)


class ChallengeConfig(_Frozen):
    challenge: ChallengeSettings = ChallengeSettings()
    exchange: ExchangeSettings = ExchangeSettings()
    sim: SimSettings = SimSettings()
    risk: RiskLimits = RiskLimits()
    evolution: EvolutionSettings = EvolutionSettings()
    allocator: AllocatorSettings = AllocatorSettings()
    persistence: PersistenceSettings = PersistenceSettings()
    api: ApiSettings = ApiSettings()
    intelligence: IntelligenceSettings = IntelligenceSettings()

    @model_validator(mode="after")
    def _check(self) -> ChallengeConfig:
        allowed = set(self.risk.allowed_symbols)
        if allowed and not set(self.challenge.symbols) <= allowed:
            raise ValueError("challenge.symbols must be a subset of risk.allowed_symbols")
        return self

    def instrument(self, symbol: str) -> InstrumentSpec:
        for spec in self.exchange.instruments:
            if spec.symbol == symbol:
                return spec
        return DEFAULT_INSTRUMENTS.get(
            symbol, InstrumentSpec(symbol=symbol, tick_size=0.0001, qty_step=0.1, min_qty=0.1)
        )

    def fingerprint(self) -> str:
        return content_hash(self.model_dump(mode="json"), length=16)

    @property
    def duration_ms(self) -> int:
        return int(self.challenge.duration_hours * 3_600_000)


# Conservative fallbacks mirroring Bybit linear perp lot filters. Live runs refresh these from
# /v5/market/instruments-info; they are only used when no exchange metadata is available.
DEFAULT_INSTRUMENTS: dict[str, InstrumentSpec] = {
    "BTCUSDT": InstrumentSpec(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001),
    "ETHUSDT": InstrumentSpec(symbol="ETHUSDT", tick_size=0.01, qty_step=0.01, min_qty=0.01),
    "SOLUSDT": InstrumentSpec(symbol="SOLUSDT", tick_size=0.01, qty_step=0.1, min_qty=0.1),
    "DOGEUSDT": InstrumentSpec(symbol="DOGEUSDT", tick_size=0.00001, qty_step=1, min_qty=1),
    "XRPUSDT": InstrumentSpec(symbol="XRPUSDT", tick_size=0.0001, qty_step=0.1, min_qty=0.1),
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ChallengeConfig:
    """Load ``challenge.yaml`` (if present), apply dict overrides and environment overrides.

    Environment variables only override *infrastructure* settings (database URL, provider
    enablement). Risk limits can only come from the YAML file the operator controls.
    """
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            loaded = yaml.safe_load(p.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{p} must contain a mapping")
            data = loaded
    if overrides:
        data = _deep_merge(data, overrides)
    env_db = os.environ.get("DARWIN_DATABASE_URL")
    if env_db:
        data = _deep_merge(data, {"persistence": {"database_url": env_db}})
    return ChallengeConfig.model_validate(data)
