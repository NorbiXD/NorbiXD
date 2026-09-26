"""The contract between intelligence and execution.

Agents (deterministic, statistical or model-backed) can only ever produce a ``TradeIntent``.
They never see the Risk Governor's limits, never hold exchange credentials and never size the
final order. ``TradeIntent -> RiskGovernor -> ExecutionEngine -> Venue`` is the only path.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from darwin.core.types import Urgency

IntentReason = Literal[
    "entry",
    "exit",
    "flip",
    "stop_loss",
    "take_profit",
    "max_hold",
    "regime_gate",
    "agent_killed",
    "defunded",
    "rebalance",
    "challenge_end",
    "circuit_breaker",
    "kill_switch",
    "stale_exit",
    "desync_exit",
    "runtime_error",
]

SYSTEM_REASONS = frozenset(
    {
        "stop_loss",
        "take_profit",
        "max_hold",
        "agent_killed",
        "defunded",
        "rebalance",
        "challenge_end",
        "circuit_breaker",
        "kill_switch",
        "desync_exit",
        "runtime_error",
    }
)


class TradeIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_id: str
    ts: int
    agent_id: str
    genome_id: str
    symbol: str
    #: desired signed position as a multiple of the agent's capital (0 = flat)
    target_exposure: float = Field(allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: IntentReason
    stop_loss_pct: float | None = Field(default=None, gt=0)
    take_profit_pct: float | None = Field(default=None, gt=0)
    max_hold_ms: int | None = Field(default=None, gt=0)
    urgency: Urgency = Urgency.MARKET
    max_slippage_bps: float = Field(default=15.0, gt=0)
    # ---- attribution payload: what the agent knew and why
    score: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)
    regime: str = "unknown"
    features_seen: dict[str, float] = Field(default_factory=dict)
    signals_seen: tuple[str, ...] = ()
    provider: str | None = None
    model_response: dict[str, Any] | None = None
    snapshot_ref: str | None = None
    parent_intent_id: str | None = None  # for system intents (stops) -> the entry they protect

    @property
    def is_system(self) -> bool:
        return self.reason in SYSTEM_REASONS

    @property
    def direction(self) -> Literal["LONG", "SHORT", "FLAT"]:
        if self.target_exposure > 0:
            return "LONG"
        if self.target_exposure < 0:
            return "SHORT"
        return "FLAT"
