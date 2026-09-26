"""Provider interfaces for model-backed intelligence.

Two narrow contracts, so Jev / Grok / Claude / OpenAI / mocks are interchangeable and can be
benchmarked on identical inputs:

* :class:`DecisionProvider` — *fast path*: typed, narrow decisions (LONG/SHORT/WAIT, regime,
  breakout validity, relevance, ...) returned as probabilities + confidence. Modelled on
  TypeSafe's System One question types: ``noul`` (yes/no probability), ``choice`` (one of N
  options with per-option probabilities) and ``score`` (rubric level).
* :class:`NarrativeProvider` — *slow path*: periodic scans (e.g. Grok + X Search) that return
  structured narrative items (entity, sentiment, confidence, catalyst summary, citations).

Neither is ever called on the market tick path. Their outputs become ``IntelligenceSignal``
events stamped with the time they *arrived*, which is the only time agents may see them.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DecisionQuestion(_M):
    key: str
    type: Literal["noul", "choice", "score"]
    instructions: str
    #: choice: option -> description ; score: level -> rubric description
    criteria: dict[str, str] | None = None


class DecisionRequest(_M):
    request_id: str
    state: str  # compact textual rendering of what the model may know
    questions: tuple[DecisionQuestion, ...]
    symbol: str | None = None
    ts: int = 0
    model: str | None = None


class DecisionAnswer(_M):
    key: str
    type: Literal["noul", "choice", "score"]
    choice: str | None = None
    probability: float | None = Field(default=None, ge=0, le=1)  # noul: P(yes)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0, le=1)


class DecisionResponse(_M):
    request_id: str
    provider: str
    model: str
    answers: dict[str, DecisionAnswer]
    latency_ms: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class NarrativeRequest(_M):
    request_id: str
    symbols: tuple[str, ...]
    lookback_minutes: int = 60
    allowed_handles: tuple[str, ...] = ()
    ts: int = 0
    model: str | None = None


class NarrativeItem(_M):
    symbol: str | None
    topic: str = "x_narrative"
    sentiment: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    summary: str = ""
    catalyst: str | None = None
    citations: tuple[str, ...] = ()
    observed_ts: int | None = None


class NarrativeResponse(_M):
    request_id: str
    provider: str
    model: str
    items: tuple[NarrativeItem, ...]
    latency_ms: int = 0
    degraded: bool = False  # e.g. xAI answered without matching X posts
    raw: dict[str, Any] = Field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


class DecisionProvider(Protocol):
    name: str

    async def list_models(self) -> list[str]: ...

    async def decide(self, req: DecisionRequest) -> DecisionResponse: ...


class NarrativeProvider(Protocol):
    name: str

    async def list_models(self) -> list[str]: ...

    async def scan(self, req: NarrativeRequest) -> NarrativeResponse: ...


DIRECTION_QUESTION = DecisionQuestion(
    key="direction",
    type="choice",
    instructions=(
        "Given only this market state, which position is most likely to be profitable over the next "
        "30 minutes after Bybit taker fees (~11 bps round trip)?"
    ),
    criteria={
        "long": "expected return clearly positive after costs",
        "short": "expected return clearly negative after costs",
        "wait": "no edge after costs, or too uncertain",
    },
)

REGIME_QUESTION = DecisionQuestion(
    key="regime",
    type="choice",
    instructions="Classify the current market regime.",
    criteria={
        "trend_up": "persistent upward drift",
        "trend_down": "persistent downward drift",
        "range": "mean-reverting, no persistent drift",
        "high_vol": "volatility expansion / disorderly",
    },
)
