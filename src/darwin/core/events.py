"""Normalized, exchange-agnostic event schemas.

Time convention
---------------
Every event carries ``ts``: the millisecond timestamp at which the *system* learned about it
(local receive time in live trading, the recorded receive time in replay). This is the only
clock the engine uses, which is what makes replay deterministic and look-ahead-free: nothing
can influence a decision before its ``ts``. Exchange-side timestamps, where available, are kept
separately in ``exch_ts`` for latency analytics only.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from darwin.core.types import OrderStatus, Side

Level = tuple[float, float]  # (price, qty)


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: int = Field(ge=0, description="ms timestamp when the system learned of this event")


# --------------------------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------------------------


class TradeEvent(Event):
    kind: Literal["trade"] = "trade"
    symbol: str
    price: float = Field(gt=0)
    qty: float = Field(gt=0)
    taker_side: Side
    trade_id: str
    exch_ts: int | None = None


class BookSnapshot(Event):
    """Full replacement of the local book (Bybit 'snapshot' or a resync)."""

    kind: Literal["book_snapshot"] = "book_snapshot"
    symbol: str
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    update_id: int
    seq: int | None = None
    exch_ts: int | None = None


class BookDelta(Event):
    """Incremental update. A level with qty == 0 is removed."""

    kind: Literal["book_delta"] = "book_delta"
    symbol: str
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    update_id: int
    seq: int | None = None
    exch_ts: int | None = None


class TickerEvent(Event):
    """Mark/index/funding/OI. Any field may be None in a delta (= unchanged)."""

    kind: Literal["ticker"] = "ticker"
    symbol: str
    last_price: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    funding_rate: float | None = None
    next_funding_ts: int | None = None
    open_interest: float | None = None
    exch_ts: int | None = None


class LiquidationEvent(Event):
    """A forced liquidation on the exchange. ``side`` is the side of the liquidated *position*."""

    kind: Literal["liquidation"] = "liquidation"
    symbol: str
    side: Side
    price: float = Field(gt=0)
    qty: float = Field(gt=0)
    exch_ts: int | None = None


class FundingSettlement(Event):
    """Emitted at a funding timestamp; the venue applies it to open positions."""

    kind: Literal["funding"] = "funding"
    symbol: str
    rate: float
    mark_price: float = Field(gt=0)


MarketEvent = Annotated[
    TradeEvent | BookSnapshot | BookDelta | TickerEvent | LiquidationEvent | FundingSettlement,
    Field(discriminator="kind"),
]


class FeedStatus(Event):
    """Connection-level status (connected, disconnected, resync, gap) for a feed."""

    kind: Literal["feed_status"] = "feed_status"
    feed: str
    symbol: str | None = None
    status: Literal["connected", "disconnected", "resync", "gap", "stale"]
    detail: str = ""


# --------------------------------------------------------------------------------------------
# Execution (venue -> engine)
# --------------------------------------------------------------------------------------------


class OrderUpdate(Event):
    """Order lifecycle update from a venue. ``cum_qty`` is cumulative and must be monotonic."""

    kind: Literal["order_update"] = "order_update"
    account: str
    client_order_id: str
    exchange_order_id: str | None = None
    symbol: str
    status: OrderStatus
    cum_qty: float = 0.0
    avg_price: float | None = None
    reason: str = ""
    exch_ts: int | None = None


class FillEvent(Event):
    """An execution. ``exec_id`` is globally unique per venue and is the idempotency key."""

    kind: Literal["fill"] = "fill"
    account: str
    client_order_id: str
    exec_id: str
    symbol: str
    side: Side
    qty: float = Field(gt=0)
    price: float = Field(gt=0)
    fee: float  # in quote currency, positive = paid
    is_maker: bool = False
    is_liquidation: bool = False
    exch_ts: int | None = None


class FundingPayment(Event):
    """Funding applied to an account position (positive amount = paid by us)."""

    kind: Literal["funding_payment"] = "funding_payment"
    account: str
    symbol: str
    position_qty: float
    rate: float
    mark_price: float
    amount: float


class PositionSnapshot(Event):
    """Venue's view of the net position; used for reconciliation, never as the ledger source."""

    kind: Literal["position_snapshot"] = "position_snapshot"
    account: str
    symbol: str
    qty: float  # signed
    entry_price: float | None = None


class WalletSnapshot(Event):
    kind: Literal["wallet_snapshot"] = "wallet_snapshot"
    account: str
    equity: float
    wallet_balance: float
    available: float | None = None


ExecutionEvent = OrderUpdate | FillEvent | FundingPayment | PositionSnapshot | WalletSnapshot


# --------------------------------------------------------------------------------------------
# Timers / intelligence
# --------------------------------------------------------------------------------------------


class TimerEvent(Event):
    kind: Literal["timer"] = "timer"
    name: str


class IntelligenceSignal(Event):
    """Output of the slow path (LLM narrative, X search, external feed, ...).

    ``ts`` is the *availability* time (when the signal reached the system). ``observed_ts`` is
    the time of the underlying content (e.g. when a post was published). Agents may only see a
    signal once ``ts`` <= now, which is what prevents leakage in replay.
    """

    kind: Literal["intelligence"] = "intelligence"
    signal_id: str
    source: str
    symbol: str | None = None
    topic: str = ""
    value: float = Field(ge=-1.0, le=1.0, description="directional score, -1 bearish .. +1 bullish")
    confidence: float = Field(ge=0.0, le=1.0)
    half_life_ms: int = Field(default=3_600_000, gt=0)
    observed_ts: int | None = None
    provider: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
