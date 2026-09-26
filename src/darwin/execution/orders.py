"""Order requests, the venue gateway protocol and the order state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from darwin.core.types import OrderStatus, OrderType, Side, TimeInForce


class OrderRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str = Field(max_length=36)  # Bybit orderLinkId limit
    account: str
    symbol: str
    side: Side
    qty: float = Field(gt=0)
    order_type: OrderType
    tif: TimeInForce
    limit_price: float | None = Field(default=None, gt=0)
    reduce_only: bool = False
    ts: int


class ExecutionGateway(Protocol):
    """Outbound side of a venue. All results come back asynchronously as events."""

    venue: str

    def submit(self, order: OrderRequest) -> None: ...

    def cancel(self, account: str, client_order_id: str, symbol: str, ts: int) -> None: ...

    def query_order(self, account: str, client_order_id: str, symbol: str, ts: int) -> None: ...

    def query_positions(self, account: str, ts: int) -> None: ...


# Legal forward transitions. Anything else is ignored as stale/out-of-order.
_RANK = {
    OrderStatus.PENDING_NEW: 0,
    OrderStatus.UNKNOWN: 0,
    OrderStatus.NEW: 1,
    OrderStatus.PARTIALLY_FILLED: 2,
    OrderStatus.FILLED: 3,
    OrderStatus.CANCELED: 3,
    OrderStatus.REJECTED: 3,
}


@dataclass
class ManagedOrder:
    request: OrderRequest
    agent_id: str
    genome_id: str
    intent_id: str
    decision_id: str
    intent_reason: str
    confidence: float
    ref_price: float
    stop_loss_pct: float | None
    take_profit_pct: float | None
    max_hold_ms: int | None
    risk_increasing: bool = True
    status: OrderStatus = OrderStatus.PENDING_NEW
    exchange_order_id: str | None = None
    filled_qty: float = 0.0  # from fills (authoritative for positions)
    reported_cum_qty: float = 0.0  # from order updates (informational until fills arrive)
    avg_fill_price: float = 0.0
    fees: float = 0.0
    created_ts: int = 0
    acked_ts: int | None = None
    last_update_ts: int = 0
    queries: int = 0
    reason: str = ""
    exec_ids: list[str] = field(default_factory=list)

    @property
    def client_order_id(self) -> str:
        return self.request.client_order_id

    @property
    def account(self) -> str:
        return self.request.account

    @property
    def symbol(self) -> str:
        return self.request.symbol

    @property
    def unfilled(self) -> float:
        return max(self.request.qty - self.filled_qty, 0.0)

    @property
    def awaiting_fills(self) -> float:
        """Quantity the venue reports executed but whose executions have not reached us yet.

        Bybit's ``order`` and ``execution`` streams are not ordered relative to each other: a
        ``Filled`` update can arrive before the fills. Until they do, the order still carries
        exposure the ledger cannot see.
        """
        gap = self.reported_cum_qty - self.filled_qty
        return gap if gap > 1e-12 else 0.0

    @property
    def remaining(self) -> float:
        """Exposure still in flight: unfilled qty while working, missing fills once terminal."""
        return self.awaiting_fills if self.status.terminal else self.unfilled

    @property
    def signed_remaining(self) -> float:
        return self.remaining * self.request.side.sign

    @property
    def open(self) -> bool:
        return not self.status.terminal or self.awaiting_fills > 0

    def apply_status(self, status: OrderStatus, ts: int) -> bool:
        """Advance the state machine. Returns False if the update was stale/illegal."""
        if self.status.terminal:
            return False
        if _RANK[status] < _RANK[self.status]:
            return False
        self.status = status
        self.last_update_ts = ts
        if status is not OrderStatus.PENDING_NEW and self.acked_ts is None:
            self.acked_ts = ts
        return True

    def apply_fill(self, qty: float, price: float, fee: float, exec_id: str, ts: int) -> None:
        total = self.filled_qty + qty
        self.avg_fill_price = (self.avg_fill_price * self.filled_qty + price * qty) / total
        self.filled_qty = total
        self.fees += fee
        self.exec_ids.append(exec_id)
        self.last_update_ts = ts
        if self.acked_ts is None:
            self.acked_ts = ts  # a fill implies acceptance even if the ACK is late/lost
        if self.unfilled <= 1e-12:
            self.status = OrderStatus.FILLED
        elif not self.status.terminal:
            self.status = OrderStatus.PARTIALLY_FILLED
