"""Shared enums and small value types."""

from __future__ import annotations

from enum import StrEnum


class Side(StrEnum):
    BUY = "Buy"
    SELL = "Sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @staticmethod
    def from_sign(x: float) -> Side:
        return Side.BUY if x > 0 else Side.SELL


class OrderType(StrEnum):
    MARKET = "Market"
    LIMIT = "Limit"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "PostOnly"


class OrderStatus(StrEnum):
    PENDING_NEW = "PendingNew"  # sent, no exchange acknowledgement yet
    NEW = "New"  # acknowledged / resting; NOT a fill
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Cancelled"
    REJECTED = "Rejected"
    UNKNOWN = "Unknown"  # exchange state could not be determined -> reconciliation required

    @property
    def terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


class Mode(StrEnum):
    REPLAY = "replay"  # historical / synthetic data, simulated execution, as fast as possible
    SIM = "sim"  # synthetic market paced in (accelerated) wall-clock time, simulated execution
    PAPER = "paper"  # live exchange market data, simulated execution
    TESTNET = "testnet"  # exchange testnet execution
    LIVE = "live"  # real capital. Off by default, triple-gated.


class Urgency(StrEnum):
    MARKET = "market"  # IOC market order with slippage cap
    PASSIVE = "passive"  # post-only limit at the touch


class AgentStatus(StrEnum):
    ALIVE = "alive"
    PROBATION = "probation"  # demoted: no capital, still shadow-evaluated
    DEAD = "dead"


class Regime(StrEnum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


class LineageEventKind(StrEnum):
    BORN = "born"  # random immigrant / seed
    MUTATED = "mutated"  # child of one parent
    CROSSOVER = "crossover"  # child of two parents
    PROPOSED = "proposed"  # from the Level-2 meta research sandbox
    PROMOTED = "promoted"  # became champion
    DEMOTED = "demoted"  # moved to probation
    REINSTATED = "reinstated"
    KILLED = "killed"
    FUNDED = "funded"
    DEFUNDED = "defunded"
