"""Bybit V5 (linear perpetuals) message -> normalized event translation.

Formats follow the V5 public/private WebSocket documentation:

* ``orderbook.{depth}.{symbol}``  snapshot/delta with ``u`` (update id) and ``seq``;
  ``u == 1`` in a snapshot means the service restarted and the book must be rebuilt.
* ``publicTrade.{symbol}``        list of trades; ``i`` is the trade id, ``S`` the taker side.
* ``tickers.{symbol}``            snapshot then deltas containing only changed fields.
* ``allLiquidation.{symbol}``     ``S`` is the *position* side liquidated (Buy = long).
* private ``order`` / ``execution`` / ``position`` / ``wallet``.

Numbers arrive as strings. Every event is stamped with ``recv_ts`` (local receive time), which
is the engine's clock; exchange time is kept in ``exch_ts``.
"""

from __future__ import annotations

from typing import Any

from darwin.core.events import (
    BookDelta,
    BookSnapshot,
    Event,
    FillEvent,
    FundingPayment,
    LiquidationEvent,
    OrderUpdate,
    PositionSnapshot,
    TickerEvent,
    TradeEvent,
    WalletSnapshot,
)
from darwin.core.types import OrderStatus, Side

ORDER_STATUS_MAP: dict[str, OrderStatus] = {
    "Created": OrderStatus.PENDING_NEW,
    "New": OrderStatus.NEW,
    "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELED,
    "PartiallyFilledCanceled": OrderStatus.CANCELED,
    "Deactivated": OrderStatus.CANCELED,
    "Rejected": OrderStatus.REJECTED,
    "Untriggered": OrderStatus.NEW,
    "Triggered": OrderStatus.NEW,
}


def _f(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x: Any) -> int | None:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _levels(raw: list[list[str]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(p), float(q)) for p, q in raw)


def parse_public(msg: dict[str, Any], recv_ts: int) -> list[Event]:
    topic = msg.get("topic")
    if not isinstance(topic, str):
        return []
    data = msg.get("data")
    exch_ts = _i(msg.get("ts"))
    if topic.startswith("orderbook."):
        assert isinstance(data, dict)
        sym = data["s"]
        bids = _levels(data.get("b", []))
        asks = _levels(data.get("a", []))
        uid = int(data["u"])
        seq = _i(data.get("seq"))
        if msg.get("type") == "snapshot":
            return [
                BookSnapshot(
                    ts=recv_ts, symbol=sym, bids=bids, asks=asks, update_id=uid, seq=seq, exch_ts=exch_ts
                )
            ]
        return [
            BookDelta(ts=recv_ts, symbol=sym, bids=bids, asks=asks, update_id=uid, seq=seq, exch_ts=exch_ts)
        ]
    if topic.startswith("publicTrade."):
        out: list[Event] = []
        for t in data or []:
            out.append(
                TradeEvent(
                    ts=recv_ts,
                    symbol=t["s"],
                    price=float(t["p"]),
                    qty=float(t["v"]),
                    taker_side=Side(t["S"]),
                    trade_id=str(t["i"]),
                    exch_ts=_i(t.get("T")),
                )
            )
        return out
    if topic.startswith("tickers."):
        assert isinstance(data, dict)
        return [
            TickerEvent(
                ts=recv_ts,
                symbol=data["symbol"],
                last_price=_f(data.get("lastPrice")),
                mark_price=_f(data.get("markPrice")),
                index_price=_f(data.get("indexPrice")),
                funding_rate=_f(data.get("fundingRate")),
                next_funding_ts=_i(data.get("nextFundingTime")),
                open_interest=_f(data.get("openInterest")),
                exch_ts=exch_ts,
            )
        ]
    if topic.startswith(("allLiquidation.", "liquidation.")):
        rows: list[dict[str, Any]] = (
            data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        )
        out = []
        for r in rows:
            sym = r.get("s") or r.get("symbol")
            price = _f(r.get("p") or r.get("price"))
            qty = _f(r.get("v") or r.get("size"))
            side = r.get("S") or r.get("side")
            if not sym or not price or not qty or side not in ("Buy", "Sell"):
                continue
            out.append(
                LiquidationEvent(
                    ts=recv_ts,
                    symbol=sym,
                    side=Side(side),
                    price=price,
                    qty=qty,
                    exch_ts=_i(r.get("T") or r.get("updatedTime")),
                )
            )
        return out
    return []


def parse_private(msg: dict[str, Any], recv_ts: int, account: str) -> list[Event]:
    topic = msg.get("topic")
    rows = msg.get("data") or []
    out: list[Event] = []
    if topic == "order":
        for r in rows:
            if r.get("category") not in (None, "linear"):
                continue
            status = ORDER_STATUS_MAP.get(r.get("orderStatus", ""), OrderStatus.UNKNOWN)
            reason = r.get("rejectReason") or ""
            if reason == "EC_NoError":
                reason = ""
            if r.get("orderStatus") == "PartiallyFilledCanceled":
                reason = reason or "ioc_remainder_cancelled"
            out.append(
                OrderUpdate(
                    ts=recv_ts,
                    account=account,
                    client_order_id=r.get("orderLinkId") or r["orderId"],
                    exchange_order_id=r.get("orderId"),
                    symbol=r["symbol"],
                    status=status,
                    cum_qty=_f(r.get("cumExecQty")) or 0.0,
                    avg_price=_f(r.get("avgPrice")),
                    reason=reason,
                    exch_ts=_i(r.get("updatedTime")),
                )
            )
    elif topic == "execution":
        for r in rows:
            if r.get("category") not in (None, "linear"):
                continue
            exec_type = r.get("execType", "Trade")
            qty = _f(r.get("execQty")) or 0.0
            price = _f(r.get("execPrice")) or 0.0
            if exec_type == "Funding":
                # Bybit reports funding as an execution; execFee is the amount paid (+) / received (-)
                size = _f(r.get("execQty")) or 0.0
                side = Side(r["side"])
                out.append(
                    FundingPayment(
                        ts=recv_ts,
                        account=account,
                        symbol=r["symbol"],
                        position_qty=size * side.sign,
                        rate=_f(r.get("feeRate")) or 0.0,
                        mark_price=price,
                        amount=_f(r.get("execFee")) or 0.0,
                    )
                )
                continue
            if qty <= 0 or price <= 0:
                continue
            out.append(
                FillEvent(
                    ts=recv_ts,
                    account=account,
                    client_order_id=r.get("orderLinkId") or r.get("orderId", ""),
                    exec_id=str(r["execId"]),
                    symbol=r["symbol"],
                    side=Side(r["side"]),
                    qty=qty,
                    price=price,
                    fee=_f(r.get("execFee")) or 0.0,
                    is_maker=bool(r.get("isMaker", False)),
                    is_liquidation=exec_type in ("BustTrade", "AdlTrade"),
                    exch_ts=_i(r.get("execTime")),
                )
            )
    elif topic == "position":
        for r in rows:
            if r.get("category") not in (None, "linear"):
                continue
            size = _f(r.get("size")) or 0.0
            side = r.get("side")
            qty = size if side == "Buy" else -size if side == "Sell" else 0.0
            out.append(
                PositionSnapshot(
                    ts=recv_ts,
                    account=account,
                    symbol=r["symbol"],
                    qty=qty,
                    entry_price=_f(r.get("entryPrice")) or _f(r.get("avgPrice")),
                )
            )
    elif topic == "wallet":
        for r in rows:
            eq = _f(r.get("totalEquity"))
            if eq is None:
                continue
            out.append(
                WalletSnapshot(
                    ts=recv_ts,
                    account=account,
                    equity=eq,
                    wallet_balance=_f(r.get("totalWalletBalance")) or eq,
                    available=_f(r.get("totalAvailableBalance")),
                )
            )
    return out
