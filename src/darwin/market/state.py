"""Per-symbol live market state assembled from normalized events."""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.core.events import (
    BookDelta,
    BookSnapshot,
    FundingSettlement,
    LiquidationEvent,
    TickerEvent,
    TradeEvent,
)
from darwin.market.book import OrderBook

MarketDataEvent = TradeEvent | BookSnapshot | BookDelta | TickerEvent | LiquidationEvent | FundingSettlement


@dataclass
class SymbolState:
    symbol: str
    book: OrderBook
    last_price: float | None = None
    last_trade_ts: int = 0
    mark_price: float | None = None
    index_price: float | None = None
    funding_rate: float | None = None
    next_funding_ts: int | None = None
    open_interest: float | None = None
    last_ticker_ts: int = 0
    seen_trade_ids: dict[str, None] = field(default_factory=dict)
    duplicate_trades: int = 0
    max_price_age_ms: int = 5_000

    def ref_price(self, now: int | None = None) -> float | None:
        """Best *fresh* fair price: mark, then book mid, then last trade.

        With ``now`` given, a source older than ``max_price_age_ms`` is skipped — a mark from a
        ticker stream that went quiet must never size an order. Returns None if nothing is fresh.
        Without ``now`` (venue-side use) the latest known value is returned.
        """
        mid = self.book.mid() if self.book.valid else None
        if now is None:
            return self.mark_price or mid or self.last_price
        age = self.max_price_age_ms
        if self.mark_price and now - self.last_ticker_ts <= age:
            return self.mark_price
        if mid and now - self.book.last_ts <= age:
            return mid
        if self.last_price and now - self.last_trade_ts <= age:
            return self.last_price
        return None

    def last_update_ts(self) -> int:
        return max(self.book.last_ts, self.last_trade_ts, self.last_ticker_ts)

    def staleness_ms(self, now: int) -> int:
        return now - self.last_update_ts()

    def is_stale(self, now: int, max_age_ms: int) -> bool:
        if not self.book.valid:
            return True
        if now - max(self.book.last_ts, self.last_trade_ts) > max_age_ms:
            return True
        return self.ref_price(now) is None


class MarketState:
    """Holds the latest known state for each symbol. Pure: no I/O, no clock."""

    _TRADE_ID_MEMORY = 5_000

    def __init__(
        self,
        symbols: tuple[str, ...] | list[str],
        strict_sequence: bool = True,
        max_price_age_ms: int = 5_000,
    ) -> None:
        self.symbols: dict[str, SymbolState] = {
            s: SymbolState(
                symbol=s,
                book=OrderBook(symbol=s, strict_sequence=strict_sequence),
                max_price_age_ms=max_price_age_ms,
            )
            for s in symbols
        }

    def __getitem__(self, symbol: str) -> SymbolState:
        return self.symbols[symbol]

    def __contains__(self, symbol: object) -> bool:
        return symbol in self.symbols

    def apply(self, ev: MarketDataEvent) -> bool:
        """Apply a market event. Returns False for duplicates/invalidating events."""
        st = self.symbols.get(ev.symbol)
        if st is None:
            return False
        if isinstance(ev, TradeEvent):
            if ev.trade_id in st.seen_trade_ids:
                st.duplicate_trades += 1
                return False
            st.seen_trade_ids[ev.trade_id] = None
            if len(st.seen_trade_ids) > self._TRADE_ID_MEMORY:
                # dicts preserve insertion order: drop the oldest id
                st.seen_trade_ids.pop(next(iter(st.seen_trade_ids)))
            st.last_price = ev.price
            st.last_trade_ts = ev.ts
            return True
        if isinstance(ev, BookSnapshot):
            st.book.apply_snapshot(ev)
            return st.book.valid
        if isinstance(ev, BookDelta):
            return st.book.apply_delta(ev)
        if isinstance(ev, TickerEvent):
            if ev.last_price is not None:
                st.last_price = ev.last_price
            if ev.mark_price is not None:
                st.mark_price = ev.mark_price
            if ev.index_price is not None:
                st.index_price = ev.index_price
            if ev.funding_rate is not None:
                st.funding_rate = ev.funding_rate
            if ev.next_funding_ts is not None:
                st.next_funding_ts = ev.next_funding_ts
            if ev.open_interest is not None:
                st.open_interest = ev.open_interest
            st.last_ticker_ts = ev.ts
            return True
        # liquidations / funding settlements carry no book state
        return True
