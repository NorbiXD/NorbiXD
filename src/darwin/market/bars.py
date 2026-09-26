"""Time-bar aggregation.

Bars are the decision clock. A bar covering ``[start, end)`` is closed by the engine when the
first event with ``ts >= end`` arrives (or a timer fires), *before* that event is applied. All
sampled state (mark, funding, OI, book imbalance) is therefore information strictly known
before ``end`` — the structural guarantee against look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

from darwin.core.events import LiquidationEvent, TradeEvent
from darwin.core.types import Side
from darwin.market.state import SymbolState


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    start_ts: int
    end_ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    n_trades: int
    vwap: float
    mark_price: float
    funding_rate: float
    open_interest: float
    book_imbalance: float
    spread_bps: float
    liq_long_notional: float  # longs force-closed (sell pressure)
    liq_short_notional: float  # shorts force-closed (buy pressure)
    stale: bool


class BarBuilder:
    """Accumulates trades/liquidations for one symbol inside the current bar."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._reset()
        self.prev_close: float | None = None

    def _reset(self) -> None:
        self.open: float | None = None
        self.high = float("-inf")
        self.low = float("inf")
        self.close: float | None = None
        self.volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.notional = 0.0
        self.n_trades = 0
        self.liq_long = 0.0
        self.liq_short = 0.0

    def on_trade(self, ev: TradeEvent) -> None:
        if self.open is None:
            self.open = ev.price
        self.high = max(self.high, ev.price)
        self.low = min(self.low, ev.price)
        self.close = ev.price
        self.volume += ev.qty
        self.notional += ev.qty * ev.price
        if ev.taker_side is Side.BUY:
            self.buy_volume += ev.qty
        else:
            self.sell_volume += ev.qty
        self.n_trades += 1

    def on_liquidation(self, ev: LiquidationEvent) -> None:
        notional = ev.qty * ev.price
        if ev.side is Side.BUY:
            self.liq_long += notional
        else:
            self.liq_short += notional

    def close_bar(self, start_ts: int, end_ts: int, st: SymbolState, stale: bool) -> Bar | None:
        ref = st.ref_price(end_ts)
        if self.close is None:
            px = ref if ref is not None else self.prev_close
            if px is None:
                self._reset()
                return None
            o = h = lo = c = px
        else:
            assert self.open is not None
            o, h, lo, c = self.open, self.high, self.low, self.close
        vwap = self.notional / self.volume if self.volume > 0 else c
        imb = st.book.imbalance(5) if st.book.valid else None
        spread = st.book.spread_bps() if st.book.valid else None
        bar = Bar(
            symbol=self.symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=self.volume,
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            n_trades=self.n_trades,
            vwap=vwap,
            mark_price=ref or c,
            funding_rate=st.funding_rate or 0.0,
            open_interest=st.open_interest or 0.0,
            book_imbalance=imb if imb is not None else 0.0,
            spread_bps=spread if spread is not None else 0.0,
            liq_long_notional=self.liq_long,
            liq_short_notional=self.liq_short,
            stale=stale,
        )
        self.prev_close = c
        self._reset()
        return bar
