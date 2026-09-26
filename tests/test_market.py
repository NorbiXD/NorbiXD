from __future__ import annotations

from darwin.core.events import BookDelta, BookSnapshot
from darwin.market.book import OrderBook
from darwin.market.state import MarketState
from tests.conftest import T0, book, ticker, trade


def test_snapshot_then_delta_updates_levels() -> None:
    ob = OrderBook("BTCUSDT")
    ob.apply_snapshot(
        BookSnapshot(
            ts=T0, symbol="BTCUSDT", bids=((99, 1), (98, 2)), asks=((101, 1), (102, 2)), update_id=10
        )
    )
    assert ob.valid and ob.best_bid() == 99 and ob.best_ask() == 101
    ok = ob.apply_delta(
        BookDelta(ts=T0 + 1, symbol="BTCUSDT", bids=((99, 0), (100, 3)), asks=(), update_id=11)
    )
    assert ok and ob.best_bid() == 100 and 99 not in ob.bids
    assert ob.imbalance(5) == (5 - 3) / 8


def test_duplicate_delta_is_ignored_idempotently() -> None:
    ob = OrderBook("BTCUSDT")
    ob.apply_snapshot(BookSnapshot(ts=T0, symbol="BTCUSDT", bids=((99, 1),), asks=((101, 1),), update_id=10))
    d = BookDelta(ts=T0 + 1, symbol="BTCUSDT", bids=((99, 5),), asks=(), update_id=11)
    ob.apply_delta(d)
    ob.apply_delta(d)  # replayed message
    assert ob.bids[99] == 5 and ob.duplicates == 1 and ob.valid


def test_sequence_gap_invalidates_until_snapshot() -> None:
    ob = OrderBook("BTCUSDT")
    ob.apply_snapshot(BookSnapshot(ts=T0, symbol="BTCUSDT", bids=((99, 1),), asks=((101, 1),), update_id=10))
    assert not ob.apply_delta(BookDelta(ts=T0 + 1, symbol="BTCUSDT", bids=((99, 2),), asks=(), update_id=13))
    assert not ob.valid and "gap" in ob.invalid_reason and ob.gaps == 1
    # subsequent deltas are not trusted until a fresh snapshot arrives
    assert not ob.apply_delta(BookDelta(ts=T0 + 2, symbol="BTCUSDT", bids=((99, 7),), asks=(), update_id=14))
    assert ob.bids[99] == 1
    ob.apply_snapshot(
        BookSnapshot(ts=T0 + 3, symbol="BTCUSDT", bids=((98, 1),), asks=((100, 1),), update_id=1)
    )
    assert ob.valid and ob.best_bid() == 98  # u=1 snapshot after an exchange restart resets the sequence


def test_crossed_book_is_invalid() -> None:
    ob = OrderBook("BTCUSDT")
    ob.apply_snapshot(BookSnapshot(ts=T0, symbol="BTCUSDT", bids=((99, 1),), asks=((101, 1),), update_id=1))
    ob.apply_delta(BookDelta(ts=T0 + 1, symbol="BTCUSDT", bids=((102, 1),), asks=(), update_id=2))
    assert not ob.valid and "crossed" in ob.invalid_reason


def test_delta_before_snapshot_is_rejected() -> None:
    ob = OrderBook("BTCUSDT")
    assert not ob.apply_delta(BookDelta(ts=T0, symbol="BTCUSDT", bids=((99, 1),), asks=(), update_id=5))
    assert not ob.valid


def test_market_state_dedupes_trades_and_tracks_staleness() -> None:
    ms = MarketState(["BTCUSDT"])
    ms.apply(book("BTCUSDT", T0, 100.0))
    assert ms.apply(trade("BTCUSDT", T0 + 10, 100.0, tid="x1"))
    assert not ms.apply(trade("BTCUSDT", T0 + 20, 101.0, tid="x1"))  # duplicate trade id
    st = ms["BTCUSDT"]
    assert st.last_price == 100.0 and st.duplicate_trades == 1
    assert not st.is_stale(T0 + 1_000, 5_000)
    assert st.is_stale(T0 + 10_000, 5_000)
    ms.apply(ticker("BTCUSDT", T0 + 30, 100.5))
    assert st.ref_price() == 100.5  # mark preferred


def test_unknown_symbol_ignored() -> None:
    ms = MarketState(["BTCUSDT"])
    assert not ms.apply(trade("DOGEUSDT", T0, 0.1))
