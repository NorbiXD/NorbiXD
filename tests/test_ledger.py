from __future__ import annotations

import pytest

from darwin.core.types import Side
from darwin.portfolio.ledger import Ledger, Position


def test_position_open_add_reduce_flip() -> None:
    p = Position()
    assert p.apply_fill(Side.BUY, 1.0, 100.0) == 0.0
    p.apply_fill(Side.BUY, 1.0, 110.0)
    assert p.qty == 2.0 and p.avg_price == pytest.approx(105.0)
    r = p.apply_fill(Side.SELL, 0.5, 115.0)
    assert r == pytest.approx(5.0) and p.qty == 1.5 and p.avg_price == pytest.approx(105.0)
    r = p.apply_fill(Side.SELL, 2.5, 100.0)  # close 1.5 @ -5 and open short 1.0 @ 100
    assert r == pytest.approx(-7.5) and p.qty == pytest.approx(-1.0) and p.avg_price == 100.0
    assert p.unrealized(90.0) == pytest.approx(10.0)


def fill(
    led: Ledger,
    side: Side,
    qty: float,
    price: float,
    ts: int,
    intent: str,
    agent: str = "A1",
    fee: float = 0.1,
):  # type: ignore[no-untyped-def]
    return led.on_fill(
        account_id="acc",
        agent_id=agent,
        genome_id="G",
        symbol="BTCUSDT",
        side=side,
        qty=qty,
        price=price,
        fee=fee,
        ts=ts,
        intent_id=intent,
        intent_reason="entry",
        confidence=0.7,
        ref_price=price - 0.05 * side.sign,
        equity_hint=1_000.0,
        stop_loss_pct=0.02,
    )


def test_round_trip_lifecycle_and_accounting() -> None:
    led = Ledger()
    acct = led.open_account("acc", "shadow", 1_000.0)
    assert fill(led, Side.BUY, 1.0, 100.0, 1, "I1") == []
    led.mark({"BTCUSDT": 104.0}, {"BTCUSDT": 106.0}, {"BTCUSDT": 97.0}, ts=120_000, bar_start_ts=60_000)
    closed = fill(led, Side.SELL, 1.0, 105.0, 130_000, "I2")
    assert len(closed) == 1
    rt = closed[0]
    assert rt.realized == pytest.approx(5.0)
    assert rt.fees == pytest.approx(0.2)
    assert rt.net_pnl == pytest.approx(4.8)
    assert rt.mfe_pct == pytest.approx(0.06) and rt.mae_pct == pytest.approx(-0.03)
    assert rt.entry_intent_id == "I1" and rt.exit_intent_id == "I2"
    assert rt.slippage_cost == pytest.approx(0.1)  # 0.05 on entry + 0.05 on exit
    assert acct.cash == pytest.approx(1_000 + 5.0 - 0.2)


def test_flip_closes_and_opens_new_round_trip() -> None:
    led = Ledger()
    led.open_account("acc", "shadow", 1_000.0)
    fill(led, Side.BUY, 1.0, 100.0, 1, "I1")
    closed = fill(led, Side.SELL, 2.0, 110.0, 2, "I2")
    assert len(closed) == 1 and closed[0].realized == pytest.approx(10.0)
    open_rt = led.open_trades[("acc", "A1", "BTCUSDT")]
    assert open_rt.direction == -1 and open_rt.entry_intent_id == "I2"


def test_funding_distributed_by_agent_position() -> None:
    led = Ledger()
    acct = led.open_account("acc", "challenge", 1_000.0)
    fill(led, Side.BUY, 2.0, 100.0, 1, "I1", agent="A1", fee=0.0)
    fill(led, Side.SELL, 1.0, 100.0, 1, "I2", agent="A2", fee=0.0)
    cash0 = acct.cash
    residual = led.on_funding("acc", "BTCUSDT", 0.001, 100.0)
    assert residual == 0.0 and cash0 - acct.cash == pytest.approx(0.1)  # net 1.0 long * 100 * 0.001
    assert acct.pos("A1", "BTCUSDT").funding == pytest.approx(0.2)
    assert acct.pos("A2", "BTCUSDT").funding == pytest.approx(-0.1)
    assert acct.net_qty("BTCUSDT") == pytest.approx(1.0)


def test_mfe_ignores_pre_entry_bar_extremes() -> None:
    led = Ledger()
    led.open_account("acc", "shadow", 1_000.0)
    fill(led, Side.BUY, 1.0, 100.0, 30_000, "I1")  # entered mid-bar
    led.mark({"BTCUSDT": 101.0}, {"BTCUSDT": 120.0}, {"BTCUSDT": 80.0}, ts=60_000, bar_start_ts=0)
    rt = led.open_trades[("acc", "A1", "BTCUSDT")]
    assert rt.mfe_pct == pytest.approx(0.01) and rt.mae_pct == 0.0
