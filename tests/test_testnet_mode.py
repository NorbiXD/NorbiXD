"""Integration: the real testnet wiring end to end against local fakes (no network, no keys).

``build_runtime(mode=testnet)`` → account preflight over REST → public market stream (fake V5
WebSocket) → agents → governor → ``BybitExecutionGateway`` → REST ``/v5/order/create`` (ACK) →
private WebSocket ``execution`` / ``order`` / ``position`` messages (fills) → ledger →
reconciliation → end-of-challenge flatten through the same path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from darwin.core.types import Mode
from darwin.exchange.bybit.ws import BybitEndpoints
from darwin.market.synthetic import SyntheticMarket
from darwin.runtime.app import build_runtime
from darwin.runtime.engine import CHALLENGE, CHALLENGE_VENUE
from darwin.runtime.live import Clock
from tests.bybit_fakes import FakeBybitExchange, FakeBybitServer, to_bybit
from tests.conftest import make_config
from tests.test_flatten import fund_everyone

SPEED = 1_500.0


async def test_testnet_mode_trades_reconciles_and_ends_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", "test-key")
    monkeypatch.setenv("BYBIT_TESTNET_API_SECRET", "test-secret")
    t0 = 1_700_000_040_000
    symbols = ["ETHUSDT", "SOLUSDT"]
    cfg = make_config(
        challenge={"duration_hours": 2, "symbols": symbols, "mode": Mode.TESTNET.value},
        risk={"max_data_staleness_ms": 120_000},
        evolution={"population_size": 6, "generation_bars": 40, "min_trades": 2},
        allocator={"rebalance_bars": 5},
        exchange={"reconcile_interval_ms": 60_000},
        persistence={"record_market_data": False},
    )
    market = SyntheticMarket(
        symbols=tuple(symbols), start_ts=t0, duration_ms=cfg.duration_ms + 180_000, step_ms=5_000, seed=8
    )
    msgs = [m for m in (to_bybit(e) for e in market.events()) if m is not None]
    clock = Clock(speed=SPEED, origin_ms=t0)
    fake = FakeBybitExchange(equity=1_000.0)

    def pace(m: dict[str, Any]) -> float:
        return clock.seconds_until(int(m["ts"]))

    async with (
        FakeBybitServer(lambda conn: msgs if conn == 1 else [], pace=pace) as public,
        FakeBybitServer(lambda conn: [], live=fake.private) as private,
    ):
        rt = build_runtime(
            cfg,
            None,
            "testnet-e2e",
            start_ts=t0,
            endpoints=BybitEndpoints(public.url, private.url, "https://fake-bybit.test"),
            rest_transport=httpx.MockTransport(fake),
            clock=clock,
        )
        eng = rt.engine
        eng.allocator.allocate = fund_everyone  # type: ignore[method-assign]
        fake.mark = lambda s: eng.market[s].ref_price()
        final = await asyncio.wait_for(rt.run(), timeout=90)

    paths = [p for p, _ in fake.calls]
    # preflight: wallet, flatness, open orders, margin mode, one-way, per-symbol leverage
    for p in ("/v5/account/wallet-balance", "/v5/position/list", "/v5/order/realtime", "/v5/account/info"):
        assert p in paths, p
    assert paths.count("/v5/position/set-leverage") == len(symbols)
    creates = [a for p, a in fake.calls if p == "/v5/order/create"]
    assert len(creates) >= 5
    assert all(a["category"] == "linear" and a["positionIdx"] == 0 for a in creates)
    assert all(a["orderLinkId"].startswith("C") and len(a["orderLinkId"]) <= 36 for a in creates)
    # fills arrived over the private stream and were attributed to agents
    chal_orders = [o for o in eng.execution.orders.values() if o.account == CHALLENGE]
    filled = [o for o in chal_orders if o.exec_ids]
    assert len(filled) >= 5 and not eng.execution.orphans
    assert eng.stats["sys:feed_connected"] >= 2  # public + private
    # the run ended flat on both sides, with the venue and ledger agreeing
    assert eng.ended and final > 0
    assert not eng.ledger[CHALLENGE].open_positions() and not eng.execution.open_orders()
    assert all(q == 0 for q, _ in fake.positions.values())
    assert eng.stats["sys:flatten_complete"] == 1 and eng.stats["sys:flatten_incomplete"] == 0
    h = eng.health[CHALLENGE_VENUE]
    assert h.reconcile_ok, h.reconcile_detail
    assert eng.stats["sys:reconcile_mismatch"] == 0 and eng.stats["sys:wallet_drift"] == 0
    venue_pnl = fake.equity() - 1_000.0
    ledger_pnl = eng.ledger[CHALLENGE].equity(eng._marks()) - cfg.challenge.starting_capital
    assert venue_pnl == pytest.approx(ledger_pnl, abs=0.05)
