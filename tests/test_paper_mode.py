"""Integration: Bybit-format market stream -> WS client -> parser -> live driver -> engine ->
features -> agents -> governor -> simulated execution -> fills -> evolution -> leaderboard.

Runs against a local fake Bybit server on an accelerated clock (no network, no credentials).
"""

from __future__ import annotations

import asyncio
from typing import Any

from darwin.core.types import Mode
from darwin.exchange.bybit.feeds import BybitMarketFeed
from darwin.market.synthetic import SyntheticMarket
from darwin.runtime.app import _sim_venues
from darwin.runtime.engine import DarwinEngine
from darwin.runtime.live import Clock, LiveDriver
from tests.bybit_fakes import FakeBybitServer, to_bybit
from tests.conftest import make_config

SPEED = 1_500.0


async def test_paper_mode_full_pipeline_over_bybit_protocol() -> None:
    t0 = 1_700_000_040_000
    cfg = make_config(
        challenge={"duration_hours": 3, "symbols": ["BTCUSDT", "ETHUSDT"], "mode": Mode.PAPER.value},
        risk={"max_data_staleness_ms": 120_000},
        evolution={"population_size": 8, "generation_bars": 40, "min_trades": 2},
        allocator={"rebalance_bars": 20},
    )
    market = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=t0,
        duration_ms=cfg.duration_ms + 120_000,
        step_ms=5_000,
        seed=4,
    )
    msgs = [m for m in (to_bybit(e) for e in market.events()) if m is not None]
    clock = Clock(speed=SPEED, origin_ms=t0)

    def pace(m: dict[str, Any]) -> float:
        return clock.seconds_until(int(m["ts"]))

    async with FakeBybitServer(lambda conn: msgs if conn == 1 else [], pace=pace) as srv:
        engine = DarwinEngine(cfg, run_id="paper", start_ts=t0)
        driver = LiveDriver(engine, [], clock, heartbeat_ms=5_000, drain_timeout_ms=30_000)
        driver.venues = _sim_venues(cfg, engine, driver, challenge_sim=True)
        feed = BybitMarketFeed(driver, cfg.challenge.symbols, srv.url)
        engine.on_resync_needed = feed.resync
        engine.start()
        task = asyncio.create_task(feed.run())
        final = await asyncio.wait_for(driver.run(), timeout=60)
        task.cancel()

    st = engine.state()
    assert feed.ws.stats.messages > 1_000
    assert engine.bar_index >= 150
    assert st["stats"].get("intents", 0) > 10
    assert st["stats"].get("orders", 0) > 5 and st["stats"].get("fills", 0) > 5
    assert engine.population.generation >= 3
    assert engine.ended and final > 0
    board = engine.leaderboard()
    assert len(board) >= 8 and any(r["trades"] > 0 for r in board)
    assert st["stats"].get("sys:feed_connected", 0) >= 1
