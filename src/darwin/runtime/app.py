"""Mode wiring: replay / sim / paper / testnet / live all build the *same* DarwinEngine.

    replay   synthetic or recorded data, simulated venues, as fast as possible (deterministic)
    sim      synthetic data paced at ``sim.speed`` x wall-clock, simulated venues (demo/dashboard)
    paper    Bybit public streams, simulated venues
    testnet  Bybit testnet public+private streams; challenge orders go to testnet
    live     Bybit mainnet. OFF by default and triple-gated (see ``assert_live_allowed``)

In every mode the shadow evaluation venue is simulated: evolution evidence never costs money.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from darwin.agents.genome import Genome
from darwin.config.challenge import ChallengeConfig
from darwin.core.events import Event
from darwin.core.types import Mode
from darwin.exchange.bybit.feeds import BybitMarketFeed, BybitPrivateFeed
from darwin.exchange.bybit.gateway import BybitExecutionGateway
from darwin.exchange.bybit.preflight import PreflightReport, account_preflight
from darwin.exchange.bybit.rest import BybitRest
from darwin.exchange.bybit.signing import Credentials
from darwin.exchange.bybit.ws import MAINNET, TESTNET
from darwin.exchange.sim.venue import SimExchange
from darwin.intelligence.factory import build_intelligence
from darwin.market.synthetic import SyntheticMarket
from darwin.persistence.store import AuditStore
from darwin.replay.recorder import ParquetRecorder
from darwin.runtime.engine import CHALLENGE, CHALLENGE_VENUE, SHADOW_VENUE, DarwinEngine
from darwin.runtime.live import Clock, LiveDriver, SyntheticFeed
from darwin.runtime.replay import build_replay
from darwin.signals.external import WebhookSignalFeed, webhook_feeds_from_env

log = logging.getLogger(__name__)

LIVE_CONFIRM_ENV = "DARWIN_LIVE_CONFIRM"
LIVE_CONFIRM_VALUE = "I_UNDERSTAND_THIS_TRADES_REAL_MONEY"


class LiveTradingNotAllowed(RuntimeError):
    pass


def assert_live_allowed(cfg: ChallengeConfig, mode: Mode, env: dict[str, str] | None = None) -> None:
    """Real-money trading requires all three: config flag, explicit env confirmation, keys."""
    if mode is not Mode.LIVE:
        return
    env = dict(os.environ) if env is None else env
    problems = []
    if not cfg.challenge.live_enabled:
        problems.append("challenge.live_enabled is false in challenge.yaml")
    if env.get(LIVE_CONFIRM_ENV) != LIVE_CONFIRM_VALUE:
        problems.append(f"{LIVE_CONFIRM_ENV} != {LIVE_CONFIRM_VALUE}")
    if not env.get("BYBIT_API_KEY") or not env.get("BYBIT_API_SECRET"):
        problems.append("BYBIT_API_KEY / BYBIT_API_SECRET not set")
    if problems:
        raise LiveTradingNotAllowed("live trading refused: " + "; ".join(problems))


@dataclass
class Runtime:
    engine: DarwinEngine
    run: Callable[[], Awaitable[float]]
    cleanup: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    webhooks: dict[str, WebhookSignalFeed] = field(default_factory=dict)


def synthetic_stream(cfg: ChallengeConfig, start_ts: int, planted: bool = True) -> Iterable[Event]:
    m = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=start_ts,
        duration_ms=cfg.duration_ms + 2 * cfg.challenge.bar_ms,
        seed=cfg.sim.seed,
        step_ms=cfg.sim.synthetic_step_ms,
        book_every=cfg.sim.synthetic_book_every,
        planted_edges=planted,
    )
    return m.events()


def _sim_venues(
    cfg: ChallengeConfig, engine: DarwinEngine, scheduler: Any, challenge_sim: bool
) -> list[SimExchange]:
    instruments = {s: cfg.instrument(s) for s in cfg.challenge.symbols}
    common: dict[str, Any] = {
        "symbols": cfg.challenge.symbols,
        "fees": cfg.exchange.fees,
        "instruments": instruments,
        "sim": cfg.sim,
        "scheduler": scheduler,
        "engine": engine,
        "maintenance_margin_rate": cfg.risk.maintenance_margin_rate,
    }
    venues = [SimExchange(SHADOW_VENUE, seed_offset=1, **common)]
    engine.attach_gateway(SHADOW_VENUE, venues[0])
    if challenge_sim:
        venues.append(SimExchange(CHALLENGE_VENUE, seed_offset=2, **common))
        engine.attach_gateway(CHALLENGE_VENUE, venues[1])
    return venues


def _attach_intelligence(cfg: ChallengeConfig, engine: DarwinEngine, driver: LiveDriver) -> None:
    service = build_intelligence(
        cfg, deliver=lambda ts, sig: driver.schedule(ts, sig, engine), clock_ms=driver.clock.now_ms
    )
    if service is not None:
        engine.observers.append(service.on_bar)
    if cfg.persistence.record_market_data:
        driver.recorder = ParquetRecorder(cfg.persistence.parquet_dir, engine.run_id)


def _webhooks(cfg: ChallengeConfig, driver: LiveDriver) -> dict[str, WebhookSignalFeed]:
    return webhook_feeds_from_env(
        cfg.intelligence.external_webhooks, driver.push, driver.clock.now_ms, cfg.challenge.symbols
    )


def build_runtime(
    cfg: ChallengeConfig,
    store: AuditStore | None,
    run_id: str,
    events: Iterable[Event] | None = None,
    start_ts: int | None = None,
    seed_genomes: list[tuple[Genome, str]] | None = None,
) -> Runtime:
    mode = cfg.challenge.mode
    assert_live_allowed(cfg, mode)

    if mode is Mode.REPLAY:
        t0 = start_ts if start_ts is not None else 1_700_000_000_000
        h = build_replay(
            cfg,
            events if events is not None else synthetic_stream(cfg, t0),
            t0,
            store=store,
            run_id=run_id,
            intelligence=cfg.intelligence.mock.enabled,
            seed_genomes=seed_genomes,
        )

        async def run_replay() -> float:
            return await asyncio.to_thread(h.driver.run)

        return Runtime(engine=h.engine, run=run_replay, info={"mode": mode.value})

    if mode is Mode.SIM:
        if cfg.sim.speed > 600:
            log.warning(
                "sim speed %.0fx: event re-stamping degrades simulated venue timing above ~600x",
                cfg.sim.speed,
            )
        t0 = start_ts if start_ts is not None else (int(time.time() * 1000) // 60_000) * 60_000
        engine = DarwinEngine(cfg, run_id=run_id, start_ts=t0, store=store)
        clock = Clock(speed=cfg.sim.speed, origin_ms=t0)
        driver = LiveDriver(engine, [], clock)
        driver.venues = _sim_venues(cfg, engine, driver, challenge_sim=True)
        _attach_intelligence(cfg, engine, driver)
        engine.start(seed_genomes)
        feed = SyntheticFeed(driver, events if events is not None else synthetic_stream(cfg, t0))

        async def run_sim() -> float:
            task = asyncio.create_task(feed.run())
            try:
                return await driver.run()
            finally:
                task.cancel()

        return Runtime(
            engine=engine,
            run=run_sim,
            info={"mode": mode.value, "speed": cfg.sim.speed},
            webhooks=_webhooks(cfg, driver),
        )

    # ---- modes backed by Bybit streams
    endpoints = TESTNET if mode is Mode.TESTNET else MAINNET
    t0 = start_ts if start_ts is not None else int(time.time() * 1000)
    engine = DarwinEngine(cfg, run_id=run_id, start_ts=t0, store=store)
    clock = Clock()
    driver = LiveDriver(engine, [], clock)
    market_feed = BybitMarketFeed(
        driver, cfg.challenge.symbols, endpoints.public_linear_ws, depth=cfg.exchange.orderbook_depth
    )
    engine.on_resync_needed = market_feed.resync
    cleanup: list[Callable[[], Awaitable[None]]] = []
    private_feed: BybitPrivateFeed | None = None
    preflight: Callable[[], Awaitable[PreflightReport]] | None = None
    if mode is Mode.PAPER:
        driver.venues = _sim_venues(cfg, engine, driver, challenge_sim=True)
    else:
        prefix = "BYBIT_TESTNET" if mode is Mode.TESTNET else "BYBIT"
        creds = Credentials.from_env(prefix)
        if creds is None:
            raise RuntimeError(f"{mode.value} mode requires {prefix}_API_KEY and {prefix}_API_SECRET")
        rest = BybitRest(endpoints.rest, creds, recv_window_ms=cfg.exchange.recv_window_ms)
        cleanup.append(rest.close)
        instruments = {s: cfg.instrument(s) for s in cfg.challenge.symbols}
        gateway = BybitExecutionGateway(
            CHALLENGE_VENUE, rest, CHALLENGE, driver.push, instruments, clock_ms=clock.now_ms
        )
        driver.venues = _sim_venues(cfg, engine, driver, challenge_sim=False)
        engine.attach_gateway(CHALLENGE_VENUE, gateway)
        private_feed = BybitPrivateFeed(driver, gateway, endpoints.private_ws, creds, CHALLENGE)

        async def preflight() -> PreflightReport:
            return await account_preflight(rest, cfg, instruments, strict=mode is Mode.LIVE)

    _attach_intelligence(cfg, engine, driver)
    engine.start(seed_genomes)

    async def run_streams() -> float:
        if preflight is not None:
            rep = await preflight()
            for w in rep.warnings:
                log.warning("preflight: %s", w)
            log.info(
                "preflight ok: equity=%.2f margin=%s leverage=%s", rep.equity, rep.margin_mode, rep.leverage
            )
        tasks = [asyncio.create_task(market_feed.run())]
        if private_feed is not None:
            tasks.append(asyncio.create_task(private_feed.run()))
        try:
            return await driver.run()
        finally:
            for t in tasks:
                t.cancel()
            for c in cleanup:
                await c()

    return Runtime(
        engine=engine,
        run=run_streams,
        cleanup=cleanup,
        info={"mode": mode.value, "endpoints": endpoints},
        webhooks=_webhooks(cfg, driver),
    )


async def prepare_config(cfg: ChallengeConfig) -> ChallengeConfig:
    """Before the engine exists (instrument specs are frozen into the governor), pull the real
    lot/tick/leverage filters from Bybit. Best effort for paper; mandatory for testnet/live."""
    mode = cfg.challenge.mode
    if mode not in (Mode.PAPER, Mode.TESTNET, Mode.LIVE):
        return cfg
    try:
        return await refresh_instruments(cfg, testnet=mode is Mode.TESTNET)
    except Exception as e:
        if mode is Mode.PAPER:
            log.warning("instrument refresh failed (%s); using fallback specs for paper trading", e)
            return cfg
        raise


async def refresh_instruments(cfg: ChallengeConfig, testnet: bool = False) -> ChallengeConfig:
    """Pull lot/tick filters from Bybit so sizing matches the exchange (public endpoint)."""
    rest = BybitRest((TESTNET if testnet else MAINNET).rest, None)
    try:
        specs = await rest.instruments(list(cfg.challenge.symbols))
    finally:
        await rest.close()
    exch = cfg.exchange.model_copy(update={"instruments": tuple(specs.values())})
    return cfg.model_copy(update={"exchange": exch})
