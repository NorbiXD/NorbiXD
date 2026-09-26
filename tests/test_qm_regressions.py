"""Regression tests for the Quality Manager's iteration-1 findings (one section per finding)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from darwin.attribution.explain import AmbiguousIntent, explain
from darwin.config.challenge import InstrumentSpec, RiskLimits
from darwin.core.events import (
    BookDelta,
    FillEvent,
    FundingPayment,
    OrderUpdate,
    TimerEvent,
    WalletSnapshot,
)
from darwin.core.intent import TradeIntent
from darwin.core.types import AgentStatus, OrderStatus, Side
from darwin.evolution.fitness import TradeSample
from darwin.exchange.bybit.preflight import PreflightError, account_preflight
from darwin.exchange.bybit.rest import BybitRest
from darwin.exchange.bybit.signing import Credentials
from darwin.exchange.sim.venue import Chaos
from darwin.market.state import MarketState
from darwin.market.synthetic import SyntheticMarket
from darwin.persistence.store import AuditStore
from darwin.portfolio.ledger import Ledger
from darwin.risk.governor import Reservations, RiskGovernor, VenueHealth, liquidation_distance
from darwin.runtime.engine import CHALLENGE, CHALLENGE_VENUE
from darwin.runtime.replay import build_replay
from tests.conftest import T0, book, make_config, ticker, trade
from tests.test_execution import Harness
from tests.test_population import BAR, feed, make_pop

SPECS = {
    "BTCUSDT": InstrumentSpec(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001, min_notional=5),
    "ETHUSDT": InstrumentSpec(symbol="ETHUSDT", tick_size=0.01, qty_step=0.01, min_qty=0.01, min_notional=5),
    "SOLUSDT": InstrumentSpec(symbol="SOLUSDT", tick_size=0.01, qty_step=0.1, min_qty=0.1, min_notional=5),
}
PRICES = {"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0, "SOLUSDT": 150.0}


def _market(now: int = T0) -> MarketState:
    ms = MarketState(list(SPECS))
    for s, p in PRICES.items():
        ms.apply(book(s, now, p, spread=p * 1e-4, qty=1e6, step=p * 1e-4))
        ms.apply(ticker(s, now, p))
        ms.apply(trade(s, now, p, tid=f"t-{s}"))
    return ms


def _intent(agent: str, sym: str, target: float, iid: str, stop: float = 0.02) -> TradeIntent:
    return TradeIntent(
        intent_id=iid,
        ts=T0,
        agent_id=agent,
        genome_id="G",
        symbol=sym,
        target_exposure=target,
        confidence=0.5,
        reason="entry",
        stop_loss_pct=stop,
    )


# ----------------------------------------------------------------------------- C1: reservations


def test_c1_same_bar_agents_cannot_exceed_symbol_limit_together() -> None:
    """QM probe: 5 funded agents at weight 0.18 each want SOL at 5x on the same bar."""
    lim = RiskLimits(kill_switch_file=None)  # defaults: symbol 3x, gross 5x
    gov = RiskGovernor(lim, SPECS)
    led = Ledger()
    acct = led.open_account("challenge", "challenge", 200.0)
    ms = _market()
    reserved: dict[str, float] = {}
    new_pos = 0
    total = 0.0
    for i in range(5):
        d = gov.evaluate(
            _intent(f"A{i}", "SOLUSDT", 5.0, f"I{i}"),
            acct,
            0.18 * 200,
            ms["SOLUSDT"],
            PRICES,
            0.0,
            T0 + 10,
            VenueHealth(),
            Reservations(dict(reserved), new_pos),
        )
        if d.approved:
            reserved["SOLUSDT"] = reserved.get("SOLUSDT", 0.0) + abs(d.order_qty)
            new_pos += 1
            total += abs(d.order_qty) * 150.0
    assert total <= lim.max_symbol_leverage * 200 + 1e-6
    assert total > 0


@pytest.mark.parametrize("seed", range(40))
def test_c1_property_envelope_holds_after_all_in_flight_orders_fill(seed: int) -> None:
    """Random agents/targets/existing positions/flips; every approved order then fills at the
    reference price. Post-fill exposure must be inside the envelope."""
    rng = np.random.default_rng(seed)
    lim = RiskLimits(
        kill_switch_file=None,
        max_agent_leverage=20,
        max_gross_leverage=5,
        max_symbol_leverage=3,
        max_concurrent_positions=int(rng.integers(3, 12)),
        dedup_window_ms=0,
    )
    gov = RiskGovernor(lim, SPECS)
    led = Ledger()
    equity = float(rng.uniform(150, 5_000))
    acct = led.open_account("challenge", "challenge", equity)
    ms = _market()
    syms = list(SPECS)
    # pre-existing (already filled) positions inside the limits
    for k in range(int(rng.integers(0, 3))):
        s = syms[int(rng.integers(0, 3))]
        q = SPECS[s].qty_step * int(rng.integers(1, 5))
        if acct.gross_notional(PRICES) + q * PRICES[s] < 0.5 * equity:
            acct.pos(f"P{k}", s).apply_fill(Side.BUY if rng.random() < 0.5 else Side.SELL, q, PRICES[s])
    reserved: dict[str, float] = {}
    new_pos: set[tuple[str, str]] = set()
    approved: list[tuple[str, str, float]] = []
    agents = [f"A{i}" for i in range(int(rng.integers(2, 12)))] + [f"P{k}" for k in range(3)]
    for n, agent in enumerate(agents):
        s = syms[int(rng.integers(0, 3))]
        target = float(rng.uniform(-10, 10))
        cap = float(rng.uniform(0.05, 1.0)) * equity
        d = gov.evaluate(
            _intent(agent, s, target, f"I{n}"),
            acct,
            cap,
            ms[s],
            PRICES,
            0.0,
            T0 + 10 + n,
            VenueHealth(),
            Reservations(dict(reserved), len(new_pos)),
        )
        if not d.approved:
            continue
        approved.append((agent, s, d.order_qty))
        if d.risk_increasing:
            reserved[s] = reserved.get(s, 0.0) + abs(d.order_qty)
            if acct.agent_qty(agent, s) == 0:
                new_pos.add((agent, s))
    for agent, s, q in approved:  # everything in flight now fills
        acct.pos(agent, s).apply_fill(Side.BUY if q > 0 else Side.SELL, abs(q), PRICES[s])
    eq = acct.equity(PRICES)
    gross = acct.gross_notional(PRICES)
    tol = 1e-6 * equity + 1e-9
    assert gross <= lim.max_gross_leverage * eq + tol
    for s in syms:
        assert acct.symbol_gross_notional(s, PRICES[s]) <= lim.max_symbol_leverage * eq + tol
    if gross > 0:
        assert (
            liquidation_distance(eq, gross, lim.maintenance_margin_rate)
            >= lim.min_liquidation_distance_pct - 1e-9
        )
    assert len(acct.open_positions()) <= max(lim.max_concurrent_positions, 3)


def test_c1_execution_reservations_include_other_agents_in_flight_orders() -> None:
    h = Harness()
    h.submit(0.5, did="RD1")
    res = h.exe.reservations("challenge")
    assert res.qty_by_symbol == {"BTCUSDT": pytest.approx(0.5)} and res.new_positions == 1
    h.sched.drain()  # filled
    assert h.exe.reservations("challenge").qty_by_symbol == {}


# ----------------------------------------------------------------------------- M2: fills after status


def test_m2_filled_update_before_executions_keeps_exposure_pending() -> None:
    h = Harness()
    mo = h.submit(0.5)
    h.exe.on_order_update(
        OrderUpdate(
            ts=T0 + 60,
            account="challenge",
            client_order_id=mo.client_order_id,
            symbol="BTCUSDT",
            status=OrderStatus.FILLED,
            cum_qty=0.5,
        )
    )
    assert mo.status is OrderStatus.FILLED and mo.open  # terminal, but fills are still owed
    assert h.exe.pending_qty("challenge", "A1", "BTCUSDT") == pytest.approx(0.5)
    assert h.exe.has_open("challenge", "BTCUSDT")
    assert h.exe.reservations("challenge").qty_by_symbol["BTCUSDT"] == pytest.approx(0.5)
    h.exe.on_fill(
        FillEvent(
            ts=T0 + 70,
            account="challenge",
            client_order_id=mo.client_order_id,
            exec_id="e1",
            symbol="BTCUSDT",
            side=Side.BUY,
            qty=0.5,
            price=100.5,
            fee=0.03,
        )
    )
    assert not mo.open and h.exe.pending_qty("challenge", "A1", "BTCUSDT") == 0.0
    assert h.ledger["challenge"].agent_qty("A1", "BTCUSDT") == pytest.approx(0.5)


def test_m2_missing_executions_are_queried_then_reconciled() -> None:
    h = Harness(ack_timeout_ms=1_000)
    queries: list[str] = []

    class Gw:
        venue = "challenge"

        def submit(self, order: Any) -> None: ...
        def cancel(self, *a: Any) -> None: ...
        def query_order(self, account: str, cid: str, symbol: str, ts: int) -> None:
            queries.append(cid)

        def query_positions(self, *a: Any) -> None: ...

        def query_open_orders(self, *a: Any) -> None: ...

    h.exe.gateways["challenge"] = Gw()  # type: ignore[assignment]
    mo = h.submit(0.5)
    h.exe.on_order_update(
        OrderUpdate(
            ts=T0 + 60,
            account="challenge",
            client_order_id=mo.client_order_id,
            symbol="BTCUSDT",
            status=OrderStatus.FILLED,
            cum_qty=0.5,
        )
    )
    for i in range(1, 7):
        h.exe.check_timeouts(T0 + 60 + i * 1_000)
    assert len(queries) == 3
    assert not mo.open  # gave up waiting...
    assert not h.health["challenge"].reconcile_ok  # ...and position reconciliation must arbitrate


# ----------------------------------------------------------------------------- C2: persistence


def _small_replay(store: AuditStore, run_id: str) -> Any:
    cfg = make_config(
        challenge={"duration_hours": 3}, evolution={"population_size": 8, "generation_bars": 40}
    )
    m = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=T0,
        duration_ms=cfg.duration_ms + 60_000,
        step_ms=5_000,
        seed=9,
    )
    h = build_replay(cfg, m.events(), T0, store=store, run_id=run_id)
    h.driver.run()
    store.flush()
    return h


def test_c2_two_runs_share_one_database(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'shared.db'}"
    a = _small_replay(AuditStore(url, "run-a"), "run-a")
    b = _small_replay(AuditStore(url, "run-b"), "run-b")
    store = AuditStore(url, "inspect")
    ia = store.query("intents", run_id="run-a")
    ib = store.query("intents", run_id="run-b")
    assert ia and len(ia) == len(ib)  # identical config/data => identical decisions, both persisted
    assert a.engine.stats["intents"] == len(ia) and b.engine.stats["intents"] == len(ib)
    oa = {o["client_order_id"] for o in store.query("orders", run_id="run-a")}
    ob = {o["client_order_id"] for o in store.query("orders", run_id="run-b")}
    assert oa and not (oa & ob)  # exchange-facing ids never collide across runs
    iid = ia[0]["intent_id"]
    with pytest.raises(AmbiguousIntent):
        explain(store, iid)
    ex = explain(store, iid, run_id="run-b")
    assert ex is not None and ex["intent"]["run_id"] == "run-b"


def test_c2_failed_flush_requeues_and_halts_new_risk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = AuditStore(f"sqlite:///{tmp_path / 'f.db'}", "r")
    store.add("system_events", {"ts": 1, "kind": "x", "detail": {}})
    store.upsert(
        "agents",
        {
            "agent_id": "A1",
            "genome_id": "G",
            "species": "s",
            "generation": 0,
            "parent_ids": [],
            "status": "alive",
            "strikes": 0,
            "born_ts": 0,
            "died_ts": None,
            "death_reason": None,
        },
    )
    real_begin = store.engine.begin
    monkeypatch.setattr(store.engine, "begin", lambda: (_ for _ in ()).throw(RuntimeError("disk gone")))
    with pytest.raises(RuntimeError):
        store.flush()
    assert store.pending() == 2 and store.failures == 1  # nothing lost
    monkeypatch.setattr(store.engine, "begin", real_begin)
    assert store.flush() == 2 and store.pending() == 0
    # engine side: an audit failure blocks new challenge risk until the store recovers
    h = build_replay(make_config(), iter([]), T0)
    h.engine.audit_flush_failed("disk gone")
    ok, why = h.engine.health[CHALLENGE_VENUE].healthy(5)
    assert not ok and why == "audit_store_unavailable"
    h.engine.audit_flush_ok()
    assert h.engine.health[CHALLENGE_VENUE].healthy(5)[0]


# ----------------------------------------------------------------------------- M5: stale marks


def test_m5_stale_mark_is_not_used_for_sizing() -> None:
    ms = MarketState(["SOLUSDT"], max_price_age_ms=5_000)
    ms.apply(ticker("SOLUSDT", T0, 150.0))  # mark goes quiet at 150
    later = T0 + 30 * 60_000
    ms.apply(book("SOLUSDT", later, 165.0, spread=0.02, qty=1e4, step=0.01))
    ms.apply(trade("SOLUSDT", later, 165.0, tid="x"))
    st = ms["SOLUSDT"]
    assert st.ref_price(later) == pytest.approx(165.0)  # fresh mid, not the 30-minute-old mark
    gov = RiskGovernor(RiskLimits(kill_switch_file=None), SPECS)
    led = Ledger()
    acct = led.open_account("challenge", "challenge", 1_000.0)
    d = gov.evaluate(
        _intent("A", "SOLUSDT", 1.0, "I"), acct, 400.0, st, {"SOLUSDT": 165.0}, 0.0, later + 1, VenueHealth()
    )
    assert d.approved and abs(d.order_qty) * 165.0 <= 400.0 + 1e-6
    # nothing fresh at all => stale for new risk
    assert st.ref_price(later + 60_000) is None and st.is_stale(later + 60_000, 5_000)


# ----------------------------------------------------------------------------- M1/M3: evolution


def test_m1_small_number_of_free_slots_still_produces_offspring() -> None:
    pop = make_pop()
    pop.seed(0)
    ids = [a.agent_id for a in pop.alive]
    ts = feed(pop, 0, 100, {ids[0]: 0.0008, ids[1]: 0.0006}, seed=1)
    pop.evolve(ts, {a: 1000.0 for a in ids})
    ts = feed(pop, ts, 100, {ids[0]: 0.0008, ids[1]: 0.0006}, seed=2)
    eq = {a.agent_id: 1000.0 for a in pop.alive}
    for victim in ids[5:8]:  # three deaths by ruin (plus any strikes) => a few free slots
        eq[victim] = 100.0
    res = pop.evolve(ts, eq)
    children = [a for a in res.born if a.parent_ids]
    immigrants = len(res.born) - len(children)
    # before the fix, 4 immigrant slots were reserved first: <= 4 free slots meant zero offspring
    assert len(res.born) >= 3 and len(children) >= 2 and immigrants <= max(1, round(0.15 * len(res.born)))


def test_m3_selection_is_relative_to_the_cohort_on_identical_bars() -> None:
    pop = make_pop(immigrant_rate=0.1)
    pop.seed(0)
    old = [a.agent_id for a in pop.alive]
    # generation 1: a bad period for everyone
    ts = feed(pop, 0, 100, {a: -0.0005 for a in old}, seed=3)
    for a in pop.alive:
        a.born_ts = 0
    young = pop.inject(pop.alive[0].genome.model_copy(update={"entry_threshold": 0.9}), ts, origin="test")
    pop.books[young.agent_id].equity.append(1000.0)
    pop.books[young.agent_id].ts.append(ts)
    ts2 = ts
    for _ in range(100):  # generation 2: the young agent and the veterans on identical bars
        ts2 += BAR
        eq = {a: pop.books[a].equity[-1] * (1 + 0.0001) for a in old}
        eq[young.agent_id] = pop.books[young.agent_id].equity[-1] * (1 + 0.0001)
        pop.record_bar(ts2, eq, "range")
        if _ % 10 == 0:
            for a in [*old, young.agent_id]:
                pop.record_trade(a, TradeSample(0.001, 0.5, 1, ts2 - BAR, ts2))
    evals = pop.evaluate(ts2)
    ey = evals[young.agent_id]
    # the young agent's benchmark is the cohort on *its* window, not the veterans' bad history
    assert ey.cohort_median == pytest.approx(
        float(
            np.median(
                [
                    pop._window_fitness(a, pop.books[young.agent_id].ts[0], ts2, "x").fitness
                    for a in [*old, young.agent_id]
                ]
            )
        ),
        abs=0.5,
    )
    assert abs(ey.relative_fitness) < 1.0  # it did exactly what the cohort did on the same bars


def test_m3_champion_can_be_demoted() -> None:
    pop = make_pop(kill_fraction=0.5, champion_t_stat=0.0)
    pop.seed(0)
    ids = [a.agent_id for a in pop.alive]
    ts = feed(pop, 0, 100, {ids[0]: 0.001}, seed=4)
    pop.evolve(ts, {a: 1000.0 for a in ids})
    ts = feed(pop, ts, 100, {ids[0]: 0.001}, seed=5)
    pop.evolve(ts, {a.agent_id: 1000.0 for a in pop.alive})
    assert pop.champion_id == ids[0]
    for g in range(3):  # the champion degrades badly while everyone else is flat
        ts = feed(pop, ts, 100, {ids[0]: -0.003}, seed=6 + g)
        pop.evolve(ts, {a.agent_id: 1000.0 for a in pop.alive})
    assert pop.agents[ids[0]].status in (AgentStatus.PROBATION, AgentStatus.DEAD)
    assert pop.champion_id != ids[0]


# ----------------------------------------------------------------------------- M4: testnet readiness


def _preflight_rest(
    positions: list[dict[str, Any]],
    margin: str = "REGULAR_MARGIN",
    open_orders: list[dict[str, Any]] | None = None,
) -> tuple[BybitRest, list[str]]:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        body: dict[str, Any] = {"retCode": 0, "retMsg": "OK", "result": {}}
        if req.url.path == "/v5/account/wallet-balance":
            body["result"] = {"list": [{"totalEquity": "5000", "totalWalletBalance": "5000"}]}
        elif req.url.path == "/v5/position/list":
            body["result"] = {"list": positions}
        elif req.url.path == "/v5/account/info":
            body["result"] = {"marginMode": margin}
        elif req.url.path == "/v5/order/realtime":
            body["result"] = {"list": open_orders or []}
        elif req.url.path == "/v5/position/switch-mode":
            body = {"retCode": 110025, "retMsg": "Position mode is not modified", "result": {}}
        return httpx.Response(200, json=body)

    return BybitRest(
        "https://api-testnet.bybit.com", Credentials("k", "s"), transport=httpx.MockTransport(handler)
    ), calls


async def test_m4_preflight_sets_mode_and_leverage_on_flat_account() -> None:
    cfg = make_config(challenge={"symbols": ["BTCUSDT", "ETHUSDT"]})
    rest, calls = _preflight_rest([{"symbol": "BTCUSDT", "side": "", "size": "0"}])
    rep = await account_preflight(
        rest, cfg, {s: cfg.instrument(s) for s in cfg.challenge.symbols}, strict=True
    )
    assert rep.equity == 5000 and rep.margin_mode == "REGULAR_MARGIN"
    assert "/v5/position/switch-mode" in calls and calls.count("/v5/position/set-leverage") == 2
    assert rep.leverage == {"BTCUSDT": 10.0, "ETHUSDT": 10.0}  # 2 x ceil(max_gross_leverage=5)


async def test_m4_preflight_refuses_non_flat_or_isolated_accounts() -> None:
    cfg = make_config(challenge={"symbols": ["BTCUSDT"]})
    inst = {"BTCUSDT": cfg.instrument("BTCUSDT")}
    rest, _ = _preflight_rest([{"symbol": "BTCUSDT", "side": "Buy", "size": "0.01"}])
    with pytest.raises(PreflightError, match="not flat"):
        await account_preflight(rest, cfg, inst, strict=False)  # never adopt unknown positions
    rest2, _ = _preflight_rest([], margin="ISOLATED_MARGIN")
    with pytest.raises(PreflightError, match="margin mode"):
        await account_preflight(rest2, cfg, inst, strict=True)
    rest3, _ = _preflight_rest([], margin="ISOLATED_MARGIN")
    rep = await account_preflight(rest3, cfg, inst, strict=False)  # testnet: warning only
    assert rep.warnings
    rest4, _ = _preflight_rest([], open_orders=[{"symbol": "BTCUSDT", "orderLinkId": "manual-1"}])
    with pytest.raises(PreflightError, match="open orders"):  # could fill into an unknown position
        await account_preflight(rest4, cfg, inst, strict=False)


def test_m4_wallet_pnl_drift_blocks_new_risk_until_reconciled() -> None:
    h = build_replay(make_config(), iter([]), T0)
    eng = h.engine
    eng.handle(WalletSnapshot(ts=T0 + 1, account=CHALLENGE, equity=10_000.0, wallet_balance=10_000.0))
    eng.handle(
        WalletSnapshot(ts=T0 + 2, account=CHALLENGE, equity=9_990.0, wallet_balance=9_990.0)
    )  # -10 unexplained
    ok, why = eng.health[CHALLENGE_VENUE].healthy(5)
    assert not ok and "wallet pnl drift" in why
    eng.handle(WalletSnapshot(ts=T0 + 3, account=CHALLENGE, equity=10_000.0, wallet_balance=10_000.0))
    assert eng.health[CHALLENGE_VENUE].healthy(5)[0]


# ----------------------------------------------------------------------------- minor findings


def test_resync_requests_are_deduplicated() -> None:
    cfg = make_config(challenge={"symbols": ["BTCUSDT"]})
    h = build_replay(cfg, iter([]), T0)
    requests: list[str] = []
    h.engine.on_resync_needed = requests.append
    h.engine.handle(book("BTCUSDT", T0 + 1, 100.0, update_id=10))
    for k in range(20):  # a gap, then a stream of deltas we can't apply
        h.engine.handle(
            BookDelta(ts=T0 + 10 + k, symbol="BTCUSDT", bids=((99.0, 1.0),), asks=(), update_id=12 + k)
        )
    assert requests == ["BTCUSDT"]
    h.engine.handle(book("BTCUSDT", T0 + 100, 100.0, update_id=1))
    h.engine.handle(BookDelta(ts=T0 + 200, symbol="BTCUSDT", bids=((99.0, 1.0),), asks=(), update_id=5))
    assert requests == ["BTCUSDT", "BTCUSDT"]


def test_funding_uses_venue_amount_and_reports_residual() -> None:
    h = build_replay(make_config(challenge={"symbols": ["BTCUSDT"]}), iter([]), T0)
    acct = h.engine.ledger[CHALLENGE]
    acct.pos("A1", "BTCUSDT").apply_fill(Side.BUY, 1.0, 100.0)
    cash0 = acct.cash
    h.engine.handle(
        FundingPayment(
            ts=T0 + 1,
            account=CHALLENGE,
            symbol="BTCUSDT",
            position_qty=1.1,
            rate=0.001,
            mark_price=100.0,
            amount=0.11,
        )
    )
    assert acct.cash == pytest.approx(cash0 - 0.11)  # the venue's number, not ours
    assert h.engine.stats["sys:funding_unattributed"] == 1


def test_kill_switch_acts_on_heartbeat_not_only_bar_close() -> None:
    cfg = make_config(challenge={"symbols": ["BTCUSDT"]})
    h = build_replay(cfg, iter([]), T0)
    eng = h.engine
    eng.handle(book("BTCUSDT", T0 + 1, 100.0))
    eng.handle(ticker("BTCUSDT", T0 + 1, 100.0))
    eng.marks = {"BTCUSDT": 100.0}
    eng.ledger[CHALLENGE].pos("A0001", "BTCUSDT").apply_fill(Side.BUY, 0.1, 100.0)
    eng.governor.engage_kill_switch()
    before = eng.stats["intents"]
    eng.handle(TimerEvent(ts=T0 + 1_500, name="heartbeat"))  # mid-bar
    assert eng.stats["intents"] == before + 1 and eng.stats["sys:kill_switch"] == 1


def test_chaos_end_to_end_accounting_invariants() -> None:
    cfg = make_config(
        challenge={"duration_hours": 6},
        evolution={"population_size": 10, "generation_bars": 60, "min_trades": 2},
        allocator={"rebalance_bars": 15, "exploration_budget": 0.3, "max_funded_agents": 6},
    )
    m = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=T0,
        duration_ms=cfg.duration_ms + 60_000,
        step_ms=5_000,
        seed=13,
    )
    chaos = Chaos(reject_prob=0.08, drop_ack_prob=0.08, duplicate_fill_prob=0.1, late_ack_ms=300)
    h = build_replay(cfg, m.events(), T0, chaos=chaos, shadow_chaos=chaos)
    h.driver.run()
    eng = h.engine
    assert eng.execution.duplicate_fills > 3 and eng.stats["orders"] > 50
    assert not [o for o in eng.execution.open_orders()], "no stuck orders"
    for acct_id, acct in eng.ledger.accounts.items():
        venue = h.shadow if acct_id.startswith("shadow:") else h.challenge
        for s in cfg.challenge.symbols:
            assert acct.net_qty(s) == pytest.approx(venue.position(acct_id, s), abs=1e-9)
        assert acct.cash == pytest.approx(venue.accounts[acct_id].cash, abs=1e-6)


def test_book_deltas_and_gaps_through_the_full_pipeline() -> None:
    cfg = make_config(
        challenge={"duration_hours": 6},
        evolution={"population_size": 10, "generation_bars": 60, "min_trades": 2},
    )
    m = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=T0,
        duration_ms=cfg.duration_ms + 60_000,
        step_ms=2_000,
        seed=14,
        book_deltas=True,
        snapshot_every=15,
        gap_prob=0.03,
    )
    h = build_replay(cfg, m.events(), T0)
    h.driver.run()
    eng = h.engine
    assert eng.stats["book_invalid"] > 0  # gaps were detected...
    assert eng.stats["orders"] > 20  # ...and the books recovered on the next snapshot
    stale_bars = sum(b.stale for hist in eng.features.history.values() for b in hist)
    assert stale_bars > 0  # agents never decide on a bar whose book was invalid at close
    for acct_id, acct in eng.ledger.accounts.items():
        venue = h.shadow if acct_id.startswith("shadow:") else h.challenge
        for s in cfg.challenge.symbols:
            assert acct.net_qty(s) == pytest.approx(venue.position(acct_id, s), abs=1e-9)


def test_api_json_is_valid_for_state(tmp_path: Path) -> None:
    h = build_replay(make_config(), iter([]), T0)
    json.dumps(h.engine.state())
