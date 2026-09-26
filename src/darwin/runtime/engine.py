"""DarwinEngine: the single, deterministic event processor shared by replay, paper and live.

    market events ──► MarketState ──► bars ──► FeatureEngine ──► FeatureView (identical for all)
                                                         │
                  ┌──────────────── every alive Agent ◄──┘
                  ▼
             TradeIntent ──► RiskGovernor(shadow book)    ──► ExecutionEngine ──► shadow SimExchange
                         └─► RiskGovernor(challenge book) ──► ExecutionEngine ──► challenge venue
                                                                           ▲
     fills / order updates / funding ──► ExecutionEngine ──► Ledger ───────┘
                                                   │
     every bar: mark-to-market, guards, breakers   ▼
     every generation: Population.evolve (fitness → death → reproduction) ─► Allocator ─► weights

The engine never touches the network or the wall clock. Time is the ``ts`` of the event being
processed; I/O lives in drivers and gateways. That is what makes a replay reproduce a live
session decision-for-decision.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter, deque
from collections.abc import Callable
from typing import Any

import numpy as np

from darwin.agents.agent import Agent
from darwin.agents.genome import Genome
from darwin.config.challenge import ChallengeConfig
from darwin.core.events import (
    BookDelta,
    BookSnapshot,
    Event,
    FeedStatus,
    FillEvent,
    FundingPayment,
    FundingSettlement,
    IntelligenceSignal,
    LiquidationEvent,
    OrderUpdate,
    PositionSnapshot,
    TickerEvent,
    TimerEvent,
    TradeEvent,
    WalletSnapshot,
)
from darwin.core.ids import Sequence, content_hash
from darwin.core.intent import IntentReason, TradeIntent
from darwin.core.types import AgentStatus, LineageEventKind
from darwin.evolution.allocator import AllocationCandidate, make_allocator
from darwin.evolution.fitness import TradeSample
from darwin.evolution.population import EvolutionResult, LineageEvent, Population
from darwin.execution.engine import ExecutionEngine
from darwin.execution.orders import ExecutionGateway, ManagedOrder
from darwin.features.engine import FeatureEngine, FeatureView
from darwin.market.bars import Bar, BarBuilder
from darwin.market.state import MarketState
from darwin.persistence.store import AuditStore
from darwin.portfolio.ledger import Ledger, RoundTrip
from darwin.risk.governor import RiskDecision, RiskGovernor, VenueHealth
from darwin.signals.board import SignalBoard

log = logging.getLogger(__name__)

CHALLENGE = "challenge"
SHADOW_VENUE = "shadow"
CHALLENGE_VENUE = "challenge"

BarObserver = Callable[["DarwinEngine", int, dict[str, FeatureView]], None]


def shadow_account(agent_id: str) -> str:
    return f"shadow:{agent_id}"


def account_venue(account: str) -> str:
    return SHADOW_VENUE if account.startswith("shadow:") else CHALLENGE_VENUE


class DarwinEngine:
    def __init__(
        self,
        cfg: ChallengeConfig,
        run_id: str,
        start_ts: int,
        store: AuditStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.start_ts = start_ts
        self.end_ts = start_ts + cfg.duration_ms
        self.store = store
        ch = cfg.challenge
        self.symbols = tuple(ch.symbols)
        self.bar_ms = ch.bar_ms
        self.now = start_ts
        self.next_bar_ts = (start_ts // self.bar_ms + 1) * self.bar_ms
        self.bar_index = 0
        self.ended = False
        self.finalized = False
        self.final_equity: float | None = None

        self.market = MarketState(self.symbols, max_price_age_ms=cfg.risk.max_data_staleness_ms)
        self.builders = {s: BarBuilder(s) for s in self.symbols}
        max_lb = 800
        self.features = FeatureEngine(self.symbols, max_bars=max_lb)
        self.signals = SignalBoard()
        self.ledger = Ledger()
        self.instruments = {s: cfg.instrument(s) for s in self.symbols}
        self.governor = RiskGovernor(cfg.risk, self.instruments)
        self.health: dict[str, VenueHealth] = {SHADOW_VENUE: VenueHealth(), CHALLENGE_VENUE: VenueHealth()}
        self.gateways: dict[str, ExecutionGateway] = {}
        self.execution = ExecutionEngine(
            ledger=self.ledger,
            market=self.market,
            gateways=self.gateways,
            account_venue=account_venue,
            health=self.health,
            ack_timeout_ms=cfg.exchange.order_ack_timeout_ms,
            run_tag=content_hash(run_id, length=5),
        )
        self.execution.set_tick_sizes({s: spec.tick_size for s, spec in self.instruments.items()})
        horizon_days = cfg.challenge.duration_hours / 24
        self.population = Population(
            cfg.evolution, self.symbols, self.bar_ms, horizon_days, seed=cfg.sim.seed
        )
        self.allocator = make_allocator(cfg.allocator)
        self.alloc_rng = np.random.default_rng(cfg.sim.seed + 99)
        self.weights: dict[str, float] = {}
        self.intent_ids = Sequence("I", width=8)
        self.ledger.open_account(CHALLENGE, "challenge", ch.starting_capital)
        self.last_bars: dict[str, Bar] = {}
        self.marks: dict[str, float] = {}
        self.kill_seen = False
        #: accounts confirmed flat (ledger, venue and orders) while a flatten condition holds
        self.flat_confirmed: set[str] = set()
        self._last_exit_attempt: dict[tuple[str, str, str], int] = {}
        self.venue_positions: dict[str, float] = {}
        self._last_flat_query = start_ts - 5_000
        self.last_reconcile_ts = start_ts
        self._last_prune = start_ts
        self.observers: list[BarObserver] = []
        self.on_resync_needed: Callable[[str], None] | None = None
        self._resync_requested: dict[str, int] = {}
        self.venue_equity_base: float | None = None
        self.ledger_equity_base: float | None = None
        # observability
        self.stats: Counter[str] = Counter()
        self.reject_reasons: Counter[str] = Counter()
        self.recent_decisions: deque[dict[str, Any]] = deque(maxlen=300)
        self.equity_curve: deque[tuple[int, float]] = deque(maxlen=20_000)
        self.recent_lineage: deque[dict[str, Any]] = deque(maxlen=300)
        self.recent_signals: deque[dict[str, Any]] = deque(maxlen=200)
        self.venue_equity: float | None = None
        self.last_evolution: EvolutionResult | None = None
        #: held by drivers while mutating and by API readers; uncontended in single-loop modes
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ wiring
    def attach_gateway(self, venue: str, gateway: ExecutionGateway) -> None:
        self.gateways[venue] = gateway

    def start(self, seed_genomes: list[tuple[Genome, str]] | None = None, fill: bool = True) -> None:
        """Seed the population and open accounts. Call once, after gateways are attached."""
        if SHADOW_VENUE not in self.gateways or CHALLENGE_VENUE not in self.gateways:
            raise RuntimeError("attach shadow and challenge gateways before start()")
        chal = self.gateways[CHALLENGE_VENUE]
        if hasattr(chal, "open_account"):
            chal.open_account(CHALLENGE, self.cfg.challenge.starting_capital)
        if self.store is not None:
            self.store.upsert(
                "runs",
                {
                    "run_id": self.run_id,
                    "name": self.cfg.challenge.name,
                    "mode": self.cfg.challenge.mode.value,
                    "config": self.cfg.model_dump(mode="json"),
                    "config_fingerprint": self.cfg.fingerprint(),
                    "risk_fingerprint": self.governor.fingerprint,
                    "starting_capital": self.cfg.challenge.starting_capital,
                    "started_ts": self.start_ts,
                    "ends_ts": self.end_ts,
                    "ended_ts": None,
                    "final_equity": None,
                    "status": "running",
                },
            )
        for agent in self.population.seed(self.start_ts, seed_genomes, fill):
            self._on_birth(agent)
        self._persist_lineage(self.population.drain_lineage())

    def inject_genome(self, genome: Genome, origin: str) -> Agent:
        """Add an externally proposed genome (e.g. a promoted Level-2 species) as a challenger."""
        agent = self.population.inject(genome, self.now, origin)
        self._on_birth(agent)
        self._persist_lineage(self.population.drain_lineage())
        return agent

    def _on_birth(self, agent: Agent) -> None:
        acct = shadow_account(agent.agent_id)
        eval_cap = self.cfg.evolution.eval_capital
        self.ledger.open_account(acct, "shadow", eval_cap)
        shadow = self.gateways[SHADOW_VENUE]
        if hasattr(shadow, "open_account"):
            shadow.open_account(acct, eval_cap)
        self._persist_agent(agent)
        if self.store is not None:
            g = agent.genome
            self.store.upsert(
                "genomes",
                {
                    "genome_id": g.genome_id,
                    "species": g.species,
                    "genome": g.model_dump(mode="json"),
                    "origin": self.population.genome_origin.get(g.genome_id, "evolution"),
                    "created_ts": self.now,
                },
            )

    # ------------------------------------------------------------------ event entry
    def handle(self, ev: Event) -> None:
        self.stats["events"] += 1
        if ev.ts > self.now:
            self._advance(ev.ts)
            self.now = ev.ts
        if isinstance(
            ev, (TradeEvent, BookSnapshot, BookDelta, TickerEvent, LiquidationEvent, FundingSettlement)
        ):
            self._on_market(ev)
        elif isinstance(ev, FillEvent):
            self._on_fill(ev)
        elif isinstance(ev, OrderUpdate):
            mo = self.execution.on_order_update(ev)
            if mo is not None:
                self._persist_order(mo)
        elif isinstance(ev, FundingPayment):
            residual = self.ledger.on_funding(ev.account, ev.symbol, ev.rate, ev.mark_price, amount=ev.amount)
            self.stats["funding_payments"] += 1
            if abs(residual) > 1e-6:
                self._system_event(
                    "funding_unattributed",
                    {
                        "account": ev.account,
                        "symbol": ev.symbol,
                        "venue_amount": ev.amount,
                        "residual": residual,
                    },
                )
        elif isinstance(ev, PositionSnapshot):
            self._reconcile_position(ev)
        elif isinstance(ev, WalletSnapshot):
            if ev.account == CHALLENGE:
                self._reconcile_wallet(ev)
        elif isinstance(ev, IntelligenceSignal):
            self._on_signal(ev)
        elif isinstance(ev, TimerEvent):
            self._on_timer(ev)
        elif isinstance(ev, FeedStatus):
            self._on_feed_status(ev)

    def audit_flush_failed(self, error: str) -> None:
        """The audit trail is a hard requirement: without it, no new real-money risk."""
        h = self.health[CHALLENGE_VENUE]
        if not h.halted:
            h.halted = "audit_store_unavailable"
            self.stats["audit_flush_failures"] += 1
            log.error("audit store flush failed: %s -> new challenge risk halted", error)

    def audit_flush_ok(self) -> None:
        h = self.health[CHALLENGE_VENUE]
        if h.halted == "audit_store_unavailable":
            h.halted = ""
            log.warning("audit store recovered -> risk halt lifted")

    def _on_signal(self, ev: IntelligenceSignal) -> None:
        self.signals.add(ev)
        self.stats["signals"] += 1
        self.recent_signals.append(
            {
                "signal_id": ev.signal_id,
                "ts": ev.ts,
                "source": ev.source,
                "provider": ev.provider,
                "symbol": ev.symbol,
                "topic": ev.topic,
                "value": ev.value,
                "confidence": ev.confidence,
                "summary": str(ev.payload.get("summary") or ev.payload.get("choice") or "")[:200],
            }
        )
        if self.store is not None:
            self.store.upsert(
                "signals",
                {
                    "signal_id": ev.signal_id,
                    "ts": ev.ts,
                    "observed_ts": ev.observed_ts,
                    "source": ev.source,
                    "provider": ev.provider,
                    "symbol": ev.symbol,
                    "topic": ev.topic,
                    "value": ev.value,
                    "confidence": ev.confidence,
                    "half_life_ms": ev.half_life_ms,
                    "payload": ev.payload,
                },
            )

    def _on_feed_status(self, ev: FeedStatus) -> None:
        self._system_event(f"feed_{ev.status}", {"feed": ev.feed, "symbol": ev.symbol, "detail": ev.detail})
        if ev.feed.startswith("public") and ev.status in ("disconnected", "stale", "gap"):
            # never trade on a book we are no longer receiving updates for; the next snapshot
            # after reconnect/resubscribe revalidates it
            for sym, st in self.market.symbols.items():
                if ev.symbol in (None, sym):
                    st.book.invalidate(f"feed {ev.status}")
        elif ev.feed.startswith("private"):
            self.health[CHALLENGE_VENUE].connected = ev.status in ("connected", "resync")

    def advance_to(self, ts: int) -> None:
        """Close any bars that ended before ``ts`` (used by drivers at end of stream)."""
        if ts > self.now:
            self._advance(ts)
            self.now = ts

    def _advance(self, ts: int) -> None:
        """Close every bar whose boundary is <= ts. Decisions happen *at* the boundary time."""
        while ts >= self.next_bar_ts:
            end = self.next_bar_ts
            self.next_bar_ts += self.bar_ms
            self.now = max(self.now, end)
            self._on_bar_close(end)

    # ------------------------------------------------------------------ market
    def _on_market(
        self, ev: TradeEvent | BookSnapshot | BookDelta | TickerEvent | LiquidationEvent | FundingSettlement
    ) -> None:
        if ev.symbol not in self.market:
            return
        ok = self.market.apply(ev)
        if isinstance(ev, TradeEvent):
            if ok:
                self.builders[ev.symbol].on_trade(ev)
                self._check_guards(ev.symbol, ev.price)
            else:
                self.stats["duplicate_trades"] += 1
        elif isinstance(ev, LiquidationEvent):
            self.builders[ev.symbol].on_liquidation(ev)
        elif isinstance(ev, BookDelta) and not ok:
            self.stats["book_invalid"] += 1
            self._system_event(
                "book_invalid", {"symbol": ev.symbol, "reason": self.market[ev.symbol].book.invalid_reason}
            )
            last = self._resync_requested.get(ev.symbol)
            if self.on_resync_needed is not None and (last is None or ev.ts - last >= 5_000):
                self._resync_requested[ev.symbol] = ev.ts  # one request per gap, re-armed after 5s
                self.stats["resync_requests"] += 1
                self.on_resync_needed(ev.symbol)
        elif isinstance(ev, BookSnapshot) and ok:
            self._resync_requested.pop(ev.symbol, None)
        elif isinstance(ev, TickerEvent) and ev.mark_price is not None:
            self._check_guards(ev.symbol, ev.mark_price)

    # ------------------------------------------------------------------ bars / decisions
    def _on_bar_close(self, end_ts: int) -> None:
        start_ts = end_ts - self.bar_ms
        lim = self.cfg.risk
        bars: dict[str, Bar] = {}
        for sym in self.symbols:
            st = self.market[sym]
            bar = self.builders[sym].close_bar(
                start_ts, end_ts, st, stale=st.is_stale(end_ts, lim.max_data_staleness_ms)
            )
            if bar is not None:
                self.features.add_bar(bar)
                bars[sym] = bar
        if not bars:
            return
        self.bar_index += 1
        self.last_bars.update(bars)
        self.marks = {s: b.mark_price for s, b in self.last_bars.items()}
        highs = {s: b.high for s, b in bars.items()}
        lows = {s: b.low for s, b in bars.items()}
        self.ledger.mark(self.marks, highs, lows, end_ts, start_ts)

        views = {s: self.features.view(s, end_ts, self.signals) for s in self.symbols if s in self.last_bars}
        regime = views[self.symbols[0]].regime().value if self.symbols[0] in views else "unknown"
        snap_refs = self._persist_snapshots(end_ts, bars, views)

        # evidence: every alive agent is marked on every bar (identical windows for all)
        shadow_eq = {
            a.agent_id: self.ledger[shadow_account(a.agent_id)].last_equity for a in self.population.alive
        }
        self.population.record_bar(end_ts, shadow_eq, regime)
        chal = self.ledger[CHALLENGE]
        self.equity_curve.append((end_ts, chal.last_equity))
        if self.store is not None:
            self.store.add(
                "equity",
                {
                    "ts": end_ts,
                    "account_id": CHALLENGE,
                    "equity": chal.last_equity,
                    "cash": chal.cash,
                    "gross_notional": chal.gross_notional(self.marks),
                },
            )
            for aid, eq in shadow_eq.items():
                self.store.add(
                    "equity",
                    {
                        "ts": end_ts,
                        "account_id": shadow_account(aid),
                        "equity": eq,
                        "cash": self.ledger[shadow_account(aid)].cash,
                        "gross_notional": 0.0,
                    },
                )

        if end_ts >= self.end_ts:
            self._end_challenge(end_ts)
            return

        self._safety_check(chal.last_equity)
        self._sync_challenge_book()
        self.ledger.closed_trades.clear()  # consumed via on_fill's return value

        evolved = False
        if self.bar_index % self.cfg.evolution.generation_bars == 0:
            self._evolve(end_ts)
            evolved = True
        if evolved or self.bar_index % self.cfg.allocator.rebalance_bars == 0:
            self._rebalance(end_ts)

        for obs in self.observers:
            obs(self, end_ts, views)

        for agent in sorted(self.population.alive, key=lambda a: a.agent_id):
            sacct = self.ledger[shadow_account(agent.agent_id)]
            positions = {s: sacct.agent_qty(agent.agent_id, s) for s in agent.genome.symbols}
            for intent in agent.decide(views, positions, end_ts, self.bar_ms, self.intent_ids, snap_refs):
                self._route(intent)

    def _route(self, intent: TradeIntent, only: str | None = None) -> None:
        """Send one intent through the governor for the shadow and/or challenge book."""
        self.stats["intents"] += 1
        self._persist_intent(intent)
        outcomes: dict[str, Any] = {}
        accounts: list[str] = []
        aid = intent.agent_id
        if only is None or only == shadow_account(aid):
            accounts.append(shadow_account(aid))
        if only is None or only == CHALLENGE:
            w = self.weights.get(aid, 0.0)
            cur = self.ledger[CHALLENGE].agent_qty(aid, intent.symbol)
            if (w > 0 or cur != 0) and not (self.ended and intent.reason != "challenge_end"):
                accounts.append(CHALLENGE)
        for acct_id in accounts:
            if acct_id not in self.ledger.accounts:
                continue
            acct = self.ledger[acct_id]
            if acct_id == CHALLENGE:
                capital = self.weights.get(aid, 0.0) * max(acct.last_equity, 0.0)
            else:
                capital = max(acct.last_equity, 0.0)
            st = self.market[intent.symbol]
            marks = self._marks()
            pending = self.execution.pending_qty(acct_id, aid, intent.symbol)
            if intent.is_system and intent.target_exposure == 0:
                # getting flat must not wait for a working entry: cancel it and exit what is
                # filled now; anything that fills before the cancel lands is exited next pass
                inc, pending = self.execution.pending_split(acct_id, aid, intent.symbol)
                if inc:
                    self.execution.cancel_open(acct_id, self.now, aid, intent.symbol)
            decision = self.governor.evaluate(
                intent,
                acct,
                capital,
                st,
                marks,
                pending,
                self.now,
                self.health[account_venue(acct_id)],
                self.execution.reservations(acct_id),
            )
            self._persist_decision(decision)
            mo = self.execution.submit(decision, intent, self.now) if decision.approved else None
            if mo is not None:
                self.stats["orders"] += 1
                self._persist_order(mo)
            if decision.approved:
                self.stats["approved"] += 1
            else:
                self.stats["rejected"] += 1
                for r in decision.reasons:
                    self.reject_reasons[r.split("(")[0]] += 1
            outcomes["shadow" if acct_id != CHALLENGE else "challenge"] = {
                "approved": decision.approved,
                "order_qty": decision.order_qty,
                "reasons": list(decision.reasons),
                "order": mo.client_order_id if mo else None,
            }
        self.recent_decisions.append(
            {
                "intent_id": intent.intent_id,
                "ts": intent.ts,
                "agent_id": aid,
                "species": self.population.agents[aid].species if aid in self.population.agents else "?",
                "symbol": intent.symbol,
                "direction": intent.direction,
                "reason": intent.reason,
                "target_exposure": round(intent.target_exposure, 4),
                "confidence": round(intent.confidence, 4),
                "regime": intent.regime,
                "components": intent.components,
                "outcomes": outcomes,
            }
        )

    def _marks(self) -> dict[str, float]:
        out = dict(self.marks)
        for s in self.symbols:
            p = self.market[s].ref_price(self.now)
            if p:
                out[s] = p
        return out

    # ------------------------------------------------------------------ protective guards
    def _check_guards(self, symbol: str, price: float) -> None:
        for (acct_id, aid, sym), rt in list(self.ledger.open_trades.items()):
            if sym != symbol or rt.entry_price <= 0:
                continue
            if self.execution.pending_split(acct_id, aid, sym)[1] != 0:
                continue  # an exit is already in flight
            reason: IntentReason | None = None
            move = (price / rt.entry_price - 1) * rt.direction
            if rt.stop_loss_pct is not None and move <= -rt.stop_loss_pct:
                reason = "stop_loss"
            elif rt.take_profit_pct is not None and move >= rt.take_profit_pct:
                reason = "take_profit"
            elif rt.max_hold_ms is not None and self.now - rt.entry_ts >= rt.max_hold_ms:
                reason = "max_hold"
            if reason is None or not self._exit_due(acct_id, aid, sym):
                continue
            self._system_exit(acct_id, aid, sym, reason, rt.entry_intent_id)
            agent = self.population.agents.get(aid)
            if agent is not None and acct_id.startswith("shadow:"):
                agent.cooldown[sym] = max(agent.cooldown.get(sym, 0), agent.genome.cooldown_bars)

    def _system_exit(
        self, account: str, agent_id: str, symbol: str, reason: IntentReason, parent: str | None
    ) -> None:
        agent = self.population.agents.get(agent_id)
        genome_id = agent.genome.genome_id if agent else ""
        intent = TradeIntent(
            intent_id=self.intent_ids.next(),
            ts=self.now,
            agent_id=agent_id,
            genome_id=genome_id,
            symbol=symbol,
            target_exposure=0.0,
            confidence=1.0,
            reason=reason,
            parent_intent_id=parent,
        )
        self._route(intent, only=account)

    def _exit_due(self, account: str, agent_id: str, symbol: str, retry_ms: int = 1_000) -> bool:
        """May a protective exit be (re)issued now? Not while an exit order is in flight, and at
        most once per ``retry_ms`` per sub-position, so a rejecting venue is retried at a bounded
        rate instead of on every tick."""
        if self.execution.pending_split(account, agent_id, symbol)[1] != 0:
            return False
        key = (account, agent_id, symbol)
        last = self._last_exit_attempt.get(key)
        if last is not None and self.now - last < retry_ms:
            return False
        self._last_exit_attempt[key] = self.now
        return True

    def _flatten_account(self, account: str, reason: IntentReason) -> None:
        acct = self.ledger[account]
        for aid, sym, _p in acct.open_positions():
            if self._exit_due(account, aid, sym):
                self._system_exit(account, aid, sym, reason, None)

    def _flatten_targets(self) -> list[tuple[str, IntentReason]]:
        """Accounts that must be driven flat right now, and why."""
        if self.ended:
            if not self.cfg.challenge.flatten_at_end:
                return []
            return [(a, "challenge_end") for a in sorted(self.ledger.accounts)]
        if self.governor.kill_switch_active():
            return [(CHALLENGE, "kill_switch")]
        if self.cfg.risk.flatten_on_breaker and self.governor.breaker(CHALLENGE):
            return [(CHALLENGE, "circuit_breaker")]
        return []

    def is_flat(self, account: str) -> bool:
        """No ledger position, no working order and (challenge) no venue-reported position."""
        if self.ledger[account].open_positions() or self.execution.has_open_account(account):
            return False
        if account == CHALLENGE:
            return all(
                abs(q) <= (self.instruments[s].qty_step if s in self.instruments else 1e-9) / 2
                for s, q in self.venue_positions.items()
            )
        return True

    def _enforce_flat(self) -> None:
        """Flatten until flat. Runs on every heartbeat and bar close while a kill switch,
        breaker or challenge end is in force: cancels working risk-increasing orders and
        re-issues exits for every open sub-position (rejects, partial fills and UNKNOWN orders
        are retried) until the account is confirmed flat."""
        targets = self._flatten_targets()
        self.flat_confirmed &= {a for a, _ in targets}
        for acct_id, reason in targets:
            if acct_id not in self.ledger.accounts:
                continue
            self.execution.cancel_open(acct_id, self.now)
            self._flatten_account(acct_id, reason)
            if (
                acct_id == CHALLENGE
                and not self.ledger[acct_id].open_positions()
                and not self.execution.has_open_account(acct_id)
                and not self.is_flat(acct_id)
                and self.now - self._last_flat_query >= 5_000
            ):
                # the ledger is flat but the venue last reported a position: ask again
                self._last_flat_query = self.now
                gw = self.gateways.get(CHALLENGE_VENUE)
                if gw is not None:
                    gw.query_positions(CHALLENGE, self.now)
            if self.is_flat(acct_id):
                if acct_id not in self.flat_confirmed:
                    self.flat_confirmed.add(acct_id)
                    if acct_id == CHALLENGE:
                        self._system_event("flatten_complete", {"reason": reason})
            else:
                self.flat_confirmed.discard(acct_id)

    def flatten_pending(self) -> bool:
        """True while some account under a flatten condition is not yet flat."""
        return any(a in self.ledger.accounts and not self.is_flat(a) for a, _ in self._flatten_targets())

    def flatten_incomplete(self) -> None:
        """Drivers call this when they stop with a flatten condition still unmet."""
        left = {
            a: [(aid, sym, p.qty) for aid, sym, p in self.ledger[a].open_positions()]
            for a, _ in self._flatten_targets()
            if a in self.ledger.accounts and not self.is_flat(a)
        }
        log.error("stopping with accounts not flat: %s venue=%s", left, self.venue_positions)
        self._system_event("flatten_incomplete", {"open": left, "venue": dict(self.venue_positions)})

    def _sync_challenge_book(self) -> None:
        """The challenge book mirrors shadow books. If an exit filled in the shadow but not in
        the challenge (reject, stop hit at a different entry, lost fill), close the leftover."""
        chal = self.ledger[CHALLENGE]
        for aid, sym, pos in chal.open_positions():
            agent = self.population.agents.get(aid)
            reason: IntentReason
            if agent is None or not agent.alive:
                reason = "agent_killed"
            elif self.weights.get(aid, 0.0) <= 0:
                reason = "defunded"
            else:
                sq = self.ledger[shadow_account(aid)].agent_qty(aid, sym)
                if sq != 0 and (sq > 0) == (pos.qty > 0):
                    continue
                reason = "desync_exit"
            if self._exit_due(CHALLENGE, aid, sym):
                self.stats[f"sync:{reason}"] += 1
                self._system_exit(CHALLENGE, aid, sym, reason, None)

    # ------------------------------------------------------------------ evolution / capital
    def _evolve(self, ts: int) -> None:
        shadow_eq = {
            a.agent_id: self.ledger[shadow_account(a.agent_id)].last_equity for a in self.population.alive
        }
        res = self.population.evolve(ts, shadow_eq)
        self.last_evolution = res
        self.stats["generations"] += 1
        if self.store is not None:
            for e in res.evaluations.values():
                self.store.add("fitness", e.to_record())
        for agent, _reason in res.killed:
            self._persist_agent(agent)
            for acct_id in (shadow_account(agent.agent_id), CHALLENGE):
                acct = self.ledger[acct_id]
                for aid, sym, _p in acct.open_positions():
                    if aid == agent.agent_id and self._exit_due(acct_id, aid, sym):
                        self._system_exit(acct_id, aid, sym, "agent_killed", None)
            self.weights.pop(agent.agent_id, None)
        for agent in res.demoted + res.reinstated:
            self._persist_agent(agent)
        for agent in res.born:
            self._on_birth(agent)
        if res.champion_id and res.champion_id in self.population.agents:
            self._persist_agent(self.population.agents[res.champion_id])
        self._persist_lineage(self.population.drain_lineage())

    def _rebalance(self, ts: int) -> None:
        evals = self.population.evaluate(ts)
        primary = self.symbols[0]
        regime = (
            self.features.view(primary, ts).regime().value if self.features.history[primary] else "unknown"
        )
        window_start = (
            ts - self.cfg.evolution.eval_generations * self.cfg.evolution.generation_bars * self.bar_ms
        )
        cands: list[AllocationCandidate] = []
        for a in self.population.alive:
            e = evals.get(a.agent_id)
            if e is None:
                continue
            _, r = self.population.returns(a.agent_id, max(a.born_ts, window_start))
            book = self.population.books[a.agent_id]
            labels = tuple(book.regimes[-r.size :]) if r.size else ()
            cands.append(
                AllocationCandidate(
                    agent_id=a.agent_id,
                    status=a.status,
                    eligible=e.eligible,
                    adjusted_fitness=e.adjusted_fitness,
                    ruin_prob=e.report.ruin_prob,
                    net_return=e.report.net_return,
                    bar_returns=r,
                    regime_labels=labels,
                    absolute_fitness=e.report.fitness,
                )
            )
        new = self.allocator.allocate(cands, regime, self.alloc_rng)
        old = self.weights
        self.weights = new
        chal = self.ledger[CHALLENGE]
        for aid in sorted(set(old) | set(new)):
            ow, nw = old.get(aid, 0.0), new.get(aid, 0.0)
            agent = self.population.agents.get(aid)
            if agent is None:
                continue
            if nw == 0 and ow > 0:
                self._lineage_event(ts, agent, LineageEventKind.DEFUNDED, {"weight": ow})
                for a2, sym, _p in chal.open_positions():
                    if a2 == aid and self._exit_due(CHALLENGE, aid, sym):
                        self._system_exit(CHALLENGE, aid, sym, "defunded", None)
            elif nw > 0 and (ow == 0 or abs(nw - ow) / max(ow, 1e-9) > 0.2):
                if ow == 0:
                    self._lineage_event(ts, agent, LineageEventKind.FUNDED, {"weight": nw})
                self._sync_to_shadow(agent)
        if self.store is not None:
            gen = self.population.generation
            for aid, w in new.items():
                self.store.add(
                    "allocations",
                    {
                        "ts": ts,
                        "generation": gen,
                        "agent_id": aid,
                        "weight": w,
                        "capital": w * chal.last_equity,
                        "allocator": self.allocator.name,
                    },
                )

    def _sync_to_shadow(self, agent: Agent) -> None:
        """Mirror an agent's current shadow exposure into the challenge book (after (re)funding)."""
        sacct = self.ledger[shadow_account(agent.agent_id)]
        eq = sacct.last_equity
        if eq <= 0:
            return
        for sym in agent.genome.symbols:
            qty = sacct.agent_qty(agent.agent_id, sym)
            price = self.market[sym].ref_price(self.now)
            if not price or qty == 0:
                continue
            exposure = qty * price / eq
            rt = self.ledger.open_trades.get((shadow_account(agent.agent_id), agent.agent_id, sym))
            intent = TradeIntent(
                intent_id=self.intent_ids.next(),
                ts=self.now,
                agent_id=agent.agent_id,
                genome_id=agent.genome.genome_id,
                symbol=sym,
                target_exposure=exposure,
                confidence=rt.entry_confidence if rt else 0.5,
                reason="rebalance",
                stop_loss_pct=agent.genome.risk.stop_loss_pct,
                take_profit_pct=agent.genome.risk.take_profit_pct,
                max_hold_ms=agent.genome.risk.max_hold_bars * self.bar_ms,
                urgency=agent.genome.execution.urgency,
                max_slippage_bps=agent.genome.execution.max_slippage_bps,
                parent_intent_id=rt.entry_intent_id if rt else None,
            )
            self._route(intent, only=CHALLENGE)

    # ------------------------------------------------------------------ fills / reconciliation
    def _on_fill(self, ev: FillEvent) -> None:
        mo, closed = self.execution.on_fill(ev)
        if mo is None and not closed:
            if self.execution.orphans and self.execution.orphans[-1].event is ev:
                self._system_event(
                    "orphan_fill",
                    {"exec_id": ev.exec_id, "account": ev.account, "client_order_id": ev.client_order_id},
                )
            return
        self.stats["fills"] += 1
        if mo is not None:
            self._persist_order(mo)
            if self.store is not None:
                slip = None
                if mo.ref_price:
                    slip = (ev.price - mo.ref_price) / mo.ref_price * ev.side.sign * 1e4
                self.store.add(
                    "fills",
                    {
                        "exec_id": ev.exec_id,
                        "client_order_id": ev.client_order_id,
                        "account_id": ev.account,
                        "agent_id": mo.agent_id,
                        "symbol": ev.symbol,
                        "side": ev.side.value,
                        "qty": ev.qty,
                        "price": ev.price,
                        "fee": ev.fee,
                        "is_maker": ev.is_maker,
                        "is_liquidation": ev.is_liquidation,
                        "slippage_bps": slip,
                        "ts": ev.ts,
                    },
                )
        for rt in closed:
            self._on_trade_closed(rt)

    def _on_trade_closed(self, rt: RoundTrip) -> None:
        self.stats["round_trips"] += 1
        if rt.account.startswith("shadow:"):
            slip_bps = rt.slippage_cost / rt.entry_notional * 1e4 if rt.entry_notional > 0 else 0.0
            mae_equity = (
                rt.mae_pct * rt.entry_notional / rt.equity_at_entry if rt.equity_at_entry > 0 else 0.0
            )
            self.population.record_trade(
                rt.agent_id,
                TradeSample(
                    ret=rt.return_on_equity,
                    confidence=rt.entry_confidence,
                    slippage_bps=slip_bps,
                    entry_ts=rt.entry_ts,
                    exit_ts=rt.exit_ts or self.now,
                    mae=min(mae_equity, 0.0),
                ),
            )
        if self.store is not None:
            self.store.add(
                "trades",
                {
                    "trade_id": rt.trade_id,
                    "account_id": rt.account,
                    "agent_id": rt.agent_id,
                    "genome_id": rt.genome_id,
                    "symbol": rt.symbol,
                    "direction": rt.direction,
                    "entry_ts": rt.entry_ts,
                    "exit_ts": rt.exit_ts,
                    "entry_price": rt.entry_price,
                    "exit_price": rt.exit_price,
                    "max_qty": rt.max_qty,
                    "entry_notional": rt.entry_notional,
                    "realized": rt.realized,
                    "fees": rt.fees,
                    "funding": rt.funding,
                    "net_pnl": rt.net_pnl,
                    "return_on_equity": rt.return_on_equity,
                    "slippage_cost": rt.slippage_cost,
                    "mfe_pct": rt.mfe_pct,
                    "mae_pct": rt.mae_pct,
                    "entry_confidence": rt.entry_confidence,
                    "entry_intent_id": rt.entry_intent_id,
                    "exit_intent_id": rt.exit_intent_id,
                    "exit_reason": rt.exit_reason,
                },
            )

    def _reconcile_position(self, ev: PositionSnapshot) -> None:
        if ev.account not in self.ledger.accounts:
            return
        internal = self.ledger[ev.account].net_qty(ev.symbol)
        step = self.instruments[ev.symbol].qty_step if ev.symbol in self.instruments else 1e-9
        if ev.account == CHALLENGE:
            self.venue_positions[ev.symbol] = ev.qty
        venue = account_venue(ev.account)
        h = self.health[venue]
        # orders in flight make a transient mismatch legitimate
        in_flight = self.execution.has_open(ev.account, ev.symbol)
        if abs(internal - ev.qty) > step / 2 and not in_flight:
            h.reconcile_ok = False
            h.reconcile_detail = f"{ev.account}:{ev.symbol} internal={internal} venue={ev.qty}"
            self.stats["reconcile_mismatch"] += 1
            self._system_event(
                "reconcile_mismatch",
                {"account": ev.account, "symbol": ev.symbol, "internal": internal, "venue": ev.qty},
            )
        elif not h.reconcile_ok and h.reconcile_detail.startswith(f"{ev.account}:{ev.symbol}"):
            h.reconcile_ok = True
            h.reconcile_detail = ""
            self._system_event("reconcile_recovered", {"account": ev.account, "symbol": ev.symbol})

    def _safety_check(self, equity: float) -> None:
        """Breakers and kill switch. Runs at every bar close *and* every heartbeat, so a trip
        or a touched KILL file acts within a heartbeat, not a bar, and flattening is retried on
        every heartbeat until the account is actually flat."""
        if not self.ended:
            trip = self.governor.check_breakers(self.ledger[CHALLENGE], equity)
            if trip:
                self._system_event("circuit_breaker", {"reason": trip, "equity": equity})
            if self.governor.kill_switch_active():
                if not self.kill_seen:
                    self._system_event("kill_switch", {"equity": equity})
                    self.kill_seen = True
            else:
                self.kill_seen = False
        self._enforce_flat()

    def _reconcile_wallet(self, ev: WalletSnapshot) -> None:
        """Compare venue PnL with ledger PnL since the first snapshot (the venue wallet may hold
        more than the challenge capital, so levels are not comparable; changes are)."""
        self.venue_equity = ev.equity
        ledger_eq = self.ledger[CHALLENGE].equity(self._marks())
        if self.venue_equity_base is None:
            self.venue_equity_base, self.ledger_equity_base = ev.equity, ledger_eq
            return
        if self.execution.open_orders():
            return  # fills in flight make a transient difference legitimate
        assert self.ledger_equity_base is not None
        drift = (ev.equity - self.venue_equity_base) - (ledger_eq - self.ledger_equity_base)
        tol = max(1.0, 0.02 * self.cfg.challenge.starting_capital)
        h = self.health[CHALLENGE_VENUE]
        if abs(drift) > tol:
            if h.reconcile_ok:
                self._system_event(
                    "wallet_drift", {"drift": drift, "venue_equity": ev.equity, "ledger": ledger_eq}
                )
            h.reconcile_ok = False
            h.reconcile_detail = f"wallet pnl drift {drift:+.2f}"
        elif not h.reconcile_ok and h.reconcile_detail.startswith("wallet pnl drift"):
            h.reconcile_ok = True
            h.reconcile_detail = ""
            self._system_event("wallet_reconciled", {"drift": drift})

    def _on_timer(self, ev: TimerEvent) -> None:
        self.execution.check_timeouts(self.now)
        if self.marks:
            self._safety_check(self.ledger[CHALLENGE].equity(self._marks()))
        if len(self._last_exit_attempt) > 10_000:
            cutoff = self.now - 60_000
            self._last_exit_attempt = {k: t for k, t in self._last_exit_attempt.items() if t > cutoff}
        if self.now - self._last_prune >= 600_000:
            self._last_prune = self.now
            self.execution.prune(self.now)
        if self.now - self.last_reconcile_ts >= self.cfg.exchange.reconcile_interval_ms:
            self.last_reconcile_ts = self.now
            gw = self.gateways.get(CHALLENGE_VENUE)
            if gw is not None:
                gw.query_positions(CHALLENGE, self.now)

    # ------------------------------------------------------------------ end of challenge
    def _end_challenge(self, ts: int) -> None:
        if self.ended:
            return
        self.ended = True
        self._system_event("challenge_end", {"equity": self.ledger[CHALLENGE].last_equity})
        self._enforce_flat()

    def finalize(self) -> float:
        """Compute the public score after the driver has drained all in-flight events."""
        marks = self._marks()
        eq = self.ledger[CHALLENGE].equity(marks)
        self.final_equity = eq
        if not self.finalized and self.store is not None:
            for a in self.population.agents.values():
                self._persist_agent(a)
            self.store.upsert(
                "runs",
                {
                    "run_id": self.run_id,
                    "name": self.cfg.challenge.name,
                    "mode": self.cfg.challenge.mode.value,
                    "config": self.cfg.model_dump(mode="json"),
                    "config_fingerprint": self.cfg.fingerprint(),
                    "risk_fingerprint": self.governor.fingerprint,
                    "starting_capital": self.cfg.challenge.starting_capital,
                    "started_ts": self.start_ts,
                    "ends_ts": self.end_ts,
                    "ended_ts": self.now,
                    "final_equity": eq,
                    "status": "finished",
                },
            )
            self.store.flush()
        self.finalized = True
        return eq

    # ------------------------------------------------------------------ persistence helpers
    def _system_event(self, kind: str, detail: dict[str, Any]) -> None:
        self.stats[f"sys:{kind}"] += 1
        if self.store is not None:
            self.store.add("system_events", {"ts": self.now, "kind": kind, "detail": detail})

    def _lineage_event(self, ts: int, agent: Agent, kind: LineageEventKind, details: dict[str, Any]) -> None:
        ev = LineageEvent(
            ts,
            self.population.generation,
            agent.agent_id,
            kind,
            agent.genome.genome_id,
            agent.parent_ids,
            details,
        )
        self._persist_lineage([ev])

    def _persist_lineage(self, events: list[LineageEvent]) -> None:
        for ev in events:
            rec = ev.to_record()
            self.recent_lineage.append(rec)
            if self.store is not None:
                self.store.add("lineage", rec)

    def _persist_agent(self, a: Agent) -> None:
        if self.store is None:
            return
        self.store.upsert(
            "agents",
            {
                "agent_id": a.agent_id,
                "genome_id": a.genome.genome_id,
                "species": a.species,
                "generation": a.generation,
                "parent_ids": list(a.parent_ids),
                "status": a.status.value,
                "strikes": a.strikes,
                "born_ts": a.born_ts,
                "died_ts": a.died_ts,
                "death_reason": a.death_reason,
            },
        )

    def _persist_intent(self, i: TradeIntent) -> None:
        if self.store is None:
            return
        self.store.add(
            "intents",
            {
                "intent_id": i.intent_id,
                "ts": i.ts,
                "agent_id": i.agent_id,
                "genome_id": i.genome_id,
                "symbol": i.symbol,
                "reason": i.reason,
                "direction": i.direction,
                "target_exposure": i.target_exposure,
                "confidence": i.confidence,
                "score": i.score,
                "regime": i.regime,
                "components": i.components,
                "features": i.features_seen,
                "signals": list(i.signals_seen),
                "provider": i.provider,
                "model_response": i.model_response,
                "snapshot_ref": i.snapshot_ref,
                "stop_loss_pct": i.stop_loss_pct,
                "take_profit_pct": i.take_profit_pct,
                "max_hold_ms": i.max_hold_ms,
                "urgency": i.urgency.value,
                "parent_intent_id": i.parent_intent_id,
            },
        )

    def _persist_decision(self, d: RiskDecision) -> None:
        if self.store is not None:
            self.store.add("risk_decisions", d.to_record())

    def _persist_order(self, mo: ManagedOrder) -> None:
        if self.store is None:
            return
        r = mo.request
        self.store.upsert(
            "orders",
            {
                "client_order_id": r.client_order_id,
                "decision_id": mo.decision_id,
                "intent_id": mo.intent_id,
                "account_id": r.account,
                "agent_id": mo.agent_id,
                "symbol": r.symbol,
                "side": r.side.value,
                "qty": r.qty,
                "order_type": r.order_type.value,
                "tif": r.tif.value,
                "limit_price": r.limit_price,
                "status": mo.status.value,
                "filled_qty": mo.filled_qty,
                "avg_fill_price": mo.avg_fill_price,
                "fees": mo.fees,
                "exchange_order_id": mo.exchange_order_id,
                "created_ts": mo.created_ts,
                "acked_ts": mo.acked_ts,
                "last_update_ts": mo.last_update_ts,
                "reason": mo.reason,
            },
        )

    def _persist_snapshots(
        self, ts: int, bars: dict[str, Bar], views: dict[str, FeatureView]
    ) -> dict[str, str]:
        refs: dict[str, str] = {}
        for sym, bar in bars.items():
            ref = f"{self.run_id}:{sym}@{ts}"
            refs[sym] = ref
            if self.store is None:
                continue
            st = self.market[sym]
            self.store.add(
                "snapshots",
                {
                    "snapshot_id": ref,
                    "ts": ts,
                    "symbol": sym,
                    "data": {
                        "bar": {
                            k: getattr(bar, k)
                            for k in (
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "buy_volume",
                                "sell_volume",
                                "n_trades",
                                "vwap",
                            )
                        },
                        "mark": bar.mark_price,
                        "funding": bar.funding_rate,
                        "open_interest": bar.open_interest,
                        "book_imbalance": bar.book_imbalance,
                        "spread_bps": bar.spread_bps,
                        "liq_long": bar.liq_long_notional,
                        "liq_short": bar.liq_short_notional,
                        "stale": bar.stale,
                        "bids": st.book.top_bids(5) if st.book.valid else [],
                        "asks": st.book.top_asks(5) if st.book.valid else [],
                        "regime": views[sym].regime().value if sym in views else "unknown",
                    },
                },
            )
        return refs

    # ------------------------------------------------------------------ read models (API)
    def state(self) -> dict[str, Any]:
        chal = self.ledger[CHALLENGE]
        start = self.cfg.challenge.starting_capital
        eq = chal.last_equity
        return {
            "run_id": self.run_id,
            "name": self.cfg.challenge.name,
            "mode": self.cfg.challenge.mode.value,
            "starting_capital": start,
            "equity": eq,
            "venue_equity": self.venue_equity,
            "pnl": eq - start,
            "pnl_pct": eq / start - 1,
            "peak_equity": chal.peak_equity,
            "drawdown": 1 - eq / chal.peak_equity if chal.peak_equity > 0 else 0.0,
            "now": self.now,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "remaining_ms": max(self.end_ts - self.now, 0),
            "generation": self.population.generation,
            "bar_index": self.bar_index,
            "champion": self.population.champion_id,
            "alive": len(self.population.alive),
            "dead": sum(1 for a in self.population.agents.values() if not a.alive),
            "funded": len(self.weights),
            "ended": self.ended,
            "breaker": self.governor.breaker(CHALLENGE),
            "kill_switch": self.governor.kill_switch_active(),
            "health": {k: vars(v) for k, v in self.health.items()},
            "stats": dict(self.stats),
            "reject_reasons": dict(self.reject_reasons.most_common(12)),
            "risk_fingerprint": self.governor.fingerprint,
        }

    def leaderboard(self) -> list[dict[str, Any]]:
        evals = self.population.last_evaluations
        marks = self._marks()
        chal = self.ledger[CHALLENGE]
        rows = []
        for a in self.population.agents.values():
            sacct = self.ledger.accounts.get(shadow_account(a.agent_id))
            e = evals.get(a.agent_id)
            rows.append(
                {
                    "agent_id": a.agent_id,
                    "species": a.species,
                    "status": a.status.value,
                    "generation": a.generation,
                    "parents": list(a.parent_ids),
                    "shadow_equity": round(sacct.equity(marks), 4) if sacct else None,
                    "shadow_return": round(sacct.equity(marks) / sacct.initial_cash - 1, 5)
                    if sacct
                    else None,
                    "fitness": round(e.adjusted_fitness, 4) if e else None,
                    "eligible": e.eligible if e else False,
                    "trades": len(self.population.books[a.agent_id].trades),
                    "weight": self.weights.get(a.agent_id, 0.0),
                    "live_pnl": round(chal.agent_pnl(a.agent_id, marks), 4),
                    "strikes": a.strikes,
                    "champion": a.agent_id == self.population.champion_id,
                    "death_reason": a.death_reason,
                    "genome_id": a.genome.genome_id,
                }
            )

        def sort_key(r: dict[str, Any]) -> tuple[bool, bool, float]:
            f = r["fitness"]
            return (
                r["status"] == AgentStatus.DEAD.value,
                not r["eligible"],  # proven agents rank above zero-evidence ones
                -(float(f) if f is not None else -math.inf),
            )

        rows.sort(key=sort_key)
        return rows

    def positions(self) -> list[dict[str, Any]]:
        marks = self._marks()
        out = []
        for acct_id in (CHALLENGE,):
            acct = self.ledger[acct_id]
            for aid, sym, p in acct.open_positions():
                m = marks.get(sym, p.avg_price)
                out.append(
                    {
                        "account": acct_id,
                        "agent_id": aid,
                        "symbol": sym,
                        "qty": p.qty,
                        "avg_price": p.avg_price,
                        "mark": m,
                        "unrealized": p.unrealized(m),
                        "notional": abs(p.qty) * m,
                    }
                )
        return out
