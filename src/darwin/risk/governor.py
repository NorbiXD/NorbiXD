"""The Risk Governor: deterministic, immutable, outside every agent.

Agents *request* risk via ``TradeIntent.target_exposure`` (a multiple of their capital). The
governor converts that request into a lot-rounded order quantity that fits inside the
operator's envelope — or rejects it. It is the only component that decides size.

Design rules
------------
* Limits come from a frozen ``RiskLimits`` object; the governor exposes no mutators.
* Risk-reducing orders are always allowed (even when data is stale, the kill switch is on or a
  breaker has tripped): getting flat must never be blocked by the safety system itself.
* Risk-increasing orders pass every check or are clipped/rejected with explicit reasons that are
  persisted for audit.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field

from darwin.config.challenge import InstrumentSpec, RiskLimits
from darwin.core.ids import Sequence
from darwin.core.intent import TradeIntent
from darwin.market.state import SymbolState
from darwin.portfolio.ledger import Account


@dataclass
class VenueHealth:
    """Per-venue operational health, maintained by the execution layer."""

    consecutive_errors: int = 0
    reconcile_ok: bool = True
    reconcile_detail: str = ""
    connected: bool = True
    halted: str = ""  # e.g. "audit_store_unavailable": no new risk without an audit trail

    def healthy(self, max_errors: int) -> tuple[bool, str]:
        if self.halted:
            return False, self.halted
        if not self.connected:
            return False, "venue_disconnected"
        if self.consecutive_errors >= max_errors:
            return False, f"api_errors({self.consecutive_errors})"
        if not self.reconcile_ok:
            return False, f"reconciliation_mismatch({self.reconcile_detail})"
        return True, ""


@dataclass(frozen=True)
class RiskDecision:
    decision_id: str
    intent_id: str
    account_id: str
    agent_id: str
    symbol: str
    ts: int
    approved: bool
    risk_increasing: bool
    current_qty: float
    pending_qty: float
    requested_qty: float
    target_qty: float
    order_qty: float  # signed, lot-rounded delta to send
    ref_price: float | None
    agent_capital: float
    account_equity: float
    reasons: tuple[str, ...]
    clipped: bool
    limits_fingerprint: str

    def to_record(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "intent_id": self.intent_id,
            "account_id": self.account_id,
            "agent_id": self.agent_id,
            "symbol": self.symbol,
            "ts": self.ts,
            "approved": self.approved,
            "risk_increasing": self.risk_increasing,
            "current_qty": self.current_qty,
            "pending_qty": self.pending_qty,
            "requested_qty": self.requested_qty,
            "target_qty": self.target_qty,
            "order_qty": self.order_qty,
            "ref_price": self.ref_price,
            "agent_capital": self.agent_capital,
            "account_equity": self.account_equity,
            "reasons": list(self.reasons),
            "clipped": self.clipped,
            "limits_fingerprint": self.limits_fingerprint,
        }


@dataclass(frozen=True)
class Reservations:
    """Exposure already committed by in-flight orders of *all* agents in an account.

    All agents decide on the same bar close and orders take tens of milliseconds to fill, so
    limits computed from filled positions alone would let N agents each consume the full
    headroom. Risk-increasing open orders therefore reserve exposure until they resolve.
    """

    qty_by_symbol: dict[str, float] = field(default_factory=dict)  # gross |qty| still to fill
    new_positions: int = 0  # (agent, symbol) pairs flat now but with an increasing order in flight


@dataclass
class _AccountGuard:
    orders_ts: deque[int] = field(default_factory=lambda: deque(maxlen=10_000))
    recent: dict[tuple[str, str], tuple[int, float]] = field(default_factory=dict)
    breaker: str | None = None


def round_qty_toward_zero(qty: float, step: float) -> float:
    n = math.floor(abs(qty) / step + 1e-9)
    return math.copysign(round(n * step, 10), qty) if n > 0 else 0.0


def liquidation_distance(equity: float, gross_notional: float, mmr: float) -> float:
    """Uniform adverse move (fraction) that would take the account to maintenance margin.

    Conservative for crypto, where positions are highly correlated: assumes every position moves
    against us by the same fraction simultaneously.
    """
    if gross_notional <= 0:
        return math.inf
    return (equity - mmr * gross_notional) / (gross_notional * (1 - mmr))


class RiskGovernor:
    def __init__(self, limits: RiskLimits, instruments: dict[str, InstrumentSpec]) -> None:
        self._limits = limits  # frozen pydantic model; no setter is exposed
        self._instruments = dict(instruments)
        self._fingerprint = limits.fingerprint()
        self._guards: dict[str, _AccountGuard] = {}
        self._ids = Sequence("RD", width=8)
        self._manual_kill = False
        # resolved once so a later chdir cannot silently move the kill switch
        self._kill_file = os.path.abspath(limits.kill_switch_file) if limits.kill_switch_file else None

    # read-only views -------------------------------------------------------------
    @property
    def limits(self) -> RiskLimits:
        return self._limits

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def instrument(self, symbol: str) -> InstrumentSpec:
        return self._instruments[symbol]

    # operator controls (not reachable from agents) -------------------------------
    def engage_kill_switch(self) -> None:
        self._manual_kill = True

    def release_kill_switch(self) -> None:
        self._manual_kill = False

    def kill_switch_active(self) -> bool:
        if self._manual_kill:
            return True
        return self._kill_file is not None and os.path.exists(self._kill_file)

    @property
    def kill_switch_path(self) -> str | None:
        return self._kill_file

    def breaker(self, account_id: str) -> str | None:
        g = self._guards.get(account_id)
        return g.breaker if g else None

    def check_breakers(self, acct: Account, equity: float) -> str | None:
        """Update drawdown / daily-loss breakers for a challenge account. Returns new trip reason."""
        if acct.kind != "challenge":
            return None
        g = self._guard(acct.account_id)
        lim = self._limits
        if g.breaker and g.breaker.startswith("daily_loss") and not self._daily_loss_hit(acct, equity):
            g.breaker = None  # daily breaker resets on a new day
        if g.breaker:
            return None
        dd = 1 - equity / acct.peak_equity if acct.peak_equity > 0 else 0.0
        if dd >= lim.max_drawdown_pct:
            g.breaker = f"max_drawdown({dd:.1%})"
            return g.breaker
        if self._daily_loss_hit(acct, equity):
            loss = 1 - equity / acct.day_start_equity
            g.breaker = f"daily_loss({loss:.1%})"
            return g.breaker
        return None

    def _daily_loss_hit(self, acct: Account, equity: float) -> bool:
        if acct.day_start_equity <= 0:
            return False
        return 1 - equity / acct.day_start_equity >= self._limits.daily_loss_limit_pct

    def _guard(self, account_id: str) -> _AccountGuard:
        g = self._guards.get(account_id)
        if g is None:
            g = _AccountGuard()
            self._guards[account_id] = g
        return g

    # the decision ------------------------------------------------------------------
    def evaluate(
        self,
        intent: TradeIntent,
        acct: Account,
        agent_capital: float,
        market: SymbolState,
        marks: dict[str, float],
        pending_qty: float,
        now: int,
        health: VenueHealth,
        reservations: Reservations | None = None,
    ) -> RiskDecision:
        lim = self._limits
        res = reservations or Reservations()
        reasons: list[str] = []
        clipped = False
        sym = intent.symbol
        current = acct.agent_qty(intent.agent_id, sym)
        equity = acct.equity(marks)
        # fresh price for anything that adds risk; any last-known price is acceptable to get flat
        ref_fresh = market.ref_price(now)
        ref = ref_fresh or market.ref_price()

        def decide(
            approved: bool, target: float = 0.0, order: float = 0.0, requested: float = 0.0, inc: bool = False
        ) -> RiskDecision:
            d = RiskDecision(
                decision_id=self._ids.next(),
                intent_id=intent.intent_id,
                account_id=acct.account_id,
                agent_id=intent.agent_id,
                symbol=sym,
                ts=now,
                approved=approved,
                risk_increasing=inc,
                current_qty=current,
                pending_qty=pending_qty,
                requested_qty=requested,
                target_qty=target,
                order_qty=order,
                ref_price=ref,
                agent_capital=agent_capital,
                account_equity=equity,
                reasons=tuple(reasons),
                clipped=clipped,
                limits_fingerprint=self._fingerprint,
            )
            if approved:
                g = self._guard(acct.account_id)
                g.orders_ts.append(now)
                g.recent[(intent.agent_id, sym)] = (now, target)
            return d

        # ---- structural sanity
        spec = self._instruments.get(sym)
        if spec is None or (lim.allowed_symbols and sym not in lim.allowed_symbols):
            reasons.append("symbol_not_allowed")
            return decide(False)
        if not math.isfinite(intent.target_exposure):
            reasons.append("non_finite_target")
            return decide(False)
        if ref is None or ref <= 0:
            reasons.append("no_reference_price")
            return decide(False)
        if abs(pending_qty) > 0:
            reasons.append("order_pending")
            return decide(False)

        requested = intent.target_exposure * max(agent_capital, 0.0) / ref
        target = requested
        increasing = (target != 0 and (current == 0 or (target > 0) != (current > 0))) or abs(target) > abs(
            current
        ) + spec.qty_step * 0.5

        if not increasing:
            # ---- risk-reducing path: always allowed
            if target == 0 or (current != 0 and (target > 0) != (current > 0)):
                order = -current  # exact close (positions are lot multiples)
                target = 0.0
            else:
                order = round_qty_toward_zero(target - current, spec.qty_step)
                if abs(order) < spec.min_qty:
                    reasons.append("reduce_below_min_qty")
                    return decide(False, target, 0.0, requested)
            if order == 0:
                reasons.append("no_change")
                return decide(False, target, 0.0, requested)
            return decide(True, target, order, requested, inc=False)

        # ---- risk-increasing path
        is_flip = current != 0 and (target > 0) != (current > 0)

        def reject_inc() -> RiskDecision:
            # a rejected flip still closes the existing position: never stay in a position the
            # agent no longer wants just because the new one is disallowed
            if is_flip:
                reasons.append("flip_downgraded_to_close")
                return decide(True, 0.0, -current, requested, inc=False)
            return decide(False, target, 0.0, requested, True)

        if self.kill_switch_active():
            reasons.append("kill_switch")
            return reject_inc()
        g = self._guard(acct.account_id)
        if g.breaker:
            reasons.append(f"circuit_breaker:{g.breaker}")
            return reject_inc()
        ok, why = health.healthy(lim.max_consecutive_api_errors)
        if not ok:
            reasons.append(why)
            return reject_inc()
        if ref_fresh is None or market.is_stale(now, lim.max_data_staleness_ms):
            reasons.append(f"stale_data({market.staleness_ms(now)}ms,{market.book.invalid_reason or 'age'})")
            return reject_inc()
        if equity <= 0:
            reasons.append("non_positive_equity")
            return reject_inc()
        if lim.require_stop_loss and intent.stop_loss_pct is None:
            reasons.append("missing_stop_loss")
            return reject_inc()
        stop = intent.stop_loss_pct or lim.max_stop_loss_pct
        if stop > lim.max_stop_loss_pct or stop < lim.min_stop_loss_pct:
            reasons.append(f"stop_loss_out_of_bounds({stop:.4f})")
            return reject_inc()
        if intent.max_slippage_bps > lim.max_slippage_bps:
            reasons.append("slippage_tolerance_too_wide")
            return reject_inc()

        # rate limit
        while g.orders_ts and now - g.orders_ts[0] > 60_000:
            g.orders_ts.popleft()
        if len(g.orders_ts) >= lim.max_orders_per_minute:
            reasons.append("order_rate_limit")
            return reject_inc()

        # duplicate protection: same agent/symbol/target inside the dedup window
        prev = g.recent.get((intent.agent_id, sym))
        if prev is not None and now - prev[0] < lim.dedup_window_ms:
            pt = prev[1]
            if pt != 0 and (pt > 0) == (target > 0) and abs(pt - target) <= 0.05 * abs(pt):
                reasons.append("duplicate_intent")
                return reject_inc()

        # concurrent positions: filled sub-positions + new ones already in flight
        if current == 0 and len(acct.open_positions()) + res.new_positions >= lim.max_concurrent_positions:
            reasons.append("max_concurrent_positions")
            return reject_inc()

        # leverage: per agent
        max_agent_qty = lim.max_agent_leverage * agent_capital / ref
        if abs(target) > max_agent_qty:
            target = math.copysign(max_agent_qty, target)
            clipped = True
            reasons.append("clip:agent_leverage")

        # exposure after the change: per symbol and gross (gross of agent sub-positions), counting
        # exposure reserved by every agent's in-flight risk-increasing orders
        reserved_gross = sum(
            q * (ref if s == sym else marks.get(s, 0.0)) for s, q in res.qty_by_symbol.items()
        )
        gross_now = acct.gross_notional(marks) + reserved_gross
        sym_now = acct.symbol_gross_notional(sym, ref) + res.qty_by_symbol.get(sym, 0.0) * ref
        cur_notional = abs(current) * ref

        def headroom(limit_notional: float, used: float) -> float:
            return max(limit_notional - (used - cur_notional), 0.0) / ref

        max_sym_qty = headroom(lim.max_symbol_leverage * equity, sym_now)
        if abs(target) > max_sym_qty:
            target = math.copysign(max_sym_qty, target)
            clipped = True
            reasons.append("clip:symbol_exposure")
        max_gross_qty = headroom(lim.max_gross_leverage * equity, gross_now)
        if abs(target) > max_gross_qty:
            target = math.copysign(max_gross_qty, target)
            clipped = True
            reasons.append("clip:gross_exposure")

        # liquidation distance: required buffer = max(min distance, 1.5 x stop)
        mmr = lim.maintenance_margin_rate
        x_req = max(lim.min_liquidation_distance_pct, 1.5 * stop)
        gross_cap = equity / (mmr + x_req * (1 - mmr))
        max_liq_qty = max(gross_cap - (gross_now - cur_notional), 0.0) / ref
        if abs(target) > max_liq_qty:
            target = math.copysign(max_liq_qty, target)
            clipped = True
            reasons.append("clip:liquidation_distance")
        # instrument leverage cap
        max_inst_qty = spec.max_leverage * equity / ref
        if abs(target) > max_inst_qty:
            target = math.copysign(max_inst_qty, target)
            clipped = True
            reasons.append("clip:instrument_leverage")

        # lot rounding and minimums
        target = round_qty_toward_zero(target, spec.qty_step)
        order = round(target - current, 10)
        if target != 0 and (target > 0) != (requested > 0):
            reasons.append("direction_lost_in_clipping")
            return reject_inc()
        if abs(order) < spec.min_qty or abs(order) * ref < spec.min_notional:
            reasons.append("below_min_order")
            return reject_inc()
        # post-trade check (defence in depth against arithmetic slips above)
        post_gross = gross_now - cur_notional + abs(target) * ref
        if liquidation_distance(equity, post_gross, mmr) < x_req - 1e-9:
            reasons.append("liquidation_distance_violation")
            return reject_inc()
        return decide(True, target, order, requested, True)
