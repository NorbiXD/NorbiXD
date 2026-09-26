"""Double-entry-style accounting for accounts, agent sub-positions and round-trip trades.

Two kinds of account share this code:

* ``shadow:<agent_id>`` — one per agent, standardised evaluation capital, simulated fills. This
  is where *fitness evidence* comes from: every agent (funded or not) is evaluated on identical
  market data with identical costs, independent of how much real capital it currently has.
* ``challenge`` — the single real account (simulated, testnet or live). Many agents hold
  attributed sub-positions in it; the exchange nets them. Invariant checked by reconciliation:
  ``Σ agent qty == venue net position`` per symbol.

Positions only change on fills (never on order acknowledgements).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from darwin.core.types import Side

EPS = 1e-12


def _snap(x: float) -> float:
    x = round(x, 10)
    return 0.0 if abs(x) < 1e-10 else x


@dataclass
class Position:
    qty: float = 0.0  # signed
    avg_price: float = 0.0
    realized: float = 0.0
    fees: float = 0.0
    funding: float = 0.0  # positive = paid

    def apply_fill(self, side: Side, qty: float, price: float) -> float:
        """Apply a fill; returns realized PnL from any closed portion."""
        signed = qty * side.sign
        realized = 0.0
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            new_qty = self.qty + signed
            self.avg_price = (self.avg_price * abs(self.qty) + price * qty) / abs(new_qty)
            self.qty = _snap(new_qty)
        else:
            closing = min(abs(signed), abs(self.qty))
            direction = 1.0 if self.qty > 0 else -1.0
            realized = closing * (price - self.avg_price) * direction
            remaining = abs(signed) - closing
            self.qty = _snap(self.qty + math.copysign(closing, signed))
            if self.qty == 0:
                self.avg_price = 0.0
            if remaining > EPS:
                self.qty = _snap(math.copysign(remaining, signed))
                self.avg_price = price
        self.realized += realized
        return realized

    def unrealized(self, mark: float) -> float:
        return self.qty * (mark - self.avg_price) if self.qty else 0.0


@dataclass
class RoundTrip:
    """A trade from flat to flat for one agent/symbol in one account."""

    trade_id: str
    account: str
    agent_id: str
    genome_id: str
    symbol: str
    direction: int  # +1 long, -1 short
    entry_ts: int
    entry_intent_id: str | None
    entry_confidence: float
    entry_price: float = 0.0
    max_qty: float = 0.0
    entry_notional: float = 0.0
    equity_at_entry: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    realized: float = 0.0
    slippage_cost: float = 0.0  # quote currency, positive = cost vs reference price
    mfe_pct: float = 0.0  # max favourable excursion vs entry (fraction)
    mae_pct: float = 0.0  # max adverse excursion (fraction, <= 0)
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_hold_ms: int | None = None
    exit_ts: int | None = None
    exit_price: float | None = None
    exit_intent_id: str | None = None
    exit_reason: str | None = None
    fills: int = 0

    @property
    def net_pnl(self) -> float:
        return self.realized - self.fees - self.funding

    @property
    def return_on_equity(self) -> float:
        return self.net_pnl / self.equity_at_entry if self.equity_at_entry > 0 else 0.0

    @property
    def closed(self) -> bool:
        return self.exit_ts is not None

    def update_excursion(self, high: float, low: float) -> None:
        if self.entry_price <= 0:
            return
        if self.direction > 0:
            fav = high / self.entry_price - 1
            adv = low / self.entry_price - 1
        else:
            fav = 1 - low / self.entry_price
            adv = 1 - high / self.entry_price
        self.mfe_pct = max(self.mfe_pct, fav)
        self.mae_pct = min(self.mae_pct, adv)


@dataclass
class Account:
    account_id: str
    kind: str  # "shadow" | "challenge"
    initial_cash: float
    cash: float = 0.0
    positions: dict[tuple[str, str], Position] = field(default_factory=dict)  # (agent, symbol)
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day: int = -1
    halted_reason: str | None = None
    last_equity: float = 0.0

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.initial_cash
        self.peak_equity = self.initial_cash
        self.day_start_equity = self.initial_cash
        self.last_equity = self.initial_cash

    def pos(self, agent_id: str, symbol: str) -> Position:
        key = (agent_id, symbol)
        p = self.positions.get(key)
        if p is None:
            p = Position()
            self.positions[key] = p
        return p

    def agent_qty(self, agent_id: str, symbol: str) -> float:
        p = self.positions.get((agent_id, symbol))
        return p.qty if p else 0.0

    def net_qty(self, symbol: str) -> float:
        return _snap(sum(p.qty for (_, s), p in self.positions.items() if s == symbol))

    def symbols(self) -> set[str]:
        return {s for (_, s), p in self.positions.items() if p.qty != 0}

    def open_positions(self) -> list[tuple[str, str, Position]]:
        return [(a, s, p) for (a, s), p in self.positions.items() if p.qty != 0]

    def unrealized(self, marks: dict[str, float]) -> float:
        return sum(p.unrealized(marks[s]) for (_, s), p in self.positions.items() if p.qty and s in marks)

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.unrealized(marks)

    def gross_notional(self, marks: dict[str, float]) -> float:
        """Gross of *agent* sub-positions (conservative: ignores cross-agent netting)."""
        return sum(abs(p.qty) * marks[s] for (_, s), p in self.positions.items() if p.qty and s in marks)

    def symbol_gross_notional(self, symbol: str, mark: float) -> float:
        return sum(abs(p.qty) * mark for (_, s), p in self.positions.items() if s == symbol and p.qty)

    def agent_pnl(self, agent_id: str, marks: dict[str, float]) -> float:
        total = 0.0
        for (a, s), p in self.positions.items():
            if a != agent_id:
                continue
            total += p.realized - p.fees - p.funding
            if p.qty and s in marks:
                total += p.unrealized(marks[s])
        return total


class Ledger:
    def __init__(self) -> None:
        self.accounts: dict[str, Account] = {}
        self.open_trades: dict[tuple[str, str, str], RoundTrip] = {}
        self.closed_trades: list[RoundTrip] = []  # drained by the engine for persistence/fitness
        self._trade_seq = 0

    def open_account(self, account_id: str, kind: str, cash: float) -> Account:
        if account_id in self.accounts:
            raise ValueError(f"account {account_id} exists")
        acct = Account(account_id=account_id, kind=kind, initial_cash=cash)
        self.accounts[account_id] = acct
        return acct

    def __getitem__(self, account_id: str) -> Account:
        return self.accounts[account_id]

    # ------------------------------------------------------------------ fills
    def on_fill(
        self,
        *,
        account_id: str,
        agent_id: str,
        genome_id: str,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        fee: float,
        ts: int,
        intent_id: str | None,
        intent_reason: str | None,
        confidence: float,
        ref_price: float | None,
        equity_hint: float,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        max_hold_ms: int | None = None,
    ) -> list[RoundTrip]:
        acct = self.accounts[account_id]
        pos = acct.pos(agent_id, symbol)
        before = pos.qty
        signed = qty * side.sign
        key = (account_id, agent_id, symbol)
        closed: list[RoundTrip] = []

        # split into closing part and opening part (a flip is both)
        closing_qty = 0.0
        if before != 0 and (before > 0) != (signed > 0):
            closing_qty = min(abs(signed), abs(before))
        opening_qty = qty - closing_qty

        realized = pos.apply_fill(side, qty, price)
        pos.fees += fee
        acct.cash += realized - fee
        slip = 0.0
        if ref_price:
            slip = (price - ref_price) * side.sign * qty

        rt = self.open_trades.get(key)
        if closing_qty > 0 and rt is not None:
            frac = closing_qty / qty
            rt.realized += realized
            rt.fees += fee * frac
            rt.slippage_cost += slip * frac
            rt.fills += 1
            if pos.qty == 0 or (pos.qty > 0) != (rt.direction > 0):
                rt.exit_ts = ts
                rt.exit_price = price
                rt.exit_intent_id = intent_id
                rt.exit_reason = intent_reason
                closed.append(rt)
                self.closed_trades.append(rt)  # cleared by the engine every bar
                del self.open_trades[key]
                rt = None
        if opening_qty > EPS:
            frac = opening_qty / qty
            if rt is None:
                self._trade_seq += 1
                rt = RoundTrip(
                    trade_id=f"R{self._trade_seq:07d}",
                    account=account_id,
                    agent_id=agent_id,
                    genome_id=genome_id,
                    symbol=symbol,
                    direction=1 if signed > 0 else -1,
                    entry_ts=ts,
                    entry_intent_id=intent_id,
                    entry_confidence=confidence,
                    equity_at_entry=equity_hint,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    max_hold_ms=max_hold_ms,
                )
                self.open_trades[key] = rt
            rt.fees += fee * frac
            rt.slippage_cost += slip * frac
            rt.fills += 1
            rt.entry_price = pos.avg_price
            rt.max_qty = max(rt.max_qty, abs(pos.qty))
            rt.entry_notional = max(rt.entry_notional, abs(pos.qty) * pos.avg_price)
        return closed

    def on_funding(
        self, account_id: str, symbol: str, rate: float, mark: float, amount: float | None = None
    ) -> float:
        """Apply a funding settlement. Agent shares are ``qty * mark * rate``; if the venue reports
        the actual ``amount`` the account is charged exactly that, and any difference (e.g. a
        fill in flight at settlement) is returned as the unattributed residual."""
        acct = self.accounts[account_id]
        total = 0.0
        for (agent_id, s), p in acct.positions.items():
            if s != symbol or p.qty == 0:
                continue
            share = p.qty * mark * rate
            p.funding += share
            total += share
            rt = self.open_trades.get((account_id, agent_id, symbol))
            if rt is not None:
                rt.funding += share
        charged = total if amount is None else amount
        acct.cash -= charged
        return charged - total

    def liquidation_allocation(self, account_id: str, symbol: str, side: Side) -> list[tuple[str, float]]:
        """Which agents a venue liquidation fill on ``symbol`` belongs to (pro rata by size)."""
        acct = self.accounts[account_id]
        closing_sign = -side.sign
        holders = [
            (a, abs(p.qty))
            for (a, s), p in acct.positions.items()
            if s == symbol and p.qty != 0 and (p.qty > 0) == (closing_sign > 0)
        ]
        total = sum(q for _, q in holders)
        return [(a, q / total) for a, q in holders] if total > 0 else []

    # ------------------------------------------------------------------ marking
    def mark(
        self,
        marks: dict[str, float],
        highs: dict[str, float],
        lows: dict[str, float],
        ts: int,
        bar_start_ts: int | None = None,
    ) -> None:
        for rt in self.open_trades.values():
            if rt.symbol not in highs:
                continue
            if bar_start_ts is not None and rt.entry_ts > bar_start_ts:
                # entered mid-bar: the bar's extremes may predate the entry; use the close only
                rt.update_excursion(marks[rt.symbol], marks[rt.symbol])
            else:
                rt.update_excursion(highs[rt.symbol], lows[rt.symbol])
        day = ts // 86_400_000
        for acct in self.accounts.values():
            eq = acct.equity(marks)
            acct.last_equity = eq
            acct.peak_equity = max(acct.peak_equity, eq)
            if day != acct.day:
                acct.day = day
                acct.day_start_equity = eq

    def drain_closed(self) -> list[RoundTrip]:
        out, self.closed_trades = self.closed_trades, []
        return out
