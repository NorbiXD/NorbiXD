"""Agent runtime: evaluates a genome against identical market state and emits TradeIntents."""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.agents.genome import Genome
from darwin.agents.primitives import PRIMITIVES
from darwin.core.ids import Sequence
from darwin.core.intent import IntentReason, TradeIntent
from darwin.core.types import AgentStatus, Regime
from darwin.features.engine import FeatureView


@dataclass
class Agent:
    agent_id: str
    genome: Genome
    generation: int
    parent_ids: tuple[str, ...] = ()
    status: AgentStatus = AgentStatus.ALIVE
    born_ts: int = 0
    died_ts: int | None = None
    death_reason: str | None = None
    strikes: int = 0
    cooldown: dict[str, int] = field(default_factory=dict)
    last_target: dict[str, float] = field(default_factory=dict)
    #: symbol -> (target sign, ts) of the last emitted intent; suppresses re-emitting the same
    #: desire every bar when the previous intent was rejected or is still executing
    last_emit: dict[str, tuple[int, int]] = field(default_factory=dict)

    RETRY_BARS = 5

    @property
    def alive(self) -> bool:
        return self.status is not AgentStatus.DEAD

    @property
    def species(self) -> str:
        return self.genome.species

    def decide(
        self,
        views: dict[str, FeatureView],
        positions: dict[str, float],
        now: int,
        bar_ms: int,
        ids: Sequence,
        snapshot_refs: dict[str, str] | None = None,
    ) -> list[TradeIntent]:
        """Pure function of (genome, views, current position direction). No I/O.

        ``positions`` maps symbol -> the agent's current signed exposure sign/size in its
        evaluation book; only the sign is used for entry/exit logic.
        """
        g = self.genome
        intents: list[TradeIntent] = []
        for sym in g.symbols:
            base = views.get(sym)
            if base is None or base.stale or not base.ready(g.lookback_bars):
                continue
            v = base.fresh()
            if self.cooldown.get(sym, 0) > 0:
                self.cooldown[sym] -= 1
            regime = v.regime()
            pos = positions.get(sym, 0.0)
            pos_sign = (pos > 0) - (pos < 0)

            gated = bool(g.allowed_regimes) and regime not in g.allowed_regimes
            components: dict[str, float] = {}
            num = 0.0
            den = 0.0
            if not gated:
                for i, term in enumerate(g.terms):
                    s = PRIMITIVES[term.primitive].score(v, term.params)
                    components[f"{i}:{term.primitive}"] = round(s, 6)
                    num += term.weight * s
                    den += abs(term.weight)
            score = num / den if den > 0 else 0.0

            target_sign = pos_sign
            reason: IntentReason | None = None
            if gated:
                if pos_sign != 0:
                    target_sign, reason = 0, "regime_gate"
            elif pos_sign == 0:
                if self.cooldown.get(sym, 0) <= 0:
                    if score >= g.entry_threshold:
                        target_sign, reason = 1, "entry"
                    elif score <= -g.entry_threshold:
                        target_sign, reason = -1, "entry"
            elif pos_sign > 0:
                if score <= -g.entry_threshold:
                    target_sign, reason = -1, "flip"
                elif score < g.exit_threshold:
                    target_sign, reason = 0, "exit"
            elif score >= g.entry_threshold:
                target_sign, reason = 1, "flip"
            elif score > -g.exit_threshold:
                target_sign, reason = 0, "exit"

            if reason is None:
                continue
            prev = self.last_emit.get(sym)
            if prev is not None and prev[0] == target_sign and now - prev[1] < self.RETRY_BARS * bar_ms:
                continue
            self.last_emit[sym] = (target_sign, now)
            if target_sign == 0:
                self.cooldown[sym] = g.cooldown_bars
                target = 0.0
            else:
                mag = abs(score) if g.risk.confidence_scaling else 1.0
                target = target_sign * g.risk.exposure * max(mag, 0.25)
            self.last_target[sym] = target
            intents.append(
                TradeIntent(
                    intent_id=ids.next(),
                    ts=now,
                    agent_id=self.agent_id,
                    genome_id=g.genome_id,
                    symbol=sym,
                    target_exposure=target,
                    confidence=min(1.0, abs(score)),
                    reason=reason,
                    stop_loss_pct=g.risk.stop_loss_pct if target_sign != 0 else None,
                    take_profit_pct=g.risk.take_profit_pct if target_sign != 0 else None,
                    max_hold_ms=g.risk.max_hold_bars * bar_ms if target_sign != 0 else None,
                    urgency=g.execution.urgency,
                    max_slippage_bps=g.execution.max_slippage_bps,
                    score=round(score, 6),
                    components=components,
                    regime=regime.value if isinstance(regime, Regime) else str(regime),
                    features_seen=dict(v.accessed),
                    signals_seen=tuple(sorted(v.signals_seen)),
                    provider=g.provider,
                    snapshot_ref=(snapshot_refs or {}).get(sym),
                )
            )
        return intents
