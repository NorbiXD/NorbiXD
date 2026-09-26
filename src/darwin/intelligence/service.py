"""Multi-speed intelligence: fast-path decisions and slow-path narrative, off the tick path.

The service is an engine *bar observer*. It never blocks a decision:

* **fast path** (e.g. Jev): every ``fast_every_bars`` bars, per symbol, a compact textual state
  built from the same FeatureView agents see is sent to a :class:`DecisionProvider`. The typed
  answer becomes ``IntelligenceSignal(topic="model_direction", value=P(long)-P(short))``.
* **slow path** (e.g. Grok + X Search): every ``slow_interval_ms`` a :class:`NarrativeProvider`
  scan produces ``IntelligenceSignal(topic="x_narrative")`` per item.

Signals are delivered with the timestamp at which they *arrive*: in live modes the wall-clock
receive time; in deterministic replay (sync mock providers) ``now + simulated latency`` through
the replay scheduler. Agents therefore only see a model output after it could actually have
existed. Failures are counted and a provider that keeps failing is paused (circuit breaker).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from darwin.core.events import IntelligenceSignal
from darwin.features.engine import FeatureView
from darwin.intelligence.providers.base import (
    DIRECTION_QUESTION,
    DecisionProvider,
    DecisionRequest,
    DecisionResponse,
    NarrativeProvider,
    NarrativeRequest,
    NarrativeResponse,
)
from darwin.intelligence.providers.mock import MockDecisionProvider, MockNarrativeProvider

log = logging.getLogger(__name__)

Deliver = Callable[[int, IntelligenceSignal], None]


def render_state(v: FeatureView) -> str:
    """What a model is allowed to know: the agent-visible features, nothing else."""

    def f(x: float) -> str:
        return f"{x:.6g}" if math.isfinite(x) else "nan"

    parts = [
        f"symbol={v.symbol}",
        f"close={f(v.close())}",
        f"ret_5={f(v.ret(5))}",
        f"ret_30={f(v.ret(30))}",
        f"zret_30={f(v.zret(30))}",
        f"vol_30={f(v.vol(30))}",
        f"rsi_14={f(v.rsi(14))}",
        f"funding={f(v.funding())}",
        f"funding_z_240={f(v.funding_z(240))}",
        f"oi_change_60={f(v.oi_change(60))}",
        f"book_imbalance={f(v.book_imbalance())}",
        f"flow_imbalance_10={f(v.flow_imbalance(10))}",
        f"liq_intensity_10={f(v.liq_intensity(10))}",
        f"regime={v.regime().value}",
    ]
    return "; ".join(parts)


@dataclass
class ProviderHealth:
    calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    paused_until: int = 0
    latencies_ms: deque[int] = field(default_factory=lambda: deque(maxlen=200))

    def ok(self, now: int) -> bool:
        return now >= self.paused_until


class IntelligenceService:
    def __init__(
        self,
        deliver: Deliver,
        clock_ms: Callable[[], int],
        decision: DecisionProvider | None = None,
        narrative: NarrativeProvider | None = None,
        fast_every_bars: int = 5,
        slow_interval_ms: int = 900_000,
        sim_fast_latency_ms: int = 400,
        sim_slow_latency_ms: int = 20_000,
        narrative_half_life_ms: int = 2 * 3_600_000,
        decision_half_life_ms: int = 30 * 60_000,
        max_consecutive_failures: int = 5,
        pause_ms: int = 600_000,
    ) -> None:
        self.deliver = deliver
        self.clock_ms = clock_ms
        self.decision = decision
        self.narrative = narrative
        self.fast_every_bars = fast_every_bars
        self.slow_interval_ms = slow_interval_ms
        self.sim_fast_latency_ms = sim_fast_latency_ms
        self.sim_slow_latency_ms = sim_slow_latency_ms
        self.narrative_half_life_ms = narrative_half_life_ms
        self.decision_half_life_ms = decision_half_life_ms
        self.max_consecutive_failures = max_consecutive_failures
        self.pause_ms = pause_ms
        self.health: dict[str, ProviderHealth] = {}
        self._bars = 0
        self._last_slow = -(10**18)
        self._inflight: set[str] = set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self.emitted = 0

    # ------------------------------------------------------------------ observer entry
    def on_bar(self, engine: Any, ts: int, views: dict[str, FeatureView]) -> None:
        self._bars += 1
        if self.decision is not None and self._bars % self.fast_every_bars == 0:
            for sym, v in sorted(views.items()):
                if v.stale or not v.ready(60):
                    continue
                req = DecisionRequest(
                    request_id=f"fast:{sym}:{ts}",
                    state=render_state(v.fresh()),
                    questions=(DIRECTION_QUESTION,),
                    symbol=sym,
                    ts=ts,
                )
                self._run_decision(req, ts)
        if self.narrative is not None and ts - self._last_slow >= self.slow_interval_ms:
            self._last_slow = ts
            if isinstance(self.narrative, MockNarrativeProvider):
                for sym, v in views.items():
                    if v.ready(60):
                        self.narrative.last_returns[sym] = v.ret(60)
            req_n = NarrativeRequest(request_id=f"slow:{ts}", symbols=tuple(sorted(views)), ts=ts)
            self._run_narrative(req_n, ts)

    # ------------------------------------------------------------------ execution styles
    def _health(self, name: str) -> ProviderHealth:
        h = self.health.get(name)
        if h is None:
            h = ProviderHealth()
            self.health[name] = h
        return h

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _failed(self, name: str, err: Exception) -> None:
        h = self._health(name)
        h.failures += 1
        h.consecutive_failures += 1
        # the type matters: httpx timeouts stringify to an empty message
        log.warning("provider %s failed: %s: %s", name, type(err).__name__, err)
        if h.consecutive_failures >= self.max_consecutive_failures:
            h.paused_until = self.clock_ms() + self.pause_ms
            h.consecutive_failures = 0

    def _run_decision(self, req: DecisionRequest, ts: int) -> None:
        assert self.decision is not None
        name = self.decision.name
        h = self._health(name)
        if not h.ok(ts) or req.symbol in self._inflight:
            return
        h.calls += 1
        if isinstance(self.decision, MockDecisionProvider):  # deterministic replay path
            resp = self.decision.decide_sync(req)
            self._emit_decision(resp, req, ts + self.sim_fast_latency_ms)
            return
        key = req.symbol or ""
        self._inflight.add(key)

        async def go() -> None:
            try:
                resp = await self.decision.decide(req)  # type: ignore[union-attr]
                self._health(name).consecutive_failures = 0
                self._emit_decision(resp, req, self.clock_ms())
            except Exception as e:  # provider errors never propagate into the engine
                self._failed(name, e)
            finally:
                self._inflight.discard(key)

        self._spawn(go())

    def _run_narrative(self, req: NarrativeRequest, ts: int) -> None:
        assert self.narrative is not None
        name = self.narrative.name
        h = self._health(name)
        if not h.ok(ts) or "narrative" in self._inflight:
            return
        h.calls += 1
        if isinstance(self.narrative, MockNarrativeProvider):
            self._emit_narrative(self.narrative.scan_sync(req), ts + self.sim_slow_latency_ms)
            return
        self._inflight.add("narrative")

        async def go() -> None:
            try:
                resp = await self.narrative.scan(req)  # type: ignore[union-attr]
                self._health(name).consecutive_failures = 0
                self._emit_narrative(resp, self.clock_ms())
            except Exception as e:
                self._failed(name, e)
            finally:
                self._inflight.discard("narrative")

        self._spawn(go())

    # ------------------------------------------------------------------ conversion
    def _emit_decision(self, resp: DecisionResponse, req: DecisionRequest, arrive_ts: int) -> None:
        ans = resp.answers.get("direction")
        if ans is None:
            return
        p = ans.probabilities
        value = max(-1.0, min(1.0, p.get("long", 0.0) - p.get("short", 0.0)))
        self._health(resp.provider).latencies_ms.append(resp.latency_ms)
        sig = IntelligenceSignal(
            ts=arrive_ts,
            signal_id=f"SIG:{req.request_id}",
            source="fast_path",
            symbol=req.symbol,
            topic="model_direction",
            value=value,
            confidence=ans.confidence,
            half_life_ms=self.decision_half_life_ms,
            observed_ts=req.ts,
            provider=f"{resp.provider}/{resp.model}",
            payload={
                "choice": ans.choice,
                "probabilities": p,
                "latency_ms": resp.latency_ms,
                "state": req.state,
            },
        )
        self.emitted += 1
        self.deliver(arrive_ts, sig)

    def _emit_narrative(self, resp: NarrativeResponse, arrive_ts: int) -> None:
        self._health(resp.provider).latencies_ms.append(resp.latency_ms)
        for i, item in enumerate(resp.items):
            sig = IntelligenceSignal(
                ts=arrive_ts,
                signal_id=f"SIG:{resp.request_id}:{i}",
                source="slow_path",
                symbol=item.symbol,
                topic=item.topic,
                value=item.sentiment,
                confidence=item.confidence,
                half_life_ms=self.narrative_half_life_ms,
                observed_ts=item.observed_ts,
                provider=f"{resp.provider}/{resp.model}",
                payload={
                    "summary": item.summary,
                    "catalyst": item.catalyst,
                    "citations": list(item.citations),
                    "degraded": resp.degraded,
                    "latency_ms": resp.latency_ms,
                },
            )
            self.emitted += 1
            self.deliver(arrive_ts, sig)

    async def drain(self) -> None:
        while pending := [t for t in self._tasks if not t.done()]:  # see BybitExecutionGateway.drain
            await asyncio.gather(*pending, return_exceptions=True)
