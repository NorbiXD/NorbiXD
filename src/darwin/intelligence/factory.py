"""Build the intelligence service from ``challenge.yaml`` (providers are feature-gated)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from darwin.config.challenge import ChallengeConfig
from darwin.core.types import Mode
from darwin.intelligence.providers.base import DecisionProvider, NarrativeProvider, ProviderError
from darwin.intelligence.providers.grok import GrokProvider
from darwin.intelligence.providers.jev import JevProvider
from darwin.intelligence.providers.mock import MockDecisionProvider, MockNarrativeProvider
from darwin.intelligence.service import Deliver, IntelligenceService

log = logging.getLogger(__name__)


def build_intelligence(
    cfg: ChallengeConfig, deliver: Deliver, clock_ms: Callable[[], int]
) -> IntelligenceService | None:
    ic = cfg.intelligence
    deterministic = cfg.challenge.mode is Mode.REPLAY
    decision: DecisionProvider | None = None
    narrative: NarrativeProvider | None = None
    # an explicitly enabled real provider that cannot start is an error when no mock is configured:
    # never let a missing key silently turn "AI mode" into "no AI"
    strict = not ic.mock.enabled
    if ic.jev.enabled and not deterministic:
        try:
            decision = JevProvider(ic.jev.base_url, model=ic.jev.model, timeout_s=ic.jev.timeout_s)
        except ProviderError as e:
            if strict:
                raise
            log.warning("jev unavailable, the mock stands in: %s", e)
    if ic.grok.enabled and not deterministic:
        try:
            narrative = GrokProvider(ic.grok.base_url, model=ic.grok.model, timeout_s=ic.grok.timeout_s)
        except ProviderError as e:
            if strict:
                raise
            log.warning("grok unavailable, the mock stands in: %s", e)
    if ic.mock.enabled:
        decision = decision or MockDecisionProvider(seed=cfg.sim.seed)
        narrative = narrative or MockNarrativeProvider(seed=cfg.sim.seed)
    if decision is None and narrative is None:
        return None
    return IntelligenceService(
        deliver,
        clock_ms,
        decision=decision,
        narrative=narrative,
        slow_interval_ms=int(ic.grok.interval_s * 1000),
    )
