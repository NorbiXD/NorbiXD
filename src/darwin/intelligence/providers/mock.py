"""Deterministic mock providers (tests, offline demos, benchmarking baselines).

They are *honest* baselines: they only use information present in the request state (i.e. the
past), so any "edge" they show is legitimately available to a quant rule too. There is no
oracle mock that peeks at future prices — that would silently poison every downstream test.
"""

from __future__ import annotations

import math
import re
import zlib

from darwin.intelligence.providers.base import (
    DecisionAnswer,
    DecisionRequest,
    DecisionResponse,
    NarrativeItem,
    NarrativeRequest,
    NarrativeResponse,
)

_NUM = re.compile(r"(\w[\w()., ]*?)=(-?\d+(?:\.\d+)?(?:e-?\d+)?)")


def _parse_state(state: str) -> dict[str, float]:
    return {k.strip(): float(v) for k, v in _NUM.findall(state)}


class MockDecisionProvider:
    """Direction from a noisy logistic of lagged momentum in the state text."""

    name = "mock-decision"

    def __init__(self, seed: int = 0, noise: float = 0.5) -> None:
        self.seed = seed
        self.noise = noise

    async def list_models(self) -> list[str]:
        return ["mock-1"]

    def decide_sync(self, req: DecisionRequest) -> DecisionResponse:
        feats = _parse_state(req.state)
        z = feats.get("zret_30", 0.0)
        u = (zlib.crc32(f"{self.seed}:{req.request_id}".encode()) % 10_000) / 10_000 - 0.5
        x = z + self.noise * 4 * u
        p_long = 1 / (1 + math.exp(-x))
        probs = {"long": 0.8 * p_long, "short": 0.8 * (1 - p_long), "wait": 0.2}
        answers = {}
        for q in req.questions:
            if q.type == "choice" and q.criteria and set(q.criteria) >= {"long", "short", "wait"}:
                choice = max(probs, key=lambda k: probs[k])
                answers[q.key] = DecisionAnswer(
                    key=q.key, type="choice", choice=choice, probabilities=probs, confidence=probs[choice]
                )
            elif q.type == "noul":
                answers[q.key] = DecisionAnswer(
                    key=q.key, type="noul", probability=p_long, confidence=abs(2 * p_long - 1)
                )
            else:
                opts = list(q.criteria or {"n/a": ""})
                answers[q.key] = DecisionAnswer(
                    key=q.key,
                    type=q.type,
                    choice=opts[0],
                    probabilities={o: 1 / len(opts) for o in opts},
                    confidence=0.0,
                )
        return DecisionResponse(
            request_id=req.request_id,
            provider=self.name,
            model="mock-1",
            answers=answers,
            latency_ms=0,
            raw={"features": feats},
        )

    async def decide(self, req: DecisionRequest) -> DecisionResponse:
        return self.decide_sync(req)


class MockNarrativeProvider:
    """Sentiment = lagged return sign + noise: a stand-in for "the crowd talks about what just moved"."""

    name = "mock-narrative"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.last_returns: dict[str, float] = {}

    async def list_models(self) -> list[str]:
        return ["mock-1"]

    def scan_sync(self, req: NarrativeRequest) -> NarrativeResponse:
        items = []
        for s in req.symbols:
            r = self.last_returns.get(s, 0.0)
            u = (zlib.crc32(f"{self.seed}:{req.request_id}:{s}".encode()) % 10_000) / 10_000 - 0.5
            sent = max(-1.0, min(1.0, math.tanh(r * 200) * 0.7 + u))
            items.append(
                NarrativeItem(
                    symbol=s,
                    sentiment=sent,
                    confidence=0.4 + 0.4 * abs(sent),
                    summary=f"mock narrative for {s}",
                    observed_ts=req.ts,
                )
            )
        return NarrativeResponse(
            request_id=req.request_id, provider=self.name, model="mock-1", items=tuple(items)
        )

    async def scan(self, req: NarrativeRequest) -> NarrativeResponse:
        return self.scan_sync(req)
