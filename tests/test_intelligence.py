from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from darwin.attribution.explain import explain
from darwin.core.events import IntelligenceSignal
from darwin.intelligence.providers.base import (
    DIRECTION_QUESTION,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    NarrativeRequest,
    ProviderError,
)
from darwin.intelligence.providers.grok import GrokProvider
from darwin.intelligence.providers.jev import JevProvider
from darwin.intelligence.providers.mock import MockDecisionProvider
from darwin.intelligence.service import IntelligenceService
from darwin.market.synthetic import SyntheticMarket
from darwin.persistence.store import AuditStore
from darwin.runtime.replay import build_replay
from tests.conftest import T0, make_config

# --------------------------------------------------------------------------- Jev (System One)


def jev_transport(answers: dict[str, Any], models: list[str] | None = None) -> tuple[httpx.MockTransport, list[Any]]:
    seen: list[Any] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content) if req.content else None, dict(req.headers)))
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in (models or ["jev-1.13.0", "jev-latest"])]})
        if req.url.path == "/v1/systemone":
            return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 50}})
        return httpx.Response(404)

    return httpx.MockTransport(handler), seen


def dreq(questions: tuple[DecisionQuestion, ...] = (DIRECTION_QUESTION,)) -> DecisionRequest:
    return DecisionRequest(request_id="r1", state="symbol=BTCUSDT; zret_30=1.2", questions=questions)


async def test_jev_discovers_model_and_sends_system_one_request() -> None:
    t, seen = jev_transport({"direction": {"type": "choice", "choice": "long",
                                           "probabilities": {"long": 0.7, "short": 0.1, "wait": 0.2},
                                           "confidence": 0.7}})
    p = JevProvider(api_key="k", transport=t)
    resp = await p.decide(dreq())
    assert seen[0][0] == "/v1/models"  # model discovered, not hard-coded
    path, body, headers = seen[1]
    assert path == "/v1/systemone" and body["model"] == "jev-latest"
    assert body["questions"]["direction"]["type"] == "choice"
    assert set(body["questions"]["direction"]["criteria"]) == {"long", "short", "wait"}
    assert headers["authorization"] == "Bearer k"
    a = resp.answers["direction"]
    assert a.choice == "long" and a.probabilities["long"] == 0.7 and a.confidence == 0.7
    await p.close()


async def test_jev_rejects_malformed_or_out_of_domain_answers() -> None:
    t, _ = jev_transport({"direction": {"type": "choice", "choice": "moon", "confidence": 0.9}})
    p = JevProvider(api_key="k", model="jev-latest", transport=t)
    with pytest.raises(ProviderError):
        await p.decide(dreq())
    t2, _ = jev_transport({})
    p2 = JevProvider(api_key="k", model="jev-latest", transport=t2)
    with pytest.raises(ProviderError):
        await p2.decide(dreq())


async def test_jev_noul_answer() -> None:
    q = DecisionQuestion(key="breakout_valid", type="noul", instructions="Is this breakout valid?")
    t, _ = jev_transport({"breakout_valid": {"type": "noul", "probability": 0.81}})
    p = JevProvider(api_key="k", model="jev-latest", transport=t)
    resp = await p.decide(dreq((q,)))
    assert resp.answers["breakout_valid"].probability == 0.81


def test_jev_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        JevProvider()


# --------------------------------------------------------------------------- Grok (xAI + X Search)


def grok_transport(text: str, degraded: bool = False) -> tuple[httpx.MockTransport, list[Any]]:
    seen: list[Any] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content) if req.content else None))
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "grok-4"}, {"id": "grok-4.7"}, {"id": "other"}]})
        return httpx.Response(200, json={
            "model": "grok-4.7", "degraded": degraded,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        })

    return httpx.MockTransport(handler), seen


async def test_grok_uses_x_search_tool_and_parses_items() -> None:
    text = 'Here you go: {"items": [{"symbol": "SOLUSDT", "sentiment": 0.6, "confidence": 0.8, ' \
           '"catalyst": "ETF filing", "summary": "..", "citations": ["https://x.com/a/status/1"]}, ' \
           '{"symbol": "FAKEUSDT", "sentiment": 1, "confidence": 1}]}'
    t, seen = grok_transport(text)
    p = GrokProvider(api_key="k", transport=t)
    req = NarrativeRequest(request_id="n1", symbols=("SOLUSDT", "BTCUSDT"), ts=1_758_888_000_000,
                           allowed_handles=("solana",))
    resp = await p.scan(req)
    _, body = seen[-1]
    tool = body["tools"][0]
    assert tool["type"] == "x_search" and tool["allowed_x_handles"] == ["solana"] and "from_date" in tool
    assert body["model"] == "grok-4.7"  # discovered
    assert len(resp.items) == 1 and resp.items[0].symbol == "SOLUSDT"  # invented instrument dropped
    assert resp.items[0].citations == ("https://x.com/a/status/1",)


async def test_grok_degraded_cuts_confidence_and_bad_json_raises() -> None:
    t, _ = grok_transport('{"items": [{"symbol": null, "sentiment": -0.5, "confidence": 0.8}]}', degraded=True)
    p = GrokProvider(api_key="k", model="grok-4", transport=t)
    resp = await p.scan(NarrativeRequest(request_id="n", symbols=("BTCUSDT",)))
    assert resp.degraded and resp.items[0].confidence == pytest.approx(0.2)
    t2, _ = grok_transport("sorry, I cannot")
    p2 = GrokProvider(api_key="k", model="grok-4", transport=t2)
    with pytest.raises(ProviderError):
        await p2.scan(NarrativeRequest(request_id="n", symbols=("BTCUSDT",)))


# --------------------------------------------------------------------------- service / leakage


def test_replay_signals_arrive_after_latency_and_are_auditable(tmp_path: Path) -> None:
    cfg = make_config(challenge={"duration_hours": 6},
                      evolution={"population_size": 10, "generation_bars": 60, "min_trades": 2})
    store = AuditStore(f"sqlite:///{tmp_path / 'i.db'}", run_id="intel")
    market = SyntheticMarket(symbols=cfg.challenge.symbols, start_ts=T0, duration_ms=cfg.duration_ms + 60_000,
                             step_ms=5_000, seed=2)
    h = build_replay(cfg, market.events(), T0, store=store, run_id="intel", intelligence=True)
    h.driver.run()
    store.flush()
    sigs = store.query("signals")
    assert sigs, "fast and slow paths must emit signals"
    for s in sigs:
        assert s["ts"] > s["observed_ts"], "a model output can only exist after the state it was asked about"
    assert {s["topic"] for s in sigs} >= {"model_direction", "x_narrative"}
    # any intent that consumed signals only saw ones that had already arrived
    intents = [i for i in store.query("intents") if i["signals"]]
    by_id = {s["signal_id"]: s for s in sigs}
    for i in intents:
        for sid in i["signals"]:
            assert by_id[sid]["ts"] <= i["ts"]
    if intents:
        ex = explain(store, intents[0]["intent_id"])
        assert ex is not None and ex["knew"]["signals_detail"]


async def test_async_provider_failures_trip_circuit_breaker() -> None:
    class Flaky:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        async def list_models(self) -> list[str]:
            return ["x"]

        async def decide(self, req: DecisionRequest) -> DecisionResponse:
            self.calls += 1
            raise ProviderError("down")

    now = [T0]
    delivered: list[IntelligenceSignal] = []
    flaky = Flaky()
    svc = IntelligenceService(lambda ts, s: delivered.append(s), lambda: now[0], decision=flaky, fast_every_bars=1,
                              max_consecutive_failures=3, pause_ms=60_000)
    req = dreq()
    for _ in range(6):
        svc._run_decision(req.model_copy(update={"symbol": "BTCUSDT"}), now[0])
        await svc.drain()
    assert flaky.calls == 3 and not delivered
    assert svc.health["flaky"].paused_until == T0 + 60_000
    now[0] += 61_000
    svc._run_decision(req.model_copy(update={"symbol": "BTCUSDT"}), now[0])
    await svc.drain()
    assert flaky.calls == 4


def test_mock_decision_is_deterministic_and_uses_only_the_state() -> None:
    m = MockDecisionProvider(seed=1)
    a = m.decide_sync(dreq())
    b = m.decide_sync(dreq())
    assert a == b
    assert asyncio.run(m.decide(dreq())) == a
