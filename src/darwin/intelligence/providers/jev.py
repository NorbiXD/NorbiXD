"""TypeSafe System One (Jev) decision provider — EXPERIMENTAL, feature-gated off by default.

Protocol as publicly documented at the time of writing (early access, Sept 2026):

* ``GET  {base}/v1/models``     -> the model names this account may use (aliases such as
  ``jev-latest``; versioned ids are also accepted). We *discover* the model here instead of
  hard-coding an id; ``intelligence.jev.model`` in ``challenge.yaml`` can pin one.
* ``POST {base}/v1/systemone``  with ``{"state": str, "model": str, "questions": {key: {"type":
  "noul"|"choice"|"score", "instructions": str, "criteria": {...}}}}`` and header
  ``Authorization: Bearer $TYPESAFE_API_KEY``. The response carries ``answers[key]`` with the
  selected ``choice``, per-option ``probabilities`` and a ``confidence``.

The response parser is deliberately tolerant (field names are validated but alternative
spellings are accepted) because the API is in early access; anything unparseable raises
``ProviderError`` rather than inventing an answer. What remains to verify against the real
service is listed in progress.md.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from darwin.intelligence.providers.base import (
    DecisionAnswer,
    DecisionRequest,
    DecisionResponse,
    ProviderError,
)


class JevProvider:
    name = "jev"

    def __init__(
        self,
        base_url: str = "https://api.typesafe.ai",
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ProviderError("TYPESAFE_API_KEY not set")
        self._model = model
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def list_models(self) -> list[str]:
        r = await self.client.get("/v1/models")
        r.raise_for_status()
        body = r.json()
        rows = body.get("data") or body.get("models") or []
        out = []
        for row in rows:
            name = row.get("id") or row.get("name") if isinstance(row, dict) else row
            if isinstance(name, str):
                out.append(name)
        return out

    async def model(self) -> str:
        if self._model:
            return self._model
        models = await self.list_models()
        if not models:
            raise ProviderError("no System One models available to this account")
        # prefer the moving alias if offered, otherwise the first listed model
        latest = [m for m in models if m.endswith("-latest")]
        self._model = latest[0] if latest else models[0]
        return self._model

    async def decide(self, req: DecisionRequest) -> DecisionResponse:
        model = req.model or await self.model()
        questions: dict[str, Any] = {}
        for q in req.questions:
            spec: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
            if q.criteria:
                spec["criteria"] = q.criteria
            questions[q.key] = spec
        t0 = time.monotonic()
        r = await self.client.post(
            "/v1/systemone", json={"state": req.state, "model": model, "questions": questions}
        )
        latency = int((time.monotonic() - t0) * 1000)
        if r.status_code >= 400:
            raise ProviderError(f"jev HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        answers = parse_answers(body, req)
        return DecisionResponse(
            request_id=req.request_id,
            provider=self.name,
            model=str(body.get("model", model)),
            answers=answers,
            latency_ms=latency,
            raw=body,
        )


def _prob(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 1.0 else None


def parse_answers(body: dict[str, Any], req: DecisionRequest) -> dict[str, DecisionAnswer]:
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, dict):
        raise ProviderError("response has no 'answers' object")
    out: dict[str, DecisionAnswer] = {}
    for q in req.questions:
        a = raw_answers.get(q.key)
        if not isinstance(a, dict):
            raise ProviderError(f"missing answer for {q.key!r}")
        probs_raw = a.get("probabilities") or {}
        probs = (
            {str(k): p for k, v in probs_raw.items() if (p := _prob(v)) is not None}
            if isinstance(probs_raw, dict)
            else {}
        )
        conf = _prob(a.get("confidence"))
        if q.type == "noul":
            p = _prob(a.get("probability", a.get("p_yes", a.get("value"))))
            if p is None and "yes" in probs:
                p = probs["yes"]
            if p is None:
                raise ProviderError(f"noul answer {q.key!r} has no probability")
            out[q.key] = DecisionAnswer(
                key=q.key, type="noul", probability=p, confidence=conf if conf is not None else abs(2 * p - 1)
            )
            continue
        choice = a.get("choice", a.get("level", a.get("answer")))
        if choice is None and probs:
            choice = max(probs, key=lambda k: probs[k])
        if choice is None:
            raise ProviderError(f"{q.type} answer {q.key!r} has no choice")
        choice = str(choice)
        if q.criteria and choice not in q.criteria:
            raise ProviderError(f"answer {choice!r} not among options {sorted(q.criteria)}")
        out[q.key] = DecisionAnswer(
            key=q.key,
            type=q.type,
            choice=choice,
            probabilities=probs,
            confidence=conf if conf is not None else probs.get(choice, 0.0),
        )
    return out
