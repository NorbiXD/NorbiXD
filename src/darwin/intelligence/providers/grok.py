"""xAI Grok narrative provider using the supported X Search tool — feature-gated off by default.

Uses the xAI Responses API (``POST {base}/v1/responses``) with the server-side ``x_search`` tool
(optionally restricted by ``allowed_x_handles`` and a ``from_date``/``to_date`` window). X data is
only obtained through this supported API — never by scraping. Models are discovered via
``GET {base}/v1/models`` unless ``intelligence.grok.model`` pins one.

The model is asked for a strict JSON object; the parser extracts and validates it and raises
``ProviderError`` on anything malformed (no silent defaults). When xAI flags a response as
``degraded`` (no matching posts; answer synthesised from training data) the items are marked
and their confidence is cut, because stale "news" is worse than none.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from darwin.intelligence.providers.base import (
    NarrativeItem,
    NarrativeRequest,
    NarrativeResponse,
    ProviderError,
)

SYSTEM_PROMPT = (
    "You are a crypto market intelligence analyst. Use X search to find what changed in the narrative "
    "for the requested perpetual-futures symbols during the time window. Report only information "
    "published inside the window; if nothing material happened, say so with sentiment 0 and low "
    'confidence. Respond with ONLY a JSON object: {"items": [{"symbol": str|null, "sentiment": '
    'float in [-1,1], "confidence": float in [0,1], "catalyst": str|null, "summary": str, '
    '"citations": [str]}]}'
)


class GrokProvider:
    name = "grok"

    def __init__(
        self,
        base_url: str = "https://api.x.ai",
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get("XAI_API_KEY")
        if not key:
            raise ProviderError("XAI_API_KEY not set")
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
        return [row["id"] for row in r.json().get("data", []) if isinstance(row, dict) and "id" in row]

    async def model(self) -> str:
        if self._model:
            return self._model
        grok = sorted((m for m in await self.list_models() if m.startswith("grok")), reverse=True)
        if not grok:
            raise ProviderError("no grok models available")
        self._model = grok[0]
        return self._model

    def build_request(self, req: NarrativeRequest, model: str) -> dict[str, Any]:
        now = datetime.fromtimestamp(req.ts / 1000, tz=UTC) if req.ts else datetime.now(tz=UTC)
        start = now - timedelta(minutes=req.lookback_minutes)
        tool: dict[str, Any] = {
            "type": "x_search",
            "from_date": start.date().isoformat(),
            "to_date": now.date().isoformat(),
        }
        if req.allowed_handles:
            tool["allowed_x_handles"] = list(req.allowed_handles)[:20]
        user = (
            f"Symbols: {', '.join(req.symbols)}. Window: {start.isoformat()} to {now.isoformat()} (UTC). "
            "What moved the narrative, and in which direction for each symbol?"
        )
        return {
            "model": model,
            "input": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "tools": [tool],
        }

    async def scan(self, req: NarrativeRequest) -> NarrativeResponse:
        model = req.model or await self.model()
        t0 = time.monotonic()
        r = await self.client.post("/v1/responses", json=self.build_request(req, model))
        latency = int((time.monotonic() - t0) * 1000)
        if r.status_code >= 400:
            raise ProviderError(f"xai HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        text = output_text(body)
        degraded = bool(body.get("degraded", False))
        items = parse_items(text, req, degraded)
        return NarrativeResponse(
            request_id=req.request_id,
            provider=self.name,
            model=str(body.get("model", model)),
            items=items,
            latency_ms=latency,
            degraded=degraded,
            raw=body,
        )


def output_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return str(body["output_text"])
    chunks: list[str] = []
    for item in body.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if (
                isinstance(c, dict)
                and c.get("type") in ("output_text", "text")
                and isinstance(c.get("text"), str)
            ):
                chunks.append(c["text"])
    if not chunks:
        raise ProviderError("no text output in response")
    return "\n".join(chunks)


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_items(text: str, req: NarrativeRequest, degraded: bool) -> tuple[NarrativeItem, ...]:
    m = _JSON_OBJ.search(text)
    if not m:
        raise ProviderError("no JSON object in model output")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ProviderError(f"invalid JSON: {e}") from e
    rows = obj.get("items")
    if not isinstance(rows, list):
        raise ProviderError("JSON has no 'items' list")
    allowed = set(req.symbols)
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sym = row.get("symbol")
        if sym is not None and sym not in allowed:
            continue  # the model may not invent instruments
        try:
            sentiment = max(-1.0, min(1.0, float(row.get("sentiment", 0.0))))
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        if degraded:
            confidence *= 0.25
        cites = tuple(str(c) for c in row.get("citations", []) if isinstance(c, str))[:10]
        out.append(
            NarrativeItem(
                symbol=sym,
                sentiment=sentiment,
                confidence=confidence,
                summary=str(row.get("summary", ""))[:1_000],
                catalyst=(str(row["catalyst"])[:300] if row.get("catalyst") else None),
                citations=cites,
                observed_ts=req.ts or None,
            )
        )
    return tuple(out)
