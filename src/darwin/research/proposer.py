"""Meta-research proposers: where new species come from.

* :class:`TemplateProposer` — deterministic, offline stand-in for a research agent: composes
  DSL species from conditional templates over the feature API (e.g. "follow breakouts only when
  taker flow confirms", "fade funding extremes while volatility expands"). Useful for tests and
  for exercising the pipeline without an LLM.
* :class:`LLMProposer` — asks a frontier model (Grok/Claude/…, via any ``complete(prompt)``
  coroutine) for new species in the DSL, given the feature API, the rules, and what currently
  works (species × regime attribution). Output is parsed and *still has to pass the sandbox*.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from darwin.research.dsl import FEATURE_API, MATH_API
from darwin.research.sandbox import Proposal

TEMPLATES: list[tuple[str, str]] = [
    (
        "breakout confirmed by taker flow",
        """DESCRIPTION = "follow channel breakouts only when taker flow confirms the direction"
PARAMS = {"ch": [20, 240, "int"], "fw": [3, 30, "int"], "min_flow": [0.02, 0.4]}

def score(v, p):
    pos = v.channel_position(int(p["ch"]))
    fl = v.flow_imbalance(int(p["fw"]))
    if not (math.isfinite(pos) and math.isfinite(fl)):
        return 0.0
    if pos > 1.0 and fl > p["min_flow"]:
        return min(1.0, 0.5 + fl)
    if pos < 0.0 and fl < -p["min_flow"]:
        return -min(1.0, 0.5 - fl)
    return 0.0
""",
    ),
    (
        "funding fade during volatility expansion",
        """DESCRIPTION = "fade extreme funding only while short-term volatility expands"
PARAMS = {"fw": [60, 720, "int"], "z": [0.8, 3.0], "short": [5, 60, "int"], "long": [120, 600, "int"],
          "vx": [1.05, 2.0]}

def score(v, p):
    fz = v.funding_z(int(p["fw"]))
    vr = v.vol_ratio(int(p["short"]), int(p["long"]))
    if not (math.isfinite(fz) and math.isfinite(vr)):
        return 0.0
    if vr < p["vx"] or abs(fz) < p["z"]:
        return 0.0
    return -math.tanh(fz / p["z"])
""",
    ),
    (
        "trend with RSI pullback entry",
        """DESCRIPTION = "trade in the direction of the slow trend, entering on RSI pullbacks"
PARAMS = {"lb": [60, 360, "int"], "rsi": [5, 30, "int"], "band": [10, 35], "zmin": [0.3, 2.0]}

def score(v, p):
    z = v.zret(int(p["lb"]))
    r = v.rsi(int(p["rsi"]))
    if not (math.isfinite(z) and math.isfinite(r)):
        return 0.0
    if z > p["zmin"] and r < 50 - p["band"]:
        return math.tanh(z)
    if z < -p["zmin"] and r > 50 + p["band"]:
        return math.tanh(z)
    return 0.0
""",
    ),
    (
        "momentum scaled by open-interest build-up",
        """DESCRIPTION = "momentum whose conviction grows when open interest is building"
PARAMS = {"lb": [20, 240, "int"], "ow": [10, 240, "int"], "scale": [0.5, 3.0]}

def score(v, p):
    z = v.zret(int(p["lb"]))
    oi = v.oi_change(int(p["ow"]))
    if not math.isfinite(z):
        return 0.0
    boost = 1.0
    if math.isfinite(oi) and oi > 0:
        boost = 1.0 + min(oi * 20.0, 1.0)
    return max(-1.0, min(1.0, math.tanh(z / p["scale"]) * boost))
""",
    ),
]


class TemplateProposer:
    name = "template"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def propose(self, n: int) -> list[Proposal]:
        idx = self.rng.permutation(len(TEMPLATES))[: min(n, len(TEMPLATES))]
        return [
            Proposal(source=TEMPLATES[int(i)][1], rationale=TEMPLATES[int(i)][0], proposer=self.name)
            for i in idx
        ]


PROMPT = """You are the meta-research agent of an evolutionary crypto-futures trading system.
Propose {n} NEW trading signal species that are not simple copies of the existing ones.
Each species is a Python function in a strict DSL:

DESCRIPTION = "<one sentence>"
PARAMS = {{"name": [low, high] or [low, high, "int"|"float"|"log"], ...}}   # 1..8 params

def score(v, p):
    ...   # return a float in [-1, 1]; +1 = strong long, -1 = strong short, 0 = no view

Rules (violations are rejected automatically): no imports, loops, comprehensions, lambdas, try,
keyword arguments, '**', or names starting with '_'. The ONLY attributes allowed are
v.<feature> for features in {features} and math.<fn> for {math}. Builtins: abs, min, max,
float, int, round. Parameters are read as p["name"]. Features return NaN when history is short:
guard with math.isfinite. Signals are evaluated once per 1-minute bar; taker fees are ~11 bps
round trip, so the edge must survive costs.

What currently works (species x regime, mean trade return, t-stat):
{attribution}

Return each species in its own ```python fenced block, preceded by a one-line rationale."""

_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL)


class LLMProposer:
    name = "llm"

    def __init__(self, complete: Callable[[str], Awaitable[str]], name: str = "llm") -> None:
        self.complete = complete
        self.name = name

    def build_prompt(self, n: int, attribution: list[dict[str, Any]]) -> str:
        lines = [
            f"- {a['species']} in {a['regime']}: mean {a['mean_return']:+.4f}, "
            f"t={a['t_stat']:.2f}, n={a['trades']}"
            for a in sorted(attribution, key=lambda r: -abs(r.get("t_stat", 0)))[:15]
        ] or ["- (no attribution yet)"]
        return PROMPT.format(
            n=n, features=sorted(FEATURE_API), math=sorted(MATH_API), attribution="\n".join(lines)
        )

    async def propose(self, n: int, attribution: list[dict[str, Any]] | None = None) -> list[Proposal]:
        text = await self.complete(self.build_prompt(n, attribution or []))
        out: list[Proposal] = []
        for m in _BLOCK.finditer(text):
            before = text[: m.start()].strip().splitlines()
            rationale = before[-1][:200] if before else ""
            out.append(Proposal(source=m.group(1).strip() + "\n", rationale=rationale, proposer=self.name))
        return out[:n]
