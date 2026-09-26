"""Signal primitives — the building blocks of every agent genome.

A primitive maps a :class:`FeatureView` and its own parameters to a directional score in
``[-1, 1]`` (``+1`` = maximal long conviction). Named species ("momentum", "breakout", ...) are
genomes with a single dominant primitive; hybrids emerge from crossover and structural mutation.

Primitives must be pure functions of the view: no state, no I/O, no clock. The Level-2 research
sandbox registers new primitives through :func:`register_primitive` after validation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from darwin.agents.params import ParamSpec
from darwin.features.engine import FeatureView

ScoreFn = Callable[[FeatureView, Mapping[str, float]], float]


@dataclass(frozen=True)
class Primitive:
    name: str
    fn: ScoreFn
    params: Mapping[str, ParamSpec]
    description: str
    origin: str = "builtin"  # builtin | sandbox:<proposal_id>

    def score(self, view: FeatureView, params: Mapping[str, float]) -> float:
        try:
            s = float(self.fn(view, params))
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0
        if not math.isfinite(s):
            return 0.0
        return max(-1.0, min(1.0, s))


PRIMITIVES: dict[str, Primitive] = {}


def register_primitive(p: Primitive, *, replace: bool = False) -> None:
    if p.name in PRIMITIVES and not replace:
        raise ValueError(f"primitive {p.name!r} already registered")
    PRIMITIVES[p.name] = p


def _ok(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


# ----------------------------------------------------------------------------- builtins


def _momentum(v: FeatureView, p: Mapping[str, float]) -> float:
    z = v.zret(int(p["lookback"]))
    return math.tanh(z / p["scale"]) if _ok(z) else 0.0


def _breakout(v: FeatureView, p: Mapping[str, float]) -> float:
    pos = v.channel_position(int(p["channel"]))
    if not _ok(pos):
        return 0.0
    b = p["buffer"]
    if pos > 1.0 + b:
        return min(1.0, 0.5 + (pos - 1.0 - b) * 5)
    if pos < -b:
        return -min(1.0, 0.5 + (-b - pos) * 5)
    return 0.0


def _mean_reversion(v: FeatureView, p: Mapping[str, float]) -> float:
    z = v.zprice(int(p["lookback"]))
    if not _ok(z):
        return 0.0
    e = p["entry_z"]
    mag = min(1.0, max(0.0, (abs(z) - 0.5 * e) / (0.5 * e)))
    return -math.copysign(mag, z)


def _order_flow(v: FeatureView, p: Mapping[str, float]) -> float:
    fi = v.flow_imbalance(int(p["window"]))
    bi = v.book_imbalance()
    if not _ok(fi, bi):
        return 0.0
    w = p["book_weight"]
    return math.tanh(3.0 * (w * bi + (1 - w) * fi))


def _funding_oi(v: FeatureView, p: Mapping[str, float]) -> float:
    fz = v.funding_z(int(p["funding_window"]))
    oi = v.oi_change(int(p["oi_window"]))
    if not _ok(fz, oi):
        return 0.0
    if abs(fz) < p["z_entry"] or oi <= 0:
        return 0.0  # only fade crowding that is still building
    return -math.copysign(min(1.0, abs(fz) / (2 * p["z_entry"])), fz)


def _liquidation(v: FeatureView, p: Mapping[str, float]) -> float:
    w = int(p["window"])
    li = v.liq_intensity(w)
    imb = v.liq_imbalance(w)
    if not _ok(li, imb) or li < p["intensity"]:
        return 0.0
    follow = int(p["mode"]) == 1
    return imb if follow else -imb


def _volatility(v: FeatureView, p: Mapping[str, float]) -> float:
    short, long = int(p["short"]), int(p["long"])
    vr = v.vol_ratio(short, long)
    r = v.ret(short)
    if not _ok(vr, r) or vr < p["expansion"]:
        return 0.0
    return math.copysign(min(1.0, vr - p["expansion"] + 0.5), r)


def _contrarian(v: FeatureView, p: Mapping[str, float]) -> float:
    r = v.rsi(int(p["rsi_len"]))
    if not _ok(r):
        return 0.0
    band = p["band"]
    if r > 100 - band:
        return -min(1.0, (r - (100 - band)) / band + 0.3)
    if r < band:
        return min(1.0, (band - r) / band + 0.3)
    return 0.0


SIGNAL_TOPICS = ("", "x_narrative", "model_direction", "external")


def _narrative(v: FeatureView, p: Mapping[str, float]) -> float:
    topic = SIGNAL_TOPICS[int(p["topic"])]
    s = v.signal(topic)
    if abs(s) < p["min_abs"]:
        return 0.0
    return math.tanh(p["gain"] * s)


for _p in (
    Primitive(
        "momentum",
        _momentum,
        {
            "lookback": ParamSpec(3, 240, "int", log=True, doc="bars"),
            "scale": ParamSpec(0.3, 3.0, doc="z-score that maps to ~0.76 conviction"),
        },
        "Follow volatility-normalised returns.",
    ),
    Primitive(
        "breakout",
        _breakout,
        {
            "channel": ParamSpec(10, 360, "int", log=True, doc="Donchian channel length, bars"),
            "buffer": ParamSpec(0.0, 0.3, doc="fraction of channel width beyond the edge"),
        },
        "Trade closes outside the prior high/low channel.",
    ),
    Primitive(
        "mean_reversion",
        _mean_reversion,
        {
            "lookback": ParamSpec(10, 360, "int", log=True),
            "entry_z": ParamSpec(1.0, 4.0, doc="z-score of price vs SMA for full conviction"),
        },
        "Fade deviations of price from its moving average.",
    ),
    Primitive(
        "order_flow",
        _order_flow,
        {
            "window": ParamSpec(1, 30, "int", log=True),
            "book_weight": ParamSpec(0.0, 1.0),
        },
        "Follow taker-flow and order-book imbalance.",
    ),
    Primitive(
        "funding_oi",
        _funding_oi,
        {
            "funding_window": ParamSpec(30, 720, "int", log=True),
            "oi_window": ParamSpec(5, 240, "int", log=True),
            "z_entry": ParamSpec(0.5, 3.0),
        },
        "Fade extreme funding while open interest is still building (crowded positioning).",
    ),
    Primitive(
        "liquidation",
        _liquidation,
        {
            "window": ParamSpec(1, 30, "int", log=True),
            "intensity": ParamSpec(0.01, 2.0, log=True, doc="liq notional / avg bar notional"),
            "mode": ParamSpec(0, 1, "choice", choices=("fade", "follow")),
        },
        "React to liquidation cascades (fade the overshoot or follow the squeeze).",
    ),
    Primitive(
        "volatility",
        _volatility,
        {
            "short": ParamSpec(3, 60, "int", log=True),
            "long": ParamSpec(60, 600, "int", log=True),
            "expansion": ParamSpec(1.05, 3.0),
        },
        "Follow the direction of a volatility expansion.",
    ),
    Primitive(
        "contrarian",
        _contrarian,
        {
            "rsi_len": ParamSpec(3, 60, "int", log=True),
            "band": ParamSpec(5, 35, doc="RSI distance from 0/100 that triggers"),
        },
        "Fade RSI extremes.",
    ),
    Primitive(
        "narrative",
        _narrative,
        {
            "topic": ParamSpec(0, 3, "choice", choices=SIGNAL_TOPICS),
            "min_abs": ParamSpec(0.0, 0.6),
            "gain": ParamSpec(0.5, 4.0),
        },
        "Trade decay-weighted intelligence signals (X narrative, fast-path model decisions, external feeds).",
    ),
):
    register_primitive(_p)
