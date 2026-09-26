"""Attribution analytics: learn *where* intelligence works, not just whether PnL is positive.

* :func:`species_regime_performance` — closed shadow trades by (species, regime at entry):
  e.g. "breakout only makes money in trend regimes".
* :func:`signal_information` — for every recorded intelligence signal (by provider/topic), the
  rank information coefficient between its value and the *subsequent* return over several
  horizons, split by regime: e.g. "source X is informative only during high volatility".

These are hindsight *analyses* over persisted data (they look forward by design); they never feed
back into live decisions except through evolution acting on realised agent performance.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from typing import Any

import numpy as np

from darwin.persistence.store import AuditStore


def _rank_ic(x: list[float], y: list[float]) -> float | None:
    if len(x) < 8:
        return None
    a, b = np.asarray(x), np.asarray(y)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def species_regime_performance(
    store: AuditStore, run_id: str, account_prefix: str = "shadow:"
) -> list[dict[str, Any]]:
    agents = {a["agent_id"]: a for a in store.query("agents", run_id=run_id)}
    intents = {i["intent_id"]: i for i in store.query("intents", run_id=run_id)}
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for t in store.query("trades", run_id=run_id):
        if not str(t["account_id"]).startswith(account_prefix):
            continue
        entry = intents.get(t["entry_intent_id"] or "")
        regime = entry["regime"] if entry else "unknown"
        species = agents.get(t["agent_id"], {}).get("species", "?")
        groups[(species, regime)].append(float(t["return_on_equity"]))
    out = []
    for (species, regime), rets in sorted(groups.items()):
        arr = np.asarray(rets)
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(
            {
                "species": species,
                "regime": regime,
                "trades": int(arr.size),
                "win_rate": float((arr > 0).mean()),
                "mean_return": float(arr.mean()),
                "t_stat": float(arr.mean() / (sd / math.sqrt(arr.size))) if sd > 0 else 0.0,
            }
        )
    return out


def signal_information(
    store: AuditStore, run_id: str, horizons_bars: tuple[int, ...] = (5, 30, 60)
) -> list[dict[str, Any]]:
    # per-symbol bar closes and regimes from persisted snapshots
    closes: dict[str, tuple[list[int], list[float], list[str]]] = {}
    for s in sorted(store.query("snapshots", run_id=run_id), key=lambda r: (r["symbol"], r["ts"])):
        ts_l, px_l, rg_l = closes.setdefault(s["symbol"], ([], [], []))
        ts_l.append(int(s["ts"]))
        px_l.append(float(s["data"]["bar"]["close"]))
        rg_l.append(str(s["data"].get("regime", "unknown")))
    acc: dict[tuple[str, str, str, int], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for sig in store.query("signals", run_id=run_id):
        symbols = [sig["symbol"]] if sig["symbol"] else list(closes)
        for sym in symbols:
            if sym not in closes:
                continue
            ts_l, px_l, rg_l = closes[sym]
            i = bisect.bisect_left(ts_l, int(sig["ts"]))  # first bar closing at/after availability
            if i >= len(ts_l):
                continue
            for h in horizons_bars:
                j = i + h
                if j >= len(ts_l):
                    continue
                fwd = math.log(px_l[j] / px_l[i])
                for regime in (rg_l[i], "all"):
                    xs, ys = acc[(str(sig["provider"]), str(sig["topic"]), regime, h)]
                    xs.append(float(sig["value"]))
                    ys.append(fwd)
    out = []
    for (provider, topic, regime, h), (xs, ys) in sorted(acc.items()):
        hits = [np.sign(x) == np.sign(y) for x, y in zip(xs, ys, strict=True) if x != 0 and y != 0]
        out.append(
            {
                "provider": provider,
                "topic": topic,
                "regime": regime,
                "horizon_bars": h,
                "n": len(xs),
                "rank_ic": _rank_ic(xs, ys),
                "hit_rate": float(np.mean(hits)) if hits else None,
            }
        )
    return out
