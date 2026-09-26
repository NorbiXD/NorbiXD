"""Sandbox worker process: ``python -m darwin.research.worker`` (JSON in on stdin, JSON out).

Runs with a scrubbed environment and rlimits set by the parent (see ``sandbox.py``). It never
opens a database or a network connection: market data is generated in-process.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

from darwin.agents.genome import ExecutionGenes, GeneTerm, Genome, RiskGenes
from darwin.agents.primitives import PRIMITIVES, register_primitive
from darwin.config.challenge import load_config
from darwin.evolution.fitness import compute_fitness
from darwin.features.engine import FeatureEngine
from darwin.market.bars import BarBuilder
from darwin.market.state import MarketState
from darwin.market.synthetic import SyntheticMarket
from darwin.research.dsl import DSLError, validate
from darwin.research.sandbox import SECRET_MARKERS, make_primitive
from darwin.runtime.replay import build_replay


def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def main() -> int:
    req = json.loads(sys.stdin.read())
    st = req["settings"]
    stages: list[dict[str, Any]] = []
    out: dict[str, Any] = {"stages": stages}

    def stage(name: str, passed: bool, **detail: Any) -> bool:
        stages.append({"stage": name, "passed": passed, "detail": detail})
        return passed

    leaked = sorted(k for k in os.environ if any(m in k.upper() for m in SECRET_MARKERS))
    try:
        validate(req["source"])
    except DSLError as e:
        stage("static", False, error=str(e))
        _emit(out)
        return 0
    stage("static", True, env_secrets_visible=leaked)
    if leaked:
        stages[-1]["passed"] = False
        _emit(out)
        return 0
    name = req["primitive"]
    prim = make_primitive(name, req["source"], "sandbox:candidate")
    register_primitive(prim, replace=True)

    # ---------------------------------------------------------------- unit tests on real views
    t0 = 1_700_000_000_000
    syms = ("BTCUSDT", "ETHUSDT")
    mkt = SyntheticMarket(
        symbols=syms, start_ts=t0, duration_ms=12 * 3_600_000, step_ms=5_000, seed=st["seed"] + 1
    )
    ms, fe = MarketState(syms), FeatureEngine(syms)
    builders = {s: BarBuilder(s) for s in syms}
    next_bar = (t0 // 60_000 + 1) * 60_000
    views = []
    for ev in mkt.events():
        while ev.ts >= next_bar:
            for s in syms:
                bar = builders[s].close_bar(next_bar - 60_000, next_bar, ms[s], stale=False)
                if bar:
                    fe.add_bar(bar)
            if (next_bar // 60_000) % 7 == 0:
                views.append(fe.view("BTCUSDT", next_bar))
            next_bar += 60_000
        ms.apply(ev)
        if ev.kind == "trade":
            builders[ev.symbol].on_trade(ev)
    rng = np.random.default_rng(st["seed"])
    param_sets = [{k: s.sample(rng) for k, s in prim.params.items()} for _ in range(8)]
    bad: list[str] = []
    nonzero, calls = 0, 0
    t_start = time.perf_counter()
    first: list[float] = []
    for p in param_sets:
        for v in views:
            raw = prim.fn(v.fresh(), p)
            calls += 1
            ok = isinstance(raw, (int, float)) and math.isfinite(float(raw)) and -1.0 <= float(raw) <= 1.0
            if not ok and len(bad) < 5:
                bad.append(repr(raw)[:60])
            s_ = prim.score(v.fresh(), p)
            first.append(s_)
            nonzero += s_ != 0.0
    per_call_ms = (time.perf_counter() - t_start) * 1000 / max(calls, 1)
    again = [prim.score(v.fresh(), p) for p in param_sets for v in views]
    reverse = [prim.score(v.fresh(), p) for p in reversed(param_sets) for v in reversed(views)][::-1]
    unit_ok = (
        not bad and first == again and first == reverse and nonzero >= 0.01 * calls and per_call_ms < 2.0
    )
    if not stage(
        "unit",
        unit_ok,
        views=len(views),
        calls=calls,
        invalid_outputs=bad,
        nonzero_frac=nonzero / max(calls, 1),
        deterministic=first == again,
        order_independent=first == reverse,
        per_call_ms=round(per_call_ms, 4),
    ):
        _emit(out)
        return 0

    # ---------------------------------------------------------------- leakage: future bars must not matter
    cut = views[len(views) // 2].now
    fe_past = FeatureEngine(syms)
    for b in fe.history["BTCUSDT"]:
        if b.end_ts <= cut:
            fe_past.add_bar(b)
    v_full, v_past = fe.view("BTCUSDT", cut), fe_past.view("BTCUSDT", cut)
    diffs = [abs(prim.score(v_full.fresh(), p) - prim.score(v_past.fresh(), p)) for p in param_sets]
    if not stage("leakage", max(diffs) == 0.0, max_abs_diff=max(diffs)):
        _emit(out)
        return 0

    # ---------------------------------------------------------------- train / holdout replays
    train_ms = int(st["train_hours"] * 3_600_000)
    hold_ms = int(st["holdout_hours"] * 3_600_000)
    warm_ms = 4 * 3_600_000
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    events = list(
        SyntheticMarket(
            symbols=symbols,
            start_ts=t0,
            duration_ms=train_ms + hold_ms + 120_000,
            step_ms=5_000,
            seed=st["seed"],
        ).events()
    )

    def genome(params: dict[str, float], r: np.random.Generator) -> Genome:
        return Genome(
            terms=(GeneTerm(primitive=name, params=params, weight=1.0),),
            symbols=symbols,
            entry_threshold=float(r.uniform(0.2, 0.8)),
            exit_threshold=float(r.uniform(-0.3, 0.3)),
            risk=RiskGenes(
                exposure=1.0,
                stop_loss_pct=float(r.uniform(0.01, 0.05)),
                take_profit_pct=float(r.uniform(0.02, 0.15)),
                max_hold_bars=int(r.integers(60, 1440)),
                confidence_scaling=False,
            ),
            execution=ExecutionGenes(),
        )

    candidates = [
        genome({k: s.sample(rng) for k, s in prim.params.items()}, rng) for _ in range(st["genomes"])
    ]

    def run(genomes: list[Genome], start: int, end: int, measure_from: int) -> dict[str, dict[str, float]]:
        cfg = load_config(
            None,
            {
                "challenge": {"duration_hours": (end - start) / 3_600_000, "symbols": list(symbols)},
                "risk": {"kill_switch_file": None},
                "evolution": {"population_size": max(4, len(genomes)), "generation_bars": 100_000},
            },
        )
        h = build_replay(
            cfg,
            (e for e in events if start <= e.ts < end),
            start,
            seed_genomes=[(g, "sandbox") for g in genomes],
            fill=False,
        )
        h.driver.run()
        res = {}
        for a in h.engine.population.agents.values():
            book = h.engine.population.books[a.agent_id]
            ts = np.asarray(book.ts, dtype=np.int64)
            eq = np.asarray(book.equity, dtype=float)
            m = ts >= measure_from
            trades = [t for t in book.trades if t.entry_ts >= measure_from]
            rep = compute_fitness(
                eq[m], trades, end - measure_from, cfg.evolution.fitness, 7.0, seed_key=a.agent_id
            )
            res[a.genome.genome_id] = {
                "fitness": rep.fitness,
                "net_return": rep.net_return,
                "trades": rep.n_trades,
                "ruin_prob": rep.ruin_prob,
            }
        return res

    train = run(candidates, t0, t0 + train_ms, t0)
    best = max(candidates, key=lambda g: train[g.genome_id]["fitness"])
    stage("train", train[best.genome_id]["fitness"] > 0, best=train[best.genome_id], all=list(train.values()))
    champion = (
        Genome.model_validate(req["champion"])
        if req.get("champion")
        else Genome(
            terms=(GeneTerm(primitive="momentum", params={"lookback": 60, "scale": 1.0}, weight=1.0),),
            symbols=symbols,
            entry_threshold=0.5,
            exit_threshold=0.0,
            risk=RiskGenes(
                exposure=1.0,
                stop_loss_pct=0.03,
                take_profit_pct=0.1,
                max_hold_bars=720,
                confidence_scaling=False,
            ),
        )
    )
    if any(t.primitive not in PRIMITIVES for t in champion.terms):
        champion = best
    hs = t0 + train_ms
    hold = run(
        [best, champion] if champion.genome_id != best.genome_id else [best], hs - warm_ms, hs + hold_ms, hs
    )
    hb, hc = hold[best.genome_id], hold.get(champion.genome_id, hold[best.genome_id])
    stage("holdout", hb["fitness"] > 0 and hb["trades"] >= st["min_holdout_trades"], best=hb)
    stage("challenger", hb["fitness"] >= hc["fitness"] - st["challenger_margin"], candidate=hb, champion=hc)
    out["best_genome"] = best.model_dump(mode="json")
    _emit(out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MemoryError:
        _emit({"stages": [], "error": "memory limit exceeded"})
        raise SystemExit(0) from None
