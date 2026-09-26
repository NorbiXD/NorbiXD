"""``darwin`` command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn

from darwin.api.app import create_app
from darwin.attribution.explain import AmbiguousIntent, explain, find_intent
from darwin.config.challenge import ChallengeConfig, load_config
from darwin.core.types import Mode
from darwin.evolution.bench import run_bench
from darwin.evolution.tournament import load_champions, run_tournament
from darwin.market.synthetic import SyntheticMarket
from darwin.persistence.store import AuditStore, RunExistsError, new_run_id
from darwin.replay.recorder import load_events
from darwin.research.proposer import TemplateProposer
from darwin.research.sandbox import SandboxSettings, SpeciesRegistry, evaluate_proposal
from darwin.runtime.app import build_runtime, prepare_config, synthetic_stream
from darwin.runtime.replay import build_replay


def _parse_ts(s: str) -> int:
    if s.isdigit():
        return int(s)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _config(args: argparse.Namespace) -> ChallengeConfig:
    overrides: dict[str, Any] = {"challenge": {}, "sim": {}, "persistence": {}}
    if getattr(args, "mode", None):
        overrides["challenge"]["mode"] = args.mode
    if getattr(args, "hours", None):
        overrides["challenge"]["duration_hours"] = args.hours
    if getattr(args, "speed", None):
        overrides["sim"]["speed"] = args.speed
    if getattr(args, "seed", None) is not None:
        overrides["sim"]["seed"] = args.seed
    if getattr(args, "db", None):
        overrides["persistence"]["database_url"] = args.db
    if getattr(args, "ai", None):
        wanted = {p.strip() for p in args.ai.split(",") if p.strip()}
        unknown = wanted - AI_PROVIDERS.keys()
        if unknown or not wanted:
            choices = ", ".join(sorted(AI_PROVIDERS))
            raise SystemExit(f"--ai: unknown provider(s) {sorted(unknown)}; choose from {choices}")
        # real models only: the mock is switched off so nothing can silently stand in for them
        overrides["intelligence"] = {
            "mock": {"enabled": False},
            **{name: {"enabled": name in wanted} for name in AI_PROVIDERS},
        }
    return load_config(args.config, overrides)


#: provider name -> environment variable holding its API key
AI_PROVIDERS = {"jev": "TYPESAFE_API_KEY", "grok": "XAI_API_KEY"}


def _check_ai(cfg: ChallengeConfig) -> str | None:
    """Explain why the requested real-AI setup cannot run, or None if it can."""
    ic = cfg.intelligence
    enabled = [n for n in AI_PROVIDERS if getattr(ic, n).enabled]
    if not enabled:
        return None
    if cfg.challenge.mode is Mode.REPLAY:
        return "AI providers are not called in deterministic replay; use --mode sim or paper"
    missing = [AI_PROVIDERS[n] for n in enabled if not os.environ.get(AI_PROVIDERS[n])]
    if missing:
        return f"AI provider key(s) not set: {', '.join(missing)} (export them or load .env first)"
    return None


def _print_leaderboard(rows: list[dict[str, Any]], limit: int = 30) -> None:
    hdr = (
        f"{'agent':7} {'species':38} {'status':9} {'gen':>3} {'trades':>6} "
        f"{'shadow%':>8} {'fitness':>8} {'weight':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:limit]:
        star = "*" if r.get("champion") else " "
        fit = f"{r['fitness']:8.3f}" if r.get("fitness") is not None else "       -"
        sr = r.get("shadow_return")
        print(
            f"{r['agent_id']:6}{star} {r['species'][:38]:38} {r['status']:9} {r['generation']:>3} "
            f"{r['trades']:>6} {(sr or 0) * 100:8.2f} {fit} {r.get('weight', 0):6.2f}"
        )


# --------------------------------------------------------------------------- commands


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


async def _serve_api(server: uvicorn.Server) -> None:
    """The dashboard/API is observability: its failure must never take trading down."""
    try:
        await server.serve()
    except (SystemExit, OSError) as e:
        logging.getLogger(__name__).error("API server stopped: %r (trading continues)", e)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    problem = _check_ai(cfg)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    cfg = asyncio.run(prepare_config(cfg))
    if not args.no_api and not _port_free(cfg.api.host, cfg.api.port):
        print(f"API port {cfg.api.host}:{cfg.api.port} is busy; free it or pass --no-api", file=sys.stderr)
        return 2
    run_id = args.run_id or new_run_id(cfg.challenge.mode.value)
    store = AuditStore(cfg.persistence.database_url, run_id=run_id)
    seeds = load_champions(args.seed_genomes, min_holdout_fitness=0.0) if args.seed_genomes else None
    if seeds is not None:
        print(f"seeding {len(seeds)} champion genomes from {args.seed_genomes}")
    promoted = SpeciesRegistry(args.species_dir).load(register=True)
    rt = build_runtime(cfg, store, run_id, seed_genomes=seeds)
    for prim, genome in promoted:
        if genome is not None:
            a = rt.engine.inject_genome(genome, origin=f"sandbox:{prim}")
            print(f"injected promoted species {prim} as challenger {a.agent_id}")
    logging.getLogger(__name__).info(
        "run %s mode=%s db=%s", run_id, cfg.challenge.mode.value, cfg.persistence.database_url
    )

    async def main() -> float:
        server: uvicorn.Server | None = None
        server_task: asyncio.Task[None] | None = None
        if not args.no_api:
            app = create_app(rt.engine, rt.webhooks)
            server = uvicorn.Server(
                uvicorn.Config(app, host=cfg.api.host, port=cfg.api.port, log_level="warning")
            )
            server_task = asyncio.create_task(_serve_api(server))
            print(f"dashboard: http://{cfg.api.host}:{cfg.api.port}/   api docs: /api/docs")
        try:
            final = await rt.run()
        finally:
            store.close()
        if server is not None and server_task is not None:
            if args.keep_api:
                print("challenge finished; API still serving (Ctrl-C to exit)")
            else:
                server.should_exit = True  # graceful shutdown (no lifespan cancellation noise)
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
        return final

    final = asyncio.run(main())
    st = rt.engine.state()
    print(
        f"\nrun {run_id} finished: equity {final:.2f} (start {cfg.challenge.starting_capital:.2f}), "
        f"generations {st['generation']}, alive {st['alive']}, dead {st['dead']}"
    )
    _print_leaderboard(rt.engine.leaderboard())
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    args.mode = Mode.REPLAY.value
    cfg = _config(args)
    run_id = args.run_id or new_run_id("replay")
    store = AuditStore(cfg.persistence.database_url, run_id=run_id) if not args.no_db else None
    t0 = 1_700_000_000_000
    h = build_replay(
        cfg,
        synthetic_stream(cfg, t0, planted=not args.null_market),
        t0,
        store=store,
        run_id=run_id,
        intelligence=cfg.intelligence.mock.enabled,
    )
    started = time.time()
    final = h.driver.run()
    if store is not None:
        store.close()
    st = h.engine.state()
    print(f"run {run_id}: {h.driver.events} events in {time.time() - started:.1f}s")
    print(
        f"final equity {final:.2f} (start {cfg.challenge.starting_capital:.2f}) "
        f"| generations {st['generation']} | alive {st['alive']} dead {st['dead']} "
        f"| champion {st['champion']}"
    )
    print(f"stats: {json.dumps(st['stats'])}")
    _print_leaderboard(h.engine.leaderboard())
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    args.mode = Mode.REPLAY.value
    cfg = _config(args)
    t0 = 1_700_000_000_000
    train_ms, hold_ms = int(args.train_hours * 3_600_000), int(args.holdout_hours * 3_600_000)
    m = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=t0,
        duration_ms=train_ms + hold_ms,
        seed=cfg.sim.seed,
        step_ms=cfg.sim.synthetic_step_ms,
        book_every=cfg.sim.synthetic_book_every,
        planted_edges=not args.null_market,
    )
    events = list(load_events(args.data) if args.data else m.events())
    if args.data:
        t0 = events[0].ts
    res = run_tournament(
        cfg,
        events,
        t0,
        train_ms,
        hold_ms,
        top_k=args.top,
        population_size=args.population,
        generation_bars=args.generation_bars,
    )
    res.save(args.out)
    print(
        f"train: {res.train_generations} generations, {res.train_population_born} genomes born; "
        f"{len(res.champions)} selected; train->holdout fitness degradation {res.degradation():+.3f} "
        f"({res.elapsed_s:.0f}s)"
    )
    print(f"{'genome':14} {'species':38} {'train fit':>9} {'hold fit':>9} {'hold ret':>9} {'trades':>6}")
    for c in res.champions:
        print(
            f"{c.genome_id:14} {c.species[:38]:38} {c.train.get('fitness', float('nan')):9.3f} "
            f"{c.holdout['fitness']:9.3f} {c.holdout['net_return'] * 100:8.2f}% {int(c.holdout['trades']):6}"
        )
    print(f"saved {args.out}  (use: darwin run --seed-genomes {args.out})")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    args.mode = Mode.REPLAY.value
    cfg = _config(args)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    rows, summaries = run_bench(cfg, seeds, args.hours, planted=not args.null_market)
    print(
        f"{'allocator':18} {'runs':>4} {'mean final':>11} {'median':>9} {'mean logG':>10} {'mean maxDD':>10} "
        f"{'Δ vs equal':>11} {'t':>6} {'wins':>5}"
    )
    for s in summaries:
        print(
            f"{s.allocator:18} {s.runs:4} {s.mean_final:11.2f} {s.median_final:9.2f} "
            f"{s.mean_log_growth:10.4f} {s.mean_max_drawdown:10.3f} {s.paired_vs_baseline_mean:+11.4f} "
            f"{s.paired_vs_baseline_t:6.2f} {s.wins_vs_baseline:5}"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"rows": [r.__dict__ for r in rows], "summary": [s.__dict__ for s in summaries]}, indent=2
            )
        )
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    registry = SpeciesRegistry(args.registry)
    proposals = TemplateProposer(seed=args.seed or 0).propose(args.proposals)
    settings = SandboxSettings(
        train_hours=args.train_hours,
        holdout_hours=args.holdout_hours,
        genomes=args.genomes,
        seed=args.seed or 7,
    )
    for prop in proposals:
        print(f"\n{prop.proposal_id} [{prop.proposer}] {prop.rationale}")
        rep = evaluate_proposal(prop, settings)
        for st in rep.stages:
            print(
                f"  {'PASS' if st.passed else 'FAIL'} {st.stage:10} "
                f"{json.dumps(st.detail, default=str)[:160]}"
            )
        if rep.error:
            print(f"  error: {rep.error[:300]}")
        if rep.passed:
            path = registry.save(prop, rep)
            print(f"  PROMOTED -> {path}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    cfg = _config(args)
    store = AuditStore(cfg.persistence.database_url, run_id=args.run or "?")
    iid = args.intent
    if iid is None:
        if not (args.agent and args.at):
            print("need --intent, or --agent and --at", file=sys.stderr)
            return 2
        iid = find_intent(store, args.agent, args.symbol, _parse_ts(args.at), run_id=args.run)
        if iid is None:
            print("no decision found", file=sys.stderr)
            return 1
    try:
        ex = explain(store, iid, run_id=args.run)
    except AmbiguousIntent as e:
        print(str(e), file=sys.stderr)
        return 2
    if ex is None:
        print("unknown intent", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(ex, indent=2, default=str))
        return 0
    i = ex["intent"]
    print(ex["question"].replace(str(i["ts"]), _fmt_ts(i["ts"])))
    print(
        f"  intent {i['intent_id']}  reason={i['reason']}  target={i['target_exposure']:+.3f}x  "
        f"confidence={i['confidence']:.2f}  regime={i['regime']}"
    )
    print(f"  genome {i['genome_id']}  species={ex['genome']['species'] if ex['genome'] else '?'}")
    print("  what it saw:")
    for k, v in sorted(ex["knew"]["features_seen"].items()):
        print(f"    {k:28} {v: .6g}")
    print(f"  component scores: {ex['knew']['components']}")
    if ex["knew"]["signals_seen"]:
        print(f"  intelligence signals seen: {ex['knew']['signals_seen']}")
    print("  lineage:")
    for lr in ex["lineage"]:
        print(
            f"    {_fmt_ts(lr['ts'])} gen {lr['generation']:>3} {lr['kind']:10} parents={lr['parent_ids']} "
            f"{json.dumps(lr['details'], default=str)[:120]}"
        )
    print("  risk decisions:")
    for d in ex["risk_decisions"]:
        print(
            f"    {d['account_id']:16} approved={d['approved']} order_qty={d['order_qty']:+.6g} "
            f"reasons={d['reasons']}"
        )
    for f in ex["fills"]:
        print(
            f"  fill {f['exec_id']} {f['side']} {f['qty']:.6g} @ {f['price']:.6g} fee {f['fee']:.4f} "
            f"slippage {f['slippage_bps'] or 0:.2f}bps"
        )
    for t in ex["trades"]:
        print(
            f"  trade {t['trade_id']} [{t['account_id']}] pnl {t['net_pnl']:+.4f} MFE {t['mfe_pct']:+.2%} "
            f"MAE {t['mae_pct']:+.2%} exit={t['exit_reason']}"
        )
    if ex["fitness_contribution"]:
        print(f"  fitness contribution: {ex['fitness_contribution']}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    cfg = _config(args)
    store = AuditStore(cfg.persistence.database_url, run_id="?")
    for r in sorted(store.query("runs"), key=lambda r: r["started_ts"] or 0):
        fe = r["final_equity"]
        print(
            f"{r['run_id']:28} {r['mode']:8} {r['status']:9} start={r['starting_capital']} "
            f"final={fe if fe is None else round(fe, 2)} risk={r['risk_fingerprint']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="darwin", description="DARWIN self-evolving trading challenge")
    p.add_argument("--config", default="challenge.yaml")
    p.add_argument("--db", help="database URL (overrides config / DARWIN_DATABASE_URL)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the challenge (mode from config or --mode)")
    r.add_argument("--mode", choices=[m.value for m in Mode])
    r.add_argument("--hours", type=float)
    r.add_argument("--speed", type=float, help="sim mode acceleration")
    r.add_argument("--seed", type=int)
    r.add_argument("--run-id")
    r.add_argument("--no-api", action="store_true")
    r.add_argument("--keep-api", action="store_true", help="keep serving the dashboard after the end")
    r.add_argument("--seed-genomes", help="champion set JSON from `darwin evolve`")
    r.add_argument("--species-dir", default="./data/species", help="promoted Level-2 species to inject")
    r.add_argument(
        "--ai",
        nargs="?",
        const="jev,grok",
        metavar="PROVIDERS",
        help="use real AI providers (default: jev,grok) instead of the mock; needs TYPESAFE_API_KEY / "
        "XAI_API_KEY and refuses to start without them",
    )
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("replay", help="fast deterministic replay on synthetic data")
    rp.add_argument("--hours", type=float, default=24)
    rp.add_argument("--seed", type=int)
    rp.add_argument("--run-id")
    rp.add_argument("--null-market", action="store_true", help="random walk without planted structure")
    rp.add_argument("--no-db", action="store_true")
    rp.set_defaults(fn=cmd_replay)

    rs2 = sub.add_parser("research", help="Level-2: propose new species, sandbox-validate, promote")
    rs2.add_argument("--proposals", type=int, default=2)
    rs2.add_argument("--train-hours", type=float, default=36)
    rs2.add_argument("--holdout-hours", type=float, default=12)
    rs2.add_argument("--genomes", type=int, default=6)
    rs2.add_argument("--seed", type=int)
    rs2.add_argument("--registry", default="./data/species")
    rs2.set_defaults(fn=cmd_research)

    ev = sub.add_parser("evolve", help="offline accelerated evolution: train -> holdout -> champion set")
    ev.add_argument("--train-hours", type=float, default=72)
    ev.add_argument("--holdout-hours", type=float, default=24)
    ev.add_argument("--population", type=int, default=48)
    ev.add_argument("--generation-bars", type=int, default=120)
    ev.add_argument("--top", type=int, default=8)
    ev.add_argument("--seed", type=int)
    ev.add_argument("--data", help="recorded parquet run directory instead of synthetic data")
    ev.add_argument("--null-market", action="store_true")
    ev.add_argument("--out", default="champions.json")
    ev.set_defaults(fn=cmd_evolve)

    b = sub.add_parser("bench-allocators", help="paired benchmark: equal vs fitness-weighted vs Thompson")
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--first-seed", type=int, default=100)
    b.add_argument("--hours", type=float, default=48)
    b.add_argument("--null-market", action="store_true")
    b.add_argument("--json", help="write rows + summary to this file")
    b.set_defaults(fn=cmd_bench)

    e = sub.add_parser("explain", help="why did an agent take a decision?")
    e.add_argument("--run")
    e.add_argument("--intent")
    e.add_argument("--agent")
    e.add_argument("--symbol")
    e.add_argument("--at", help="ISO time or epoch ms")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_explain)

    rs = sub.add_parser("runs", help="list recorded runs")
    rs.set_defaults(fn=cmd_runs)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        rc: int = args.fn(args)
    except RunExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
