"""``darwin`` command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

import uvicorn

from darwin.api.app import create_app
from darwin.attribution.explain import explain, find_intent
from darwin.config.challenge import ChallengeConfig, load_config
from darwin.core.types import Mode
from darwin.persistence.store import AuditStore
from darwin.runtime.app import build_runtime, synthetic_stream
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
    return load_config(args.config, overrides)


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


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    run_id = args.run_id or f"{cfg.challenge.mode.value}-{int(time.time())}"
    store = AuditStore(cfg.persistence.database_url, run_id=run_id)
    rt = build_runtime(cfg, store, run_id)
    logging.getLogger(__name__).info(
        "run %s mode=%s db=%s", run_id, cfg.challenge.mode.value, cfg.persistence.database_url
    )

    async def main() -> float:
        server: uvicorn.Server | None = None
        server_task: asyncio.Task[None] | None = None
        if not args.no_api:
            app = create_app(rt.engine)
            server = uvicorn.Server(
                uvicorn.Config(app, host=cfg.api.host, port=cfg.api.port, log_level="warning")
            )
            server_task = asyncio.create_task(server.serve())
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
    run_id = args.run_id or f"replay-{int(time.time())}"
    store = AuditStore(cfg.persistence.database_url, run_id=run_id) if not args.no_db else None
    t0 = 1_700_000_000_000
    h = build_replay(
        cfg, synthetic_stream(cfg, t0, planted=not args.null_market), t0, store=store, run_id=run_id
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
    ex = explain(store, iid)
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
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("replay", help="fast deterministic replay on synthetic data")
    rp.add_argument("--hours", type=float, default=24)
    rp.add_argument("--seed", type=int)
    rp.add_argument("--run-id")
    rp.add_argument("--null-market", action="store_true", help="random walk without planted structure")
    rp.add_argument("--no-db", action="store_true")
    rp.set_defaults(fn=cmd_replay)

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
    rc: int = args.fn(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
