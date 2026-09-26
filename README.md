# DARWIN

**A self-evolving population of AI trading agents competing for $200 of crypto-futures capital.**

Agents receive identical market state, decide independently, trade through one immutable risk
gateway, get ranked on evidence of edge, and then die, mutate, cross over and get replaced. A
capital allocator decides which forms of intelligence deserve real money right now. Every
decision can be traced back to exactly what the agent saw and what happened next.

```
MARKET → AGENTS → TRADE INTENTS → RISK GOVERNOR → EXECUTION → FILLS → LEDGER
       → ATTRIBUTION → FITNESS → SELECTION → DEATH / MUTATION / CROSSOVER
       → NEW GENERATION → CAPITAL REALLOCATION
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [progress.md](progress.md) for the
current status.

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,postgres]"

# 1. Deterministic replay on the synthetic market (24h in a few seconds)
.venv/bin/darwin replay --hours 24

# 2. Watch it live: synthetic market at 120x (1 market-hour per 30s) + dashboard/API
.venv/bin/darwin run --mode sim --hours 48 --speed 120 --keep-api
#   → http://127.0.0.1:8000/  ·  API docs: http://127.0.0.1:8000/api/docs

# 3. "Why did agent A0012 go short SOL at 14:03?"
.venv/bin/darwin explain --run <run_id> --agent A0012 --symbol SOLUSDT --at 2026-09-26T14:03:00Z

# 4. Paper trading on Bybit's real public streams (no keys needed). Starts warm: the last 800
#    closed 1m candles are preloaded, so indicators and the AI fast path work immediately.
.venv/bin/darwin run --mode paper

# 4b. Paper trading driven by the real AI providers instead of the mock
#     (TypeSafe System One fast path + Grok/X Search slow path; refuses to start without the keys)
export TYPESAFE_API_KEY=...  XAI_API_KEY=...
.venv/bin/darwin run --mode paper --ai          # or --ai grok / --ai jev

# 5. Testnet (needs BYBIT_TESTNET_API_KEY / _SECRET; refuses a non-flat account)
.venv/bin/darwin run --mode testnet

# 6. Offline accelerated evolution -> champion set -> seed a live population
.venv/bin/darwin evolve --train-hours 72 --holdout-hours 24 --out champions.json
.venv/bin/darwin run --mode sim --seed-genomes champions.json

# 7. Allocator benchmark (paired, identical populations per seed)
.venv/bin/darwin bench-allocators --seeds 6 --hours 48

# 8. Level-2: propose new species, sandbox-validate, promote (injected on the next run)
.venv/bin/darwin research --proposals 2
```

Docker (PostgreSQL + DARWIN in sim mode):

```bash
cp .env.example .env   # set api.host: 0.0.0.0 in challenge.yaml for container access
docker compose up --build
```

## Safety

* **Live trading is off by default** and needs three independent switches:
  `challenge.live_enabled: true`, `DARWIN_LIVE_CONFIRM=I_UNDERSTAND_THIS_TRADES_REAL_MONEY`, and
  `BYBIT_API_KEY/SECRET`.
* Risk limits live in `challenge.yaml`, are frozen at start-up, are fingerprinted into every run
  record, and are not reachable from agents, evolution or LLM providers.
* `touch KILL` (or `POST /api/kill-switch` with `DARWIN_OPERATOR_TOKEN`) halts new risk,
  cancels working entries and flattens the challenge account, **retrying every heartbeat until
  ledger, venue and orders are all clear**: rejects, partial fills, lost orders and lost fills
  are retried or recovered, and positions only the exchange knows about are closed with
  `reduceOnly`. Breakers and the end of the challenge flatten the same way; after the end a live
  run keeps trying for `end_flatten_timeout_s` (30 min) and alerts every minute. **Ctrl-C (or
  SIGTERM) ends the challenge and flattens first**; press it again to exit immediately.
  DARWIN only ever trades, books and flattens the challenge symbols: positions in other symbols
  on the same account are ignored and reported (live refuses to start with them). There are no
  exchange-side stop orders yet, so an exchange that stays unreachable (or a killed process)
  cannot be flattened by DARWIN — that is the remaining gap. Risk limits never block an exit.
* Machine-generated species run in-process only after passing a static allowlist DSL and a
  credential-free sandbox; a species that raises is quarantined, not the trading loop.
* Tests never need credentials or network.

## Development

```bash
.venv/bin/pytest            # look-ahead perturbation, e2e evolution, multi-seed known-answer science,
                            # chaos + flatten failure injection, testnet wiring against a fake Bybit,
                            # DSL bomb tests, governor property tests (-m "not slow" skips ~3 min)
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
.venv/bin/mypy              # strict
```

## Repository map

| Path | What |
|---|---|
| `src/darwin/core` | event schemas, TradeIntent contract, ids, enums |
| `src/darwin/config` | typed `challenge.yaml` (frozen `RiskLimits`) |
| `src/darwin/market` | order book, market state, bars, synthetic market |
| `src/darwin/features` | incremental feature engine / `FeatureView` |
| `src/darwin/agents` | primitives, genome DSL, mutation/crossover, agent runtime |
| `src/darwin/risk` | Risk Governor |
| `src/darwin/execution` | order state machine, execution engine |
| `src/darwin/exchange/sim` | simulated venue (latency, book walking, fees, funding, liquidation, chaos) |
| `src/darwin/exchange/bybit` | Bybit V5 parser, WebSocket client, REST, execution gateway, feeds |
| `src/darwin/portfolio` | ledger, agent sub-positions, round trips (MFE/MAE) |
| `src/darwin/evolution` | fitness, population/selection/lineage, allocators, tournament, benchmark |
| `src/darwin/runtime` | engine, replay driver, live driver, mode wiring |
| `src/darwin/persistence` | SQL schema + buffered audit store |
| `src/darwin/attribution` | decision reconstruction ("explain") |
| `src/darwin/api` | FastAPI observability/control API |
| `src/darwin/dashboard` | single-file live dashboard served at `/` |
| `src/darwin/intelligence` | Jev / Grok / mock providers, fast+slow path service |
| `src/darwin/signals` | point-in-time signal board, external (webhook) feeds |
| `src/darwin/replay` | Parquet recorder, DuckDB loader, Bybit trade-dump loader |
| `src/darwin/research` | Level-2 species DSL, sandbox pipeline, proposers, registry |

**Disclaimer:** experimental research software. The synthetic market validates the machinery and
says nothing about real-market profitability. Don't trade money you can't lose.
