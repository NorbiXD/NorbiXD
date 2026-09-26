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

# 4. Paper trading on Bybit's real public streams (no keys needed)
.venv/bin/darwin run --mode paper

# 5. Testnet (needs BYBIT_TESTNET_API_KEY / _SECRET)
.venv/bin/darwin run --mode testnet
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
* `touch KILL` (or `POST /api/kill-switch` with `DARWIN_OPERATOR_TOKEN`) halts new risk and
  flattens the challenge account. Getting flat is never blocked by the safety system.
* Tests never need credentials or network.

## Development

```bash
.venv/bin/pytest            # ~110 tests, incl. look-ahead perturbation and end-to-end evolution
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
| `src/darwin/evolution` | fitness, population/selection/lineage, capital allocators |
| `src/darwin/runtime` | engine, replay driver, live driver, mode wiring |
| `src/darwin/persistence` | SQL schema + buffered audit store |
| `src/darwin/attribution` | decision reconstruction ("explain") |
| `src/darwin/api` | FastAPI observability/control API |

**Disclaimer:** experimental research software. The synthetic market validates the machinery and
says nothing about real-market profitability. Don't trade money you can't lose.
