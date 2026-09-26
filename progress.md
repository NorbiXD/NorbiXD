# progress.md — resume here

_Last updated: 2026-09-26 (session 1)._ Branch: `claude/clever-tesla-kk8pm0`.

## Environment facts
* Python 3.12 venv at `.venv` (`uv venv --python 3.12 .venv && uv pip install -e ".[dev,postgres]"`).
* This cloud sandbox's network policy **blocks Bybit, xAI and TypeSafe hosts**: integrations are
  implemented to the documented protocols and tested against local fakes (fake WS server, httpx
  MockTransport). Nothing has been exercised against the real endpoints yet.
* Gates: `.venv/bin/pytest` · `.venv/bin/ruff check src tests` · `.venv/bin/mypy` (strict). All green at last commit.

## Milestone 1 — vertical slice: DONE (pending QM approval)
Bybit stream → normalized events → features → competing agents → TradeIntent → Risk Governor →
simulated/testnet execution → fills → ledger → fitness → kill/promote/mutate → new generation →
leaderboard (CLI + API). Proven by `tests/test_e2e_loop.py` (DoD assertions), `tests/test_paper_mode.py`
(Bybit protocol → full pipeline), `tests/test_lookahead.py` (future-perturbation), and unit suites.

Key decisions (details in ARCHITECTURE.md): shadow evaluation book per agent + one attributed
challenge book; target-exposure intents; bars close at their boundary before same-ts events;
CRRA + empirical-Bayes fitness; strikes/probation before death; frozen RiskLimits.

## QM review log
* Iteration 1: pending.

## Next (Milestone 2), in priority order
1. Dashboard (`src/darwin/dashboard/index.html`, served at `/`, fed by `/api/stream`).
2. Intelligence layer: `DecisionProvider` (Jev via `POST /v1/systemone`, models from `GET /v1/models`,
   question types noul/choice/score) and `NarrativeProvider` (Grok via xAI Responses API with
   `x_search` tool); async slow path emitting `IntelligenceSignal`s with availability timestamps;
   mock providers; provider benchmarking harness. All feature-gated off.
3. `ExternalSignalFeed` interface + authenticated webhook feed (FastAPI route) — Telegram etc. disabled unless ToS permit.
4. Parquet recorder (live events → partitioned parquet) + DuckDB loader for replay; Bybit public
   trade-dump CSV loader.
5. Offline accelerated evolution (`darwin evolve`): large populations on replay → champion set → seed live population.
6. Allocator benchmark (`darwin bench-allocators`): equal vs fitness-weighted vs Thompson across seeds, report honestly.
7. Level-2 meta-research sandbox: proposal → AST/import checks → subprocess with scrubbed env/rlimits →
   unit/replay/holdout/leakage checks → challenger → `Population.inject`.
8. Null-market false-discovery test (evolution on random walk must not show significant OOS edge).

## Known limitations / open questions
* Bybit `orderbook` delta `u` contiguity is assumed (strict mode); verify on a live connection
  (`OrderBook.strict_sequence`).
* Instrument specs: fallbacks in `config/challenge.py`; `runtime.app.refresh_instruments()` pulls
  real filters (call before live runs; not yet wired into `darwin run`).
* No internal crossing/netting of opposing agent orders in the challenge book (both pay fees).
* `ExecutionEngine.orders` / `_seen_exec` grow unbounded (fine for a 7-day challenge; prune later).
* Sim venue passive fills are conservative (trade-through only; no queue model).
