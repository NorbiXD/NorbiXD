# progress.md — resume here

_Last updated: 2026-09-26 (session 1)._ Branch: `claude/clever-tesla-kk8pm0` (local commits; the
GitHub push was refused with 403 — the Claude GitHub App lacks access to `NorbiXD/NorbiXD`).

## Environment facts
* Python 3.12 venv at `.venv` (`uv venv --python 3.12 .venv && uv pip install -e ".[dev,postgres]"`).
* This sandbox's network policy **blocks Bybit, xAI and TypeSafe hosts**: integrations are built to
  the documented protocols and tested against local fakes (fake V5 WebSocket server, httpx
  MockTransport). Nothing has touched the real endpoints yet — first real step: `darwin run --mode paper`.
* Gates: `.venv/bin/pytest` (~220 tests, ~4 min; `-m "not slow"` to skip the long science tests) ·
  `.venv/bin/ruff check src tests` · `.venv/bin/ruff format --check src tests` · `.venv/bin/mypy` (strict).

## Status
* **Milestone 1 (vertical slice)** — done: Bybit stream → events → features → agents → TradeIntent →
  Risk Governor → sim/testnet execution → fills → ledger → fitness → kill/promote/mutate → new
  generation → leaderboard. DoD assertions: `tests/test_e2e_loop.py`, `tests/test_paper_mode.py`.
* **Milestone 2** — done: dashboard (`/`), intelligence layer (Jev/Grok/mock, fast+slow path,
  arrival-time signals), attribution analytics, external webhook feeds, Parquet recorder + DuckDB
  loader + Bybit trade-dump loader, offline tournament (`darwin evolve`), allocator benchmark
  (`darwin bench-allocators`), Level-2 DSL + sandbox + registry (`darwin research`).

## QM review log
* **Iteration 1 (commit a94dd61): 6.5/10, NOT APPROVED.** Blockers: C1 governor ignored other agents'
  in-flight orders (5 agents → 4.5x vs 3x symbol cap); C2 audit DB not run-scoped (second run crashed /
  lost audit trail). Majors: M1 reproduction starved by immigrant quota; M2 terminal status cleared
  pending before fills; M3 non-matched fitness windows; M4 no testnet account preflight/reconciliation;
  M5 stale marks used for sizing. All fixed in 22fca71 with regression tests
  (`tests/test_qm_regressions.py`, incl. a 40-seed governor property test, chaos e2e, book deltas+gaps e2e).
* Iteration 2: pending (reviews M1 fixes + M2).

## Findings worth knowing
* **Evaluation windows must span regimes.** With a 6h window evolution killed trend followers in range
  regimes and never found the planted trend edge; with 24h (6 × 240 bars) it converges on trend-family
  species. Holdout results per champion are high-variance on a single 24h path.
* **Planted market is a real known-answer environment**: hand-built momentum(60) +30%, breakout(120)
  +47% at 1x over 3 days; mean reversion −35…−53% (the OU edge does not survive costs).
* **Null market**: train winners degrade out of sample (contrarian: train fitness 12.3 → holdout −16%).
* **Allocator benchmark** (6 seeds × 48h, planted market, $200, paired; `bench-allocators --seeds 6 --hours 48`):

  | allocator | mean final | median | mean log-growth | mean maxDD | Δ log-growth vs equal | t | wins |
  |---|---|---|---|---|---|---|---|
  | equal | 207.77 | 201.33 | +3.58% | 4.0% | — | — | — |
  | fitness_weighted | 212.10 | 203.80 | +5.43% | 4.9% | +1.84% | 1.25 | 3/6 |
  | thompson (contextual) | 196.13 | 196.56 | −1.99% | 3.4% | −5.57% | −2.40 | 1/6 |

  Thompson sampling is *worse* here (it funds on noisy posterior draws of bar returns, incl. unproven
  agents, and churns capital); fitness-weighted is modestly but not significantly better than equal.
  Default stays `fitness_weighted`. Re-run on recorded real data before believing either.

## Next steps (priority)
1. Real-endpoint verification from an unrestricted network: `paper` (public streams; verify orderbook
   `u` contiguity with `OrderBook.strict_sequence`), then `testnet` (preflight, order/execution
   stream ordering, reconciliation), record Parquet sessions and replay them.
2. Verify the Jev and xAI response shapes against the live APIs (parsers are strict and tolerant of
   alternate spellings but were built from public docs only).
3. Evolution efficiency: longer training on recorded data; robustness-weighted selection for the
   tournament (holdout variance is the dominant problem).
4. Internal crossing/netting of opposing agent orders in the challenge book (fees).
5. Crash-resume (currently: restart = new challenge; preflight refuses a non-flat account).

## Known limitations
* Sim venue passive fills are conservative (trade-through only; no queue model); trade-dump replays
  synthesize the book (optimistic slippage for size).
* At `sim.speed` > ~600x event re-stamping degrades simulated venue timing (warning logged).
* Liquidation model assumes cross margin with uniform adverse moves (conservative for correlated crypto).
* Level-2 DSL has no loops by design; strategies needing iteration must use feature-API aggregates.
