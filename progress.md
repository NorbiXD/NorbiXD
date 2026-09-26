# progress.md — resume here

_Last updated: 2026-09-26 (session 1)._ Branch: `claude/clever-tesla-kk8pm0` (local commits; the
GitHub push was refused with 403 — the Claude GitHub App lacks access to `NorbiXD/NorbiXD`).

## Environment facts
* Python 3.12 venv at `.venv` (`uv venv --python 3.12 .venv && uv pip install -e ".[dev,postgres]"`).
* This sandbox's network policy **blocks Bybit, xAI and TypeSafe hosts**: integrations are built to
  the documented protocols and tested against local fakes (fake V5 WebSocket server, httpx
  MockTransport). Nothing has touched the real endpoints yet — first real step: `darwin run --mode paper`.
* Gates: `.venv/bin/pytest` (245 tests, ~3 min on 4 cores; `-m "not slow"`: 242 in ~1 min) ·
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
* **Iteration 2 (commit cda45d0): 7.0/10, NOT APPROVED.** Critical: flatten was fire-once and
  blocked by pending orders (a reject during a kill left the account exposed). Required: flatten
  until flat + failure-injection tests; species isolation (one raising species killed the loop;
  DSL allowed `"a" * 999999999`); clones holding capital; audit poison rows halting trading
  forever; multi-seed statistical known-answer test; integrated testnet test. Fixed in 370f203,
  56eab80 and the following commit (tests: `test_flatten.py`, `test_qm2_regressions.py`,
  `test_testnet_mode.py`, rewritten `test_evolution_science.py`). Fixing it surfaced a real bug:
  exits were priced off a lagging mark, so after a gap a stop could be unfillable.
* Iteration 3: pending.

## Findings worth knowing
* **Default-config 168h replay** (`darwin replay --hours 168`, 1.87M events in 151 s): 42 generations,
  55 offspring (38 mutated, 17 crossover) vs 13 immigrants (before the QM-1 fix: 14 vs 127); deaths:
  35 inactive, 33 persistent inferiority; 38 distinct agents funded over time; final equity 197.75.
* **Evaluation windows must span regimes.** With a 6h window evolution killed trend followers in range
  regimes and never found the planted trend edge; with 24h (6 × 240 bars) it can.
* **Multi-seed science (48h train → 24h holdout, 12 random baseline genomes per seed, same bars).**
  Seeds 101–106 planted / 201–206 null (pre-registered after a 4+4 seed pilot):

  | market | champion holdout return, per seed | mean | excess over random baseline, per seed | mean |
  |---|---|---|---|---|
  | planted | −3.6, −6.8, +3.1, −0.4, +3.1, −0.1 % | −0.8% | +6.2, −0.3, +12.1, +1.2, +7.6, +6.0 % | **+5.5% (t≈3.0)** |
  | null | −0.9, (none), −0.3, (none), −0.8, −5.6 % | −1.9% | +3.5, –, +4.2, –, +2.6, +1.3 % | +2.9% |

  Selection reliably beats random genomes, but much of that is cost/risk avoidance that exists on
  noise too (random genomes overtrade). Planted vs null champion returns: **not separated**
  (t≈1.1 incl. the pilot). Trend-family convergence is seed-dependent (seeds 101/105/106 converged
  on mean-reversion/liquidation hybrids). The earlier single-seed claims were withdrawn.
* **Clones (96h default replay):** before de-cloning, a >0.7-correlated funded pair held >50% of
  capital in 21/93 rebalances (the exploration slot was the leak); after: 0/93, p90 max pairwise
  correlation 0.83 → 0.53. One lineage family still holds ~84% of capital on average with
  behaviourally distinct members.
* **$200 is thin against Bybit minimum lots.** The dashboard shows `challenge: below_min_order`
  rejects: a 40% slice × 0.7x exposure ≈ $57 of BTC is below 0.001 BTC. Small slices silently cannot
  trade BTC; allocation is not yet lot-aware (next steps).
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
0. Lot-aware allocation for a $200 book: fund fewer agents with larger slices, or weight toward symbols
   whose minimum order fits the slice; report unfundable intents on the dashboard.
1. Real-endpoint verification from an unrestricted network: `paper` (public streams; verify orderbook
   `u` contiguity with `OrderBook.strict_sequence`), then `testnet` (preflight, order/execution
   stream ordering, reconciliation), record Parquet sessions and replay them.
2. Verify the Jev and xAI response shapes against the live APIs (parsers are strict and tolerant of
   alternate spellings but were built from public docs only).
3. Evolution efficiency — the science table says this is the real gap: longer training on recorded
   data, multi-window (walk-forward) selection instead of one 24h holdout, and a planted-vs-null
   separation test as the acceptance bar. Consider a lineage-family capital cap.
3b. Exchange-side catastrophe stops (`/v5/position/trading-stop`) so a crashed process is not
   unprotected (QM-2 optional item, not done).
4. Internal crossing/netting of opposing agent orders in the challenge book (fees).
5. Crash-resume (currently: restart = new challenge; preflight refuses a non-flat account).

## Known limitations
* Sim venue passive fills are conservative (trade-through only; no queue model); trade-dump replays
  synthesize the book (optimistic slippage for size).
* At `sim.speed` > ~600x event re-stamping degrades simulated venue timing (warning logged).
* Liquidation model assumes cross margin with uniform adverse moves (conservative for correlated crypto).
* Level-2 DSL has no loops by design; strategies needing iteration must use feature-API aggregates.
