# DARWIN — Architecture

DARWIN is a population of competing trading agents that evolves under selection pressure during
a fixed-duration crypto-futures challenge ($200 starting capital, Bybit V5 linear perpetuals).
The public score is terminal equity. Internally, selection runs on *evidence of edge*, not on
luck.

```
                          ┌──────────────────────── drivers (I/O, clocks) ────────────────────────┐
 Bybit public WS ─┐       │  ReplayDriver: deterministic (ts, priority, seq) merge                │
 synthetic market ┼──────►│  LiveDriver:   asyncio queue + timer heap, wall/accelerated Clock     │
 recorded data   ─┘       └───────────────────────────────┬────────────────────────────────────────┘
                                                          │ normalized events (ts = time *we* knew)
                                                          ▼
┌──────────────────────────────────────────── DarwinEngine (pure, synchronous) ─────────────────────────────┐
│ MarketState ─► BarBuilder ─► FeatureEngine ─► FeatureView (identical for every agent, memoised, logged)   │
│                                                        │                                                  │
│      Population: Agent(genome) × N  ──── decide() ───► TradeIntent (target exposure, stop, confidence…)   │
│                                                        │                                                  │
│      ┌──────────── RiskGovernor (immutable limits) ◄───┴───► shadow:<agent> book  (standard capital)      │
│      │                                                 └───► challenge book       (allocated capital)     │
│      ▼                                                                                                     │
│  ExecutionEngine (idempotent ids, ACK≠fill, dedup, timeouts→query→UNKNOWN) ──► ExecutionGateway           │
│      ▲                                                                                                     │
│  fills / order updates / funding ──► Ledger (accounts, agent sub-positions, round trips, MFE/MAE)          │
│                                                                                                            │
│  every bar: mark-to-market · protective stops · breakers · kill switch · allocation rebalance              │
│  every generation: fitness → strikes/death → champion/challenger → mutation/crossover/immigrants           │
│  AuditStore (buffered): intents, features seen, risk decisions, orders, fills, trades, lineage, fitness    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │ SimExchange (shadow venue, always)        │ SimExchange (replay/sim/paper) or BybitExecutionGateway
        ▼                                           ▼ (testnet/live, REST + private WS)
```

## 1. The contract: `TradeIntent → RiskGovernor → ExecutionEngine → Venue`

Agents are pure functions `(genome, FeatureView, own position sign) → TradeIntent`. An intent
states a **target exposure** (signed multiple of the agent's capital), a stop, take-profit,
max hold, urgency and slippage tolerance, plus its full attribution payload (every feature
read, component scores, regime, signals, model response). Agents never size orders, never see
limits, never hold credentials. Target semantics make retries idempotent and let the same intent
drive two books at different capital levels.

## 2. Two books per decision: evidence vs. capital

The single most important design choice.

* **Shadow book (`shadow:<agent>`)** — every agent, funded or not, trades a standardised
  evaluation account (`evolution.eval_capital`) on the simulated venue, with the same latency,
  book-walking slippage, fees, funding and liquidation model. **All fitness evidence comes from
  here.** Comparisons are on identical data and costs, independent of how much real capital an
  agent happens to hold, and exploration costs nothing.
* **Challenge book (`challenge`)** — the one real account (sim/testnet/live). Funded agents'
  intents are routed here too, sized by the allocator's weight × account equity. The exchange nets
  agents' sub-positions (one-way mode); the ledger keeps per-agent attribution and
  reconciliation checks `Σ agent qty == venue position`.

With $200 and Bybit lot minimums (BTC ≈ $60–110 per 0.001), funding 24 agents is impossible;
shadow evaluation is what makes a *population* meaningful at this capital size.

## 3. Time, determinism and look-ahead

* Events carry `ts` = when the system learned them (receive time live, recorded receive time in
  replay). The engine has no clock of its own.
* A bar `[start, end)` closes **at `end`, before any event stamped `end` is applied**. Drivers
  schedule explicit bar-boundary events with priority ahead of same-timestamp market data, so
  simulated fills can never use a book from after the order's arrival.
* `FeatureEngine.view(now)` exposes only bars with `end_ts <= now`; `SignalBoard` ignores
  signals whose availability `ts > now`.
* Test `test_future_perturbation_does_not_change_past_decisions` runs the entire system twice,
  once with the future (after `T_cut`) scrambled (+5% jump, flipped flow, extreme funding/OI,
  a fake liquidation cascade and a max-strength narrative signal), and asserts every intent, fill
  and lineage event at `ts <= T_cut` is identical. Replays are bit-for-bit deterministic
  (`test_replay_is_deterministic`).

## 4. Market data (Bybit V5)

`exchange/bybit/`: `parser.py` (V5 topics → normalized events), `ws.py` (reconnect with
exponential backoff + jitter, auth, chunked subscribe, app-level ping, stale-feed detection,
targeted resubscribe), `feeds.py`, `rest.py`, `gateway.py`, `signing.py`.

* Order book: snapshot resets; delta with `u <= last` ignored (duplicates); a gap invalidates the
  book and triggers a resubscribe for a fresh snapshot; crossed books are invalid; `u=1`
  snapshots (exchange restart) reset the sequence. Invalid books ⇒ symbol is "stale" ⇒ the
  governor rejects new risk.
* Public feed disconnect/stale ⇒ all books invalidated until the next snapshot.
* Trades deduplicated by trade id.

## 5. Execution

* Deterministic `orderLinkId` (`C-<agent>-<seq>`, ≤36 chars); venues reject duplicates, so a
  retry can never double-fill.
* **ACK ≠ fill**: positions change only on `FillEvent`s; an `OrderUpdate(FILLED)` without
  executions moves nothing.
* Fills deduplicated by `exec_id`; order state machine is monotonic (stale/out-of-order updates
  ignored; a late ACK cannot regress FILLED).
* "Market" orders are marketable IOC limits with an explicit slippage cap. Exits widen the cap.
* No ACK within `order_ack_timeout_ms` ⇒ query by `orderLinkId` (open → history → not found),
  replaying executions. Three silent queries ⇒ `UNKNOWN`, venue health degraded, new risk blocked
  until the venue answers definitively.
* Bybit REST: a `retCode 0` create means *request accepted* (ACK). Definitive error codes ⇒
  REJECTED; ambiguous failures (timeouts, 5xx, rate limits) emit nothing and are reconciled;
  `orderLinkId is duplicate` ⇒ the earlier attempt landed ⇒ reconcile. Private-stream reconnect
  ⇒ re-query all open orders + positions.
* Periodic position snapshots reconcile the challenge account; a mismatch (with no orders in
  flight) blocks new risk.

## 6. Risk Governor (not evolvable)

Frozen `RiskLimits` from `challenge.yaml` (fingerprint stored per run). Risk-reducing orders are
**always** allowed (kill switch, stale data and breakers never block getting flat). Risk
increasing orders pass, or are clipped/rejected with explicit persisted reasons:
kill switch (manual or `KILL` file) · drawdown & daily-loss breakers (flatten) · venue health
(API errors, reconciliation, private stream) · stale data · mandatory bounded stop-loss ·
slippage tolerance · order-rate limit · duplicate intents · max concurrent positions ·
per-agent leverage · per-symbol and gross exposure (gross of agent sub-positions, conservative)
· liquidation distance (uniform-adverse-move model, requires `max(min_distance, 1.5×stop)`) ·
instrument leverage · lot/min-notional rounding · post-trade liquidation re-check. A rejected
flip is downgraded to a close. Martingale is structurally impossible: size is a function of
current equity and allocation only; agents cannot request more after losses.

## 7. Agents: a strategy DSL

`score = Σ wᵢ·primitiveᵢ(features; θᵢ) / Σ|wᵢ|`, regime gate, entry/exit thresholds, cooldown,
risk genes (exposure request, stop, take-profit, max hold, confidence scaling), execution genes.
Primitives: momentum, breakout, mean_reversion, order_flow, funding_oi, liquidation, volatility,
contrarian, narrative (reads decayed intelligence signals: X narrative, Jev decisions, external
feeds). Single-term genomes are the classic species; negative weights create "anti-" species;
crossover of different species and structural mutation create hybrids nobody wrote. Genomes are
content-addressed (`genome_id` = hash) and persisted verbatim.

## 8. Fitness (internal) vs. terminal equity (public)

Selecting on terminal equity over a short horizon breeds the luckiest leveraged gambler. Fitness
(all components in "% growth per day" equivalents, weights in `challenge.yaml`):

* **CRRA certainty-equivalent growth** of per-trade returns — `γ=1` is log utility
  (growth-optimal/Kelly); `γ<1` more aggressive; `γ=0` risk-neutral (ruin-seeking).
* **Empirical-Bayes shrinkage** toward a slightly negative prior with data weight
  `(n/s²)/(n/s² + 1/τ²)` — precision, not trade count. Three trades with one +60% outlier carry
  ~no weight; forty consistent trades nearly full weight. The same weight scales the realised
  terminal-growth term.
* **Bootstrap ruin probability** from (return, intra-trade MAE) pairs — a leveraged bet that
  won still shows how close it came to liquidation.
* Drawdown (amortised), consistency across sub-windows and confidence→outcome rank IC (bounded
  tie-breakers), slippage.

Test `test_lucky_suicidal_bet_ranks_below_steady_edge` pins the core property.

## 9. Selection

* Births only at generation boundaries; every alive agent is marked on every bar ⇒ any two
  agents share an identical contiguous window (the younger one's life).
* Eligibility: `min_trades` and `min_age_generations`. Nobody dies before that — except by
  **ruin** (shadow equity < `ruin_fraction`).
* Persistent inferiority = bottom `kill_fraction` **and** paired t-stat of bar returns vs. the
  cohort median on identical bars `< -kill_t_stat` ⇒ strike + probation (no capital, still
  evaluated). `max_strikes` ⇒ death. Recovery removes strikes. Inactive agents die after
  `max_inactive_generations`.
* Diversity: fitness sharing penalises correlation with better-ranked agents; species share cap;
  immigrants favour under-represented primitives; offspring deduped by genome hash/distance.
* Reproduction: tournament selection among eligible, positive, low-ruin agents; crossover then
  light mutation, or mutation; if nobody qualifies, only immigrants are born (losers don't
  breed). Every mutation is recorded (`terms[0].momentum.lookback: 20 → 34`).
* Champion/challenger: a challenger dethrones the champion only with higher adjusted fitness
  **and** a paired t-stat above `champion_t_stat` on common bars.

## 10. Capital allocator

`equal` (baseline), `fitness_weighted` (softmax over adjusted fitness, top-K, capped, cash
buffer, exploration slice for the best unproven challenger), `thompson` (contextual Thompson
sampling over each agent's posterior mean bar return in the current regime, Kelly-like
weights). Weights rebalance every `rebalance_bars` and at every generation; defunded or killed
agents are flattened; newly funded agents are synced to their current shadow exposure. Whether
Thompson beats fitness-weighting must be shown by benchmark, not asserted (see progress.md).

## 11. Persistence & attribution

SQLAlchemy Core schema portable across SQLite (local/tests) and PostgreSQL (docker compose).
Writes are buffered (`AuditStore`) and flushed off the hot path. `attribution/explain.py`
reconstructs "why did agent X go long SOL at T": the intent (features seen, component scores,
regime, signals, model response), the market snapshot, genome and lineage, both risk decisions
with reasons, orders, fills (slippage), the round trip (MFE/MAE/PnL) and its fitness
contribution. CLI: `darwin explain`, API: `/api/why`, `/api/explain/{intent_id}`.

## 12. Modes

`replay` (fast, deterministic) · `sim` (synthetic market, accelerated wall clock, dashboard) ·
`paper` (Bybit public streams, simulated venues) · `testnet` · `live` (off by default: needs
`challenge.live_enabled: true`, `DARWIN_LIVE_CONFIRM=I_UNDERSTAND_THIS_TRADES_REAL_MONEY`, and
keys). The shadow venue is simulated in every mode.

## 13. Synthetic market (known-answer environment)

Regime-switching (trend/range/high-vol) with planted, cost-aware structure: trend drift,
range OU reversion, liquidation cascades that overshoot and retrace, and a weak book-imbalance
drift that should *not* survive taker fees. `planted_edges=False` gives a null market with the
same microstructure surface and no structure. It validates the machinery; it is not evidence
about real markets.
