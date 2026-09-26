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
* An order stays *open* (still reserving exposure) while the venue reports more executed
  quantity than we have fills for: Bybit's `order` and `execution` streams are not mutually
  ordered, so `Filled` can arrive before the executions. Missing executions are queried and, if
  they never come, position reconciliation arbitrates.
* "Market" orders are marketable IOC limits with an explicit slippage cap. Exits widen the cap
  (≥2%) and anchor it on the *worse* of the reference price and the executable touch, so a book
  that gapped away from a lagging mark cannot leave a stop unfillable (found by
  `test_stop_loss_exit_goes_through_while_an_increasing_order_is_working`).
* No ACK within `order_ack_timeout_ms` ⇒ query by `orderLinkId` (open → history → not found),
  replaying executions (idempotent by `exec_id`). An order that *was* acknowledged but whose
  final state never arrives is queried too: an IOC after 2× the ACK timeout, a resting order
  after its cancel (or 10× if none was sent). Three silent queries ⇒ `UNKNOWN`, venue health
  degraded, new risk blocked, re-queried every 10× until the venue answers definitively.
* Bybit REST: a `retCode 0` create means *request accepted* (ACK). Definitive error codes ⇒
  REJECTED; ambiguous failures (timeouts, 5xx, rate limits) emit nothing and are reconciled;
  `orderLinkId is duplicate` ⇒ the earlier attempt landed ⇒ reconcile. Private-stream reconnect
  ⇒ re-query all open orders + positions.
* Periodic position snapshots reconcile the challenge account. A mismatch blocks new risk only if
  it persists over two consecutive snapshots with no order in flight and no fill in the last 5s
  (a snapshot can predate a fill); an order older than 60 s no longer counts as "in flight", so
  a stuck order cannot hide a mismatch. One `reconcile_mismatch` event per episode. Venue *business* rejects (Bybit `EC_*` stream reasons and
  `110xxx` retCodes: balance, price band, min size, reduce-only) do not degrade venue health;
  lost requests, auth and transport failures do.

## 6. Risk Governor (not evolvable)

Frozen `RiskLimits` from `challenge.yaml` (fingerprint stored per run). Risk-reducing orders are
**always** allowed (kill switch, stale data and breakers never block getting flat; exits may use
the last known price when no fresh one exists). Risk increasing orders need a **fresh** reference
price (a mark from a ticker stream that went quiet falls back to the book mid, then the last
trade, else the symbol is stale) and pass, or are clipped/rejected with explicit persisted
reasons. **Exposure reservations:** every agent decides on the same bar close and orders take
tens of milliseconds to fill, so the per-symbol, gross, concurrent-position and liquidation
checks count the remaining quantity of *all agents'* in-flight risk-increasing orders (and
orders whose fills the venue reported but we have not yet received), not just filled positions.
Without this, N agents could each consume the full headroom (QM iteration 1, C1; pinned by a
40-seed randomized property test). Checks:
kill switch (manual or `KILL` file) · drawdown & daily-loss breakers (flatten) · venue health
(API errors, reconciliation, private stream) · stale data · mandatory bounded stop-loss ·
slippage tolerance · order-rate limit · duplicate intents · max concurrent positions ·
per-agent leverage · per-symbol and gross exposure (gross of agent sub-positions, conservative)
· liquidation distance (uniform-adverse-move model, requires `max(min_distance, 1.5×stop)`) ·
instrument leverage · lot/min-notional rounding · post-trade liquidation re-check. A rejected
flip is downgraded to a close. Martingale is structurally impossible: size is a function of
current equity and allocation only; agents cannot request more after losses.

**Flatten until flat.** A kill switch, a tripped breaker (`flatten_on_breaker`) or the end of the
challenge is a *condition*, not a one-shot action. On every heartbeat and bar close while it
holds, the engine cancels working risk-increasing orders and re-issues exits for every open
sub-position, until ledger, venue positions and working orders are all clear
(`flatten_complete`). A system exit is never blocked by a working entry (the entry is cancelled
and the filled quantity exited; a late fill is exited on the next pass); only an in-flight exit
blocks another one, because an exit that is `UNKNOWN` may already have executed and a duplicate
would flip the position (lost final reports are recovered by the stale-order queries of §5).
Retries per sub-position are rate-bounded.

A position that only the *venue* holds (a manual trade, fills lost for good) is adopted under a
flatten condition once two consecutive snapshots show it with nothing in flight: it becomes a
`SYSTEM` sub-position at the current price and is closed through the same governed, persisted
exit path, with `reduceOnly` so it can never open or flip a position. Book sub-positions that
exactly offset each other are closed against each other internally (no orders).

**What is guaranteed, and for how long.** The replay driver drains for up to 15 simulated
minutes after the end; the live driver (sim/paper/testnet/live) keeps the process alive for
`challenge.end_flatten_timeout_s` (default 30 min), raising `flatten_overdue` every minute,
and records `flatten_incomplete` if it still has to stop. Within those bounds the account ends
flat unless the venue refuses every exit for the whole window, or the venue cannot be reached
at all (no exchange-side catastrophe stops yet: progress.md). Pinned by failure-injection tests:
a minute of venue rejects during a kill, partial exit fills, lost exit orders that go `UNKNOWN`,
lost final reports and fills, a venue-only position, a breaker trip under rejects, challenge end
under chaos, 5 minutes of rejected exits after a testnet end, a resting entry at kill time, and
a stop racing a working entry (`tests/test_flatten.py`, `tests/test_testnet_mode.py`).

The challenge book also re-syncs to the shadow books every bar: if an exit filled in an agent's
shadow book but not in the challenge book (a reject, a stop hit at a different entry, a lost
fill), the leftover is closed (`desync_exit`); leftovers of dead or defunded agents too.

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
* **Window-matched relative fitness** drives selection: an agent's fitness on its window minus
  the median fitness of every agent alive over *exactly the same bars*, so agents of different
  ages are never compared across different regimes. Parents, the champion and capital also
  require positive *absolute* fitness (beating a losing crowd is not an edge). The evaluation
  window must span regimes (6 generations × 240 bars = 24h by default): with a 6h window
  evolution killed trend followers during range regimes and never found the planted trend edge;
  with 24h it can, though convergence on trend species is seed-dependent (§15).
* Eligibility: `min_trades` and `min_age_generations`. Nobody dies before that — except by
  **ruin** (shadow equity < `ruin_fraction`).
* Persistent inferiority = bottom `kill_fraction` by window-matched relative fitness **and**
  paired t-stat of bar returns vs. the cohort median on identical bars `< -kill_t_stat` ⇒
  strike + probation (no capital, still evaluated). `max_strikes` ⇒ death. Recovery removes
  strikes. The champion is not immune. Inactive agents die after `max_inactive_generations`.
* Diversity: fitness sharing penalises correlation with better-ranked agents
  (`correlation_penalty` 1.0); species share cap; at most `max_offspring_per_parent` children per
  parent per generation; immigrants favour under-represented primitives; offspring deduped by
  genome hash/distance. The cohort median includes agents that died inside the window, on the
  bars they lived (no survivorship bias in the benchmark).
* Reproduction: tournament selection among qualified agents; crossover then light mutation, or
  mutation. Immigrants take a fraction of the free slots (never all of them when parents
  qualify); if nobody qualifies, only immigrants are born (losers don't breed). Every mutation
  is recorded (`terms[0].momentum.lookback: 20 → 34`).
* Champion/challenger: a challenger dethrones the champion only with higher adjusted fitness
  **and** a paired t-stat above `champion_t_stat` on common bars.

## 10. Capital allocator

`equal` (baseline), `fitness_weighted` (softmax over adjusted fitness, top-K, capped, cash
buffer, exploration slice for the best unproven challenger), `thompson` (contextual Thompson
sampling over each agent's posterior mean bar return in the current regime, Kelly-like
weights). Weights rebalance every `rebalance_bars` and at every generation; defunded or killed
agents are flattened; newly funded agents are synced to their current shadow exposure. Whether
Thompson beats fitness-weighting must be shown by benchmark, not asserted (see progress.md).

**De-cloning.** All three policies walk candidates best-first and give no capital to one whose
bar returns correlate above `max_pair_correlation` (0.7) with an agent already funded; the
exploration slot obeys the same rule (exploration buys information, not more of the same bet).
On the 96h default replay this took rebalances in which a >0.7-correlated pair held most of the
capital from 21/93 to 0/93. Capital can still concentrate in one *lineage family* whose members
behave differently (mean largest-family share 0.84): that is diversification within a winning
family, not cloning, and it is reported rather than hidden.

## 11. Persistence & attribution

SQLAlchemy Core schema portable across SQLite (local/tests) and PostgreSQL (docker compose).
Every table is keyed by `(run_id, id)`, so many runs share one database; exchange-facing client
order ids carry a per-run tag so they never collide across runs. Writes are buffered
(`AuditStore`) and flushed off the hot path. Rows are sanitized on the way in (NaN/Inf → NULL,
numpy scalars, oversized integers). A *transient* failure (connection lost, database locked or
unreachable) puts the batch back and **halts new challenge risk** until the store recovers (no
trading without an audit trail); a row the database refuses is isolated by a row-by-row retry
and written to `dead_letters`, so one bad row can never halt trading forever. A run id that
already has an audit trail is refused; default run ids carry a random suffix.
`attribution/analytics.py` learns *where* intelligence works: trade performance by
(species, regime at entry) and each signal source's forward-return rank IC by regime and
horizon ("source X is informative only in high volatility"). `attribution/explain.py`
reconstructs "why did agent X go long SOL at T": the intent (features seen, component scores,
regime, signals, model response), the market snapshot, genome and lineage, both risk decisions
with reasons, orders, fills (slippage), the round trip (MFE/MAE/PnL) and its fitness
contribution. CLI: `darwin explain`, API: `/api/why`, `/api/explain/{intent_id}`.

## 12. Modes

`replay` (fast, deterministic) · `sim` (synthetic market, accelerated wall clock, dashboard) ·
`paper` (Bybit public streams, simulated venues) · `testnet` · `live` (off by default: needs
`challenge.live_enabled: true`, `DARWIN_LIVE_CONFIRM=I_UNDERSTAND_THIS_TRADES_REAL_MONEY`, and
keys). The shadow venue is simulated in every mode. Before paper/testnet/live the real
lot/tick/leverage filters are pulled from Bybit (instrument specs are frozen into the governor).
Before testnet/live an **account preflight** refuses to start unless the account is flat on the
challenge symbols (there is no crash-resume), the wallet covers the starting capital, margin is
cross, position mode is one-way, and per-symbol leverage is set. During the run, venue wallet
PnL is continuously compared with ledger PnL; drift beyond tolerance blocks new risk. Breakers
and the kill switch are evaluated on every heartbeat, not only at bar close.

**Warm start.** Paper/testnet/live start with no history, which used to leave the fast AI path
(it needs 60 bars) and long-lookback agents idle for up to hours. At start, up to 800 closed
1-minute candles per symbol are loaded from Bybit's public kline endpoint into the feature
history — indicators only: nothing is marked, no agent decides, no evidence or fitness is
recorded, and the still-forming candle is dropped. Candles carry OHLCV only, so trade-flow,
book and liquidation features warm up from live data.

## 13. Synthetic market (known-answer environment)

Regime-switching (trend/range/high-vol) with planted, cost-aware structure: trend drift,
range OU reversion, liquidation cascades that overshoot and retrace, and a weak book-imbalance
drift that should *not* survive taker fees. `planted_edges=False` gives a null market with the
same microstructure surface and no structure. By default it streams Bybit-style book deltas
with a periodic snapshot and occasional sequence gaps, so every sim run exercises gap detection
(the book is invalid, bars go stale, no decisions) and recovery on the next snapshot. It
validates the machinery; it is not evidence about real markets.

## 14. Multi-speed intelligence

`intelligence/`: a `DecisionProvider` (fast path: typed LONG/SHORT/WAIT, regime and validity
questions answered as probabilities; Jev / TypeSafe System One implemented against
`POST /v1/systemone`, models discovered via `GET /v1/models`) and a `NarrativeProvider` (slow
path: Grok via the xAI Responses API with the `x_search` tool, never scraping). Mocks use only
past state. The `IntelligenceService` is a bar observer: requests never block decisions, and
outputs become `IntelligenceSignal`s stamped with their **arrival** time (live: receive time;
replay: `now + simulated latency`), so agents can only use a model output after it could have
existed (the future-perturbation test also runs with intelligence enabled). Failing providers
are paused (circuit breaker); failures are logged with their exception type. X Search is agentic
and slow (20–60 s is normal), so the slow path's timeout is 180 s — it is asynchronous and never
delays a decision. `darwin run --ai` switches the mock off and refuses to start without the
provider keys, so a missing key can never silently turn "AI mode" into "mock mode". Signals are persisted with provider, model, payload and latency
and appear in `explain`. External feeds implement `ExternalSignalFeed`; the webhook feed
requires a timestamped HMAC signature (5-minute replay window), bounds the body while streaming,
rejects NaN/Infinity and implausible `observed_ts`, validates schema and symbols, dedupes ids,
rate-limits, and forces topic `external` so a webhook cannot impersonate the model or X-narrative
channels. A failing bar observer (e.g. a provider) is disabled after three errors; trading
continues.
Platforms whose terms do not permit this use stay `DisabledFeed`s.

## 15. Offline evolution, benchmarks and known-answer tests

* `darwin evolve`: a large population on a training window, then the top genomes by
  window-matched fitness, then frozen re-evaluation on a later, unseen holdout window, then a
  champion-set JSON (`darwin run --seed-genomes`). Train→holdout degradation is reported
  (winner's curse / overfitting).
* Known answers (`tests/test_evolution_science.py`), always across seeds and against a baseline
  of random, unselected genomes evaluated on the same holdout bars. **Proven:** selection avoids
  the losses of random deployment — on the planted market (6 seeds, +5.3%, t ≈ 3.0) *and* on the
  null market (+2.9%), which on noise can only be cost/risk avoidance; and on the null market
  selection produces no positive out-of-sample return and train winners degrade. **Not proven
  (an open problem, not a claim):** that evolution *discovers the planted edge*. The null-
  controlled comparison (planted advantage minus null advantage) points the right way but is not
  significant (48h training: t ≈ 1.3; 96h: t ≈ 2.0 with only 3 of 6 null seeds producing
  champions); absolute holdout returns do not separate the markets (t ≈ 0.5); trend-family
  convergence is seed-dependent. Hand-built momentum/breakout genomes earn +26…+47% at 1x over
  3 days on the planted market, so the gap is in the search, not the environment.
* `darwin bench-allocators`: equal vs fitness-weighted vs Thompson on identical seeds. Because
  evaluation happens on shadow books, the population and its decisions are *identical* across
  allocators for a given seed, so the comparison is perfectly paired (results in progress.md).

## 16. Level 2: machine-invented species

`research/`: proposals are DSL source (`dsl.py`), a statically verified Python subset where the
only reachable attributes are `v.<feature API>` and `math.<fn>` (no imports, loops,
comprehensions, lambdas, `**`, keywords, dunders, decorators or annotations). Values are bounded
by construction: no string constants except parameter keys and signal topics; outside feature
arguments every arithmetic operand is coerced with `float()` at compile time (constants, names,
bools and comparison results alike), so `"a" * 999999999` cannot be written and
`x = True + True` followed by forty `x = x * x` overflows to `inf` in microseconds instead of
building a 2^40-bit integer (QM iteration 3); `int()` is only allowed inside feature arguments,
where no reassignment can happen. Promoted code runs in-process: any exception it raises, or a
decision step slower than `evolution.max_decide_ms`, quarantines that agent only (killed with
lineage `runtime_error`, positions flattened); the bar loop continues. `sandbox.py` runs every other stage in a subprocess with a scrubbed environment (no
credentials), CPU/memory rlimits, `RLIMIT_FSIZE=0`, no database and a timeout: unit tests on
real FeatureViews (bounded, deterministic, stateless, not dead code, fast), a leakage check,
train replay with fees and slippage, holdout, and a challenger comparison against the champion
on identical bars. Passing species are saved to a registry, re-validated on load (the source
must pass the DSL validator, hash to the record's proposal id, and carry a fully passing
report), registered as
primitives (origin `sandbox:<id>`) and injected into the live population as challengers (lineage
`proposed`). `TemplateProposer` is an offline proposer; `LLMProposer` wraps any frontier model.
The Risk Governor and execution engine sit outside all of this and are not evolvable.
