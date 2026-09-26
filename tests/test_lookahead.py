"""Look-ahead / leakage tests.

The strongest one is the *future-perturbation* test: run the full system (features, agents,
governor, execution, fitness, selection, reproduction, allocation) on a stream, then again on a
stream that is identical up to ``T_cut`` and wildly different afterwards. Every decision, order,
fill and lineage event stamped at or before ``T_cut`` must be byte-for-byte identical. Any
leakage of future information anywhere in the pipeline breaks this.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from darwin.core.events import Event, IntelligenceSignal, LiquidationEvent, TickerEvent, TradeEvent
from darwin.core.types import Side
from darwin.features.engine import FeatureEngine
from darwin.market.bars import Bar
from darwin.market.synthetic import SyntheticMarket
from darwin.runtime.replay import build_replay
from darwin.signals.board import SignalBoard
from tests.conftest import T0, make_config

HOUR = 3_600_000


def _perturb(events: list[Event], t_cut: int, seed: int = 99) -> list[Event]:
    rng = np.random.default_rng(seed)
    out: list[Event] = []
    for ev in events:
        if ev.ts < t_cut:
            out.append(ev)
            continue
        if isinstance(ev, TradeEvent):
            f = float(np.exp(rng.normal(0.05, 0.02)))  # a +5% jump and noise after the cut
            out.append(
                ev.model_copy(
                    update={
                        "price": ev.price * f,
                        "qty": ev.qty * 3,
                        "taker_side": Side.SELL if rng.random() < 0.9 else Side.BUY,
                    }
                )
            )
        elif isinstance(ev, TickerEvent) and ev.mark_price is not None:
            out.append(
                ev.model_copy(
                    update={
                        "mark_price": ev.mark_price * 1.05,
                        "funding_rate": 0.01,
                        "open_interest": (ev.open_interest or 1) * 2,
                    }
                )
            )
        else:
            out.append(ev)
    # extreme future-only information: liquidation cascades and a strong narrative signal
    out.append(LiquidationEvent(ts=t_cut + 1, symbol="BTCUSDT", side=Side.BUY, price=1.0, qty=1e6))
    out.append(
        IntelligenceSignal(
            ts=t_cut + 2, signal_id="future", source="test", symbol=None, value=1.0, confidence=1.0
        )
    )
    out.sort(key=lambda e: e.ts)
    return out


def _run(events: list[Event], cfg: Any) -> dict[str, list[Any]]:
    h = build_replay(cfg, iter(events), T0)
    rec: dict[str, list[Any]] = {"intents": [], "decisions": [], "fills": [], "lineage": [], "weights": []}
    eng = h.engine
    orig_route = eng._route
    orig_fill = eng._on_fill

    def route(intent, only=None):  # type: ignore[no-untyped-def]
        rec["intents"].append(intent.model_dump(mode="json"))
        orig_route(intent, only)

    def on_fill(ev):  # type: ignore[no-untyped-def]
        rec["fills"].append(ev.model_dump(mode="json"))
        orig_fill(ev)

    eng._route = route  # type: ignore[method-assign]
    eng._on_fill = on_fill  # type: ignore[method-assign]
    h.driver.run()
    rec["lineage"] = [lr for lr in eng.recent_lineage]
    return rec


def test_future_perturbation_does_not_change_past_decisions() -> None:
    cfg = make_config(
        challenge={"duration_hours": 6},
        evolution={"population_size": 10, "generation_bars": 60, "min_trades": 2, "min_age_generations": 0},
        allocator={"rebalance_bars": 20},
    )
    market = SyntheticMarket(
        symbols=cfg.challenge.symbols,
        start_ts=T0,
        duration_ms=cfg.duration_ms + 60_000,
        step_ms=5_000,
        seed=5,
    )
    base = list(market.events())
    t_cut = T0 + 4 * HOUR  # after several generations of evolution
    a = _run(base, cfg)
    b = _run(_perturb(base, t_cut), cfg)

    def upto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if r["ts"] <= t_cut]

    pa, pb = upto(a["intents"]), upto(b["intents"])
    assert len(pa) > 50, "test must exercise many decisions before the cut"
    assert pa == pb
    assert upto(a["fills"]) == upto(b["fills"])
    la, lb = upto(a["lineage"]), upto(b["lineage"])
    assert any(r["kind"] in ("killed", "mutated", "born") for r in la)
    assert la == lb
    # and the perturbation really did change the future
    assert a["intents"] != b["intents"]


def _bar(sym: str, end: int, close: float) -> Bar:
    return Bar(
        sym,
        end - 60_000,
        end,
        close,
        close,
        close,
        close,
        1,
        0.5,
        0.5,
        1,
        close,
        close,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        False,
    )


def test_feature_view_cannot_see_future_bars() -> None:
    fe = FeatureEngine(["BTCUSDT"])
    for i in range(1, 11):
        fe.add_bar(_bar("BTCUSDT", T0 + i * 60_000, 100.0 + i))
    v = fe.view("BTCUSDT", T0 + 5 * 60_000)
    assert v.n_bars == 5 and v.close() == 105.0
    assert v.a.close[-1] == 105.0 and len(v.a.close) == 5


def test_bar_is_closed_before_boundary_event_is_applied() -> None:
    cfg = make_config(challenge={"duration_hours": 1, "symbols": ["BTCUSDT"]})
    from tests.conftest import book, ticker, trade

    evs: list[Event] = [
        book("BTCUSDT", T0 + 1, 100.0),
        ticker("BTCUSDT", T0 + 1, 100.0),
        trade("BTCUSDT", T0 + 30_000, 100.0, tid="a"),
    ]
    boundary = (T0 // 60_000 + 1) * 60_000
    evs.append(trade("BTCUSDT", boundary, 999.0, tid="b"))  # exactly on the boundary: next bar
    h = build_replay(cfg, iter(evs), T0)
    for e in evs:
        h.engine.handle(e)
    bars = list(h.engine.features.history["BTCUSDT"])
    assert bars and bars[-1].end_ts == boundary and bars[-1].close == 100.0


def test_signal_board_hides_future_signals() -> None:
    sb = SignalBoard()
    sb.add(
        IntelligenceSignal(
            ts=T0 + 10_000, signal_id="s1", source="x", symbol="BTCUSDT", value=1.0, confidence=1.0
        )
    )
    v, ids = sb.value("BTCUSDT", "", T0)
    assert v == 0.0 and ids == []
    v2, ids2 = sb.value("BTCUSDT", "", T0 + 10_000)
    assert v2 > 0 and ids2 == ["s1"]
