from __future__ import annotations

from typing import Any

import pytest

from darwin.config.challenge import InstrumentSpec, RiskLimits
from darwin.core.intent import TradeIntent
from darwin.core.types import Side
from darwin.market.state import MarketState
from darwin.portfolio.ledger import Ledger
from darwin.risk.governor import RiskGovernor, VenueHealth, liquidation_distance
from tests.conftest import T0, book, ticker, trade

SPEC = {
    "BTCUSDT": InstrumentSpec(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001, min_notional=5)
}


def intent(target: float, **kw: Any) -> TradeIntent:
    base: dict[str, Any] = {
        "intent_id": kw.pop("intent_id", "I1"),
        "ts": T0,
        "agent_id": "A1",
        "genome_id": "G1",
        "symbol": "BTCUSDT",
        "target_exposure": target,
        "confidence": 0.8,
        "reason": "entry",
        "stop_loss_pct": 0.02,
    }
    base.update(kw)
    return TradeIntent(**base)


@pytest.fixture
def env() -> tuple[RiskGovernor, Ledger, MarketState]:
    lim = RiskLimits(kill_switch_file=None, max_gross_leverage=5, max_symbol_leverage=3, max_agent_leverage=5)
    gov = RiskGovernor(lim, SPEC)
    led = Ledger()
    led.open_account("challenge", "challenge", 1_000.0)
    ms = MarketState(["BTCUSDT"])
    ms.apply(book("BTCUSDT", T0, 50_000.0, spread=1.0, qty=5.0))
    ms.apply(ticker("BTCUSDT", T0, 50_000.0))
    ms.apply(trade("BTCUSDT", T0, 50_000.0))
    return gov, led, ms


def evaluate(
    gov: RiskGovernor,
    led: Ledger,
    ms: MarketState,
    it: TradeIntent,
    capital: float = 1_000.0,
    now: int = T0 + 100,
    pending: float = 0.0,
    health: VenueHealth | None = None,
):  # type: ignore[no-untyped-def]
    return gov.evaluate(
        it,
        led["challenge"],
        capital,
        ms["BTCUSDT"],
        {"BTCUSDT": 50_000.0},
        pending,
        now,
        health or VenueHealth(),
    )


def test_basic_sizing_and_lot_rounding(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    d = evaluate(gov, led, ms, intent(1.0))
    # 1.0 x 1000 / 50000 = 0.02 BTC (lot 0.001)
    assert d.approved and d.order_qty == pytest.approx(0.02) and d.risk_increasing


def test_agent_leverage_clip(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    d = evaluate(
        gov, led, ms, intent(4.9), capital=100.0
    )  # agent lev ok (<=5) but symbol cap 3x equity(1000)
    assert d.approved
    d2 = evaluate(gov, led, ms, intent(50.0, intent_id="I2", agent_id="A9"), capital=1_000.0, now=T0 + 200)
    assert "clip:agent_leverage" in d2.reasons or "clip:symbol_exposure" in d2.reasons
    assert abs(d2.order_qty) * 50_000 <= 3 * 1_000 + 1e-6


def test_missing_stop_loss_rejected(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    d = evaluate(gov, led, ms, intent(1.0, stop_loss_pct=None))
    assert not d.approved and "missing_stop_loss" in d.reasons


def test_stop_loss_out_of_bounds(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    d = evaluate(gov, led, ms, intent(1.0, stop_loss_pct=0.5))
    assert not d.approved and any(r.startswith("stop_loss_out_of_bounds") for r in d.reasons)


def test_stale_data_blocks_new_risk_but_not_exits(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    late = T0 + 60_000
    d = evaluate(gov, led, ms, intent(1.0), now=late)
    assert not d.approved and any(r.startswith("stale_data") for r in d.reasons)
    # give the agent a position, then ask to exit on stale data: must be allowed
    led["challenge"].pos("A1", "BTCUSDT").apply_fill(Side.BUY, 0.02, 50_000.0)
    d2 = evaluate(gov, led, ms, intent(0.0, reason="exit", stop_loss_pct=None, intent_id="I2"), now=late)
    assert d2.approved and d2.order_qty == pytest.approx(-0.02) and not d2.risk_increasing


def test_kill_switch_blocks_increase_allows_reduce(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    gov.engage_kill_switch()
    assert not evaluate(gov, led, ms, intent(1.0)).approved
    led["challenge"].pos("A1", "BTCUSDT").apply_fill(Side.BUY, 0.02, 50_000.0)
    assert evaluate(
        gov, led, ms, intent(0.0, reason="kill_switch", stop_loss_pct=None, intent_id="I2")
    ).approved


def test_kill_switch_file(env, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, led, ms = env
    f = tmp_path / "KILL"
    gov = RiskGovernor(RiskLimits(kill_switch_file=str(f)), SPEC)
    assert evaluate(gov, led, ms, intent(1.0)).approved
    f.write_text("stop")
    d = evaluate(gov, led, ms, intent(0.5, intent_id="I2"), now=T0 + 10_000)
    assert not d.approved and "kill_switch" in d.reasons


def test_flip_rejected_is_downgraded_to_close(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    led["challenge"].pos("A1", "BTCUSDT").apply_fill(Side.BUY, 0.02, 50_000.0)
    gov.engage_kill_switch()
    d = evaluate(gov, led, ms, intent(-1.0, reason="flip"))
    assert d.approved and d.order_qty == pytest.approx(-0.02) and "flip_downgraded_to_close" in d.reasons


def test_pending_order_blocks(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    d = evaluate(gov, led, ms, intent(1.0), pending=0.01)
    assert not d.approved and "order_pending" in d.reasons


def test_duplicate_intent_protection(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    assert evaluate(gov, led, ms, intent(1.0)).approved
    d = evaluate(gov, led, ms, intent(1.0, intent_id="I2"), now=T0 + 500)
    assert not d.approved and "duplicate_intent" in d.reasons


def test_below_min_order(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    d = evaluate(gov, led, ms, intent(0.02), capital=100.0)  # $2 notional
    assert not d.approved and "below_min_order" in d.reasons


def test_gross_exposure_across_agents(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    acct = led["challenge"]
    acct.pos("A2", "BTCUSDT").apply_fill(Side.BUY, 0.05, 50_000.0)  # $2500 = 2.5x of 1000 equity
    d = evaluate(gov, led, ms, intent(2.0))
    # symbol cap 3x => only $500 more allowed on BTC
    assert d.approved and d.clipped
    assert abs(d.order_qty) * 50_000 <= 500 + 1e-6


def test_liquidation_distance_clip(env) -> None:  # type: ignore[no-untyped-def]
    _, led, ms = env
    lim = RiskLimits(
        kill_switch_file=None,
        max_gross_leverage=50,
        max_symbol_leverage=50,
        max_agent_leverage=50,
        min_liquidation_distance_pct=0.2,
    )
    gov = RiskGovernor(lim, SPEC)
    d = evaluate(gov, led, ms, intent(40.0))
    assert d.approved and "clip:liquidation_distance" in d.reasons
    post_gross = abs(d.order_qty) * 50_000
    assert liquidation_distance(1_000.0, post_gross, lim.maintenance_margin_rate) >= 0.2 - 1e-9


def test_circuit_breaker(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    acct = led["challenge"]
    acct.peak_equity = 2_000.0  # equity 1000 => 50% drawdown
    trip = gov.check_breakers(acct, 1_000.0)
    assert trip is not None and trip.startswith("max_drawdown")
    d = evaluate(gov, led, ms, intent(1.0))
    assert not d.approved and any(r.startswith("circuit_breaker") for r in d.reasons)


def test_api_health_blocks_new_risk(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    h = VenueHealth(consecutive_errors=10)
    d = evaluate(gov, led, ms, intent(1.0), health=h)
    assert not d.approved and any(r.startswith("api_errors") for r in d.reasons)
    h2 = VenueHealth(reconcile_ok=False, reconcile_detail="x")
    assert not evaluate(gov, led, ms, intent(1.0, intent_id="I9"), health=h2).approved


def test_rate_limit(env) -> None:  # type: ignore[no-untyped-def]
    _, led, ms = env
    gov = RiskGovernor(RiskLimits(kill_switch_file=None, max_orders_per_minute=2, dedup_window_ms=0), SPEC)
    for i in range(2):
        led2 = Ledger()
        led2.open_account("challenge", "challenge", 1_000.0)
        assert gov.evaluate(
            intent(1.0, intent_id=f"I{i}", agent_id=f"A{i}"),
            led2["challenge"],
            1_000.0,
            ms["BTCUSDT"],
            {"BTCUSDT": 50_000.0},
            0.0,
            T0 + 100 + i,
            VenueHealth(),
        ).approved
    d = evaluate(gov, led, ms, intent(1.0, intent_id="I3", agent_id="A3"), now=T0 + 200)
    assert not d.approved and "order_rate_limit" in d.reasons


def test_limits_are_immutable() -> None:
    lim = RiskLimits(kill_switch_file=None)
    gov = RiskGovernor(lim, SPEC)
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen model raises ValidationError
        gov.limits.max_gross_leverage = 100  # type: ignore[misc]
    assert not hasattr(gov, "set_limits")
    assert gov.fingerprint == lim.fingerprint()


def test_non_finite_target_rejected() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic rejects inf/nan at construction
        intent(float("inf"))


def test_unknown_symbol_rejected(env) -> None:  # type: ignore[no-untyped-def]
    gov, led, ms = env
    it = intent(1.0, symbol="DOGEUSDT")
    d = gov.evaluate(
        it, led["challenge"], 1_000.0, ms["BTCUSDT"], {"BTCUSDT": 50_000.0}, 0.0, T0, VenueHealth()
    )
    assert not d.approved and "symbol_not_allowed" in d.reasons
