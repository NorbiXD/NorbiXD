from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from darwin.api.app import create_app
from darwin.core.events import BookSnapshot, Event, IntelligenceSignal, TradeEvent
from darwin.market.synthetic import SyntheticMarket
from darwin.persistence.store import AuditStore
from darwin.replay.bybit_dumps import read_trade_dump, with_synthetic_book
from darwin.replay.recorder import ParquetRecorder, load_events
from darwin.runtime.replay import build_replay
from darwin.signals.external import DisabledFeed, WebhookRejected, WebhookSignalFeed
from tests.conftest import T0, make_config

# --------------------------------------------------------------------------- parquet / duckdb


def test_parquet_round_trip_preserves_events_and_order(tmp_path: Path) -> None:
    m = SyntheticMarket(symbols=("BTCUSDT", "ETHUSDT"), start_ts=T0, duration_ms=3_600_000, step_ms=5_000, seed=1)
    events: list[Event] = list(m.events())
    events.append(IntelligenceSignal(ts=T0 + 1_000, signal_id="s", source="t", value=0.3, confidence=0.5))
    events.sort(key=lambda e: e.ts)
    rec = ParquetRecorder(tmp_path, "run1", flush_every=500)
    rec.record_all(events)
    assert len(list((tmp_path / "run1").glob("*.parquet"))) > 1
    loaded = list(load_events(tmp_path / "run1"))
    assert loaded == events
    only_btc_trades = list(load_events(tmp_path / "run1", kinds=("trade",), symbols=("BTCUSDT",)))
    assert only_btc_trades and all(isinstance(e, TradeEvent) and e.symbol == "BTCUSDT" for e in only_btc_trades)
    window = list(load_events(tmp_path / "run1", start_ts=T0 + 600_000, end_ts=T0 + 1_200_000))
    assert all(T0 + 600_000 <= e.ts < T0 + 1_200_000 for e in window)


def test_replay_from_recording_is_identical_to_direct(tmp_path: Path) -> None:
    cfg = make_config(challenge={"duration_hours": 3}, evolution={"population_size": 8, "generation_bars": 40})
    m = SyntheticMarket(symbols=cfg.challenge.symbols, start_ts=T0, duration_ms=cfg.duration_ms + 60_000,
                        step_ms=5_000, seed=6)
    events = list(m.events())
    ParquetRecorder(tmp_path, "rec").record_all(events)
    a = build_replay(cfg, iter(events), T0)
    b = build_replay(cfg, load_events(tmp_path / "rec"), T0)
    assert a.driver.run() == b.driver.run()
    assert a.engine.stats == b.engine.stats


def test_bybit_trade_dump_loader(tmp_path: Path) -> None:
    csv_path = tmp_path / "BTCUSDT2026-09-01.csv"
    csv_path.write_text(
        "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional\n"
        "1756684800.1234,BTCUSDT,Buy,0.010,108000.5,PlusTick,m1,1.08e+11,0.01,1080.005\n"
        "1756684800.9000,BTCUSDT,Sell,0.002,108000.0,MinusTick,m2,2.16e+10,0.002,216.0\n"
        "1756684802.0000,BTCUSDT,Buy,0.001,108001.0,PlusTick,m3,1.08e+10,0.001,108.001\n"
    )
    trades = list(read_trade_dump(csv_path))
    assert [t.trade_id for t in trades] == ["m1", "m2", "m3"] and trades[0].ts == 1756684800123
    evs = list(with_synthetic_book(trades, tick=0.1, every_ms=1_000))
    books = [e for e in evs if isinstance(e, BookSnapshot)]
    assert len(books) == 2  # at t0 and at t0+1.9s (>= 1s since the last one)
    assert books[0].bids[0][0] < 108000.5 < books[0].asks[0][0]


# --------------------------------------------------------------------------- webhooks


def _feed(out: list[IntelligenceSignal]) -> WebhookSignalFeed:
    return WebhookSignalFeed("alpha", "s3cret-s3cret-s3cret", out.append, lambda: T0, ("BTCUSDT",), max_per_minute=3)


def test_webhook_requires_valid_signature_and_schema() -> None:
    out: list[IntelligenceSignal] = []
    feed = _feed(out)
    body = json.dumps({"id": "a1", "symbol": "BTCUSDT", "value": 0.5, "confidence": 0.9}).encode()
    with pytest.raises(WebhookRejected) as e:
        feed.ingest(body, "sha256=deadbeef")
    assert e.value.status == 401
    sig = feed.ingest(body, WebhookSignalFeed.sign("s3cret-s3cret-s3cret", body))
    assert sig.ts == T0 and sig.source == "external:alpha" and out == [sig]
    with pytest.raises(WebhookRejected) as e:  # replay of the same id
        feed.ingest(body, WebhookSignalFeed.sign("s3cret-s3cret-s3cret", body))
    assert e.value.status == 409
    bad = json.dumps({"symbol": "BTCUSDT", "value": 5, "confidence": 0.9}).encode()
    with pytest.raises(WebhookRejected) as e:
        feed.ingest(bad, WebhookSignalFeed.sign("s3cret-s3cret-s3cret", bad))
    assert e.value.status == 422
    other = json.dumps({"id": "z", "symbol": "DOGEUSDT", "value": 0.1, "confidence": 0.1}).encode()
    with pytest.raises(WebhookRejected):
        feed.ingest(other, WebhookSignalFeed.sign("s3cret-s3cret-s3cret", other))


def test_webhook_rate_limit_and_disabled_feed() -> None:
    out: list[IntelligenceSignal] = []
    feed = _feed(out)
    for i in range(3):
        b = json.dumps({"id": f"x{i}", "value": 0.1, "confidence": 0.1}).encode()
        feed.ingest(b, WebhookSignalFeed.sign("s3cret-s3cret-s3cret", b))
    b = json.dumps({"id": "x9", "value": 0.1, "confidence": 0.1}).encode()
    with pytest.raises(WebhookRejected) as e:
        feed.ingest(b, WebhookSignalFeed.sign("s3cret-s3cret-s3cret", b))
    assert e.value.status == 429
    d = DisabledFeed("telegram", "platform terms do not permit this use")
    assert not d.enabled
    with pytest.raises(RuntimeError):
        d.start()


# --------------------------------------------------------------------------- API


@pytest.fixture(scope="module")
def api(tmp_path_factory: pytest.TempPathFactory) -> tuple[TestClient, list[IntelligenceSignal]]:
    tmp = tmp_path_factory.mktemp("api")
    cfg = make_config(challenge={"duration_hours": 6},
                      evolution={"population_size": 10, "generation_bars": 60, "min_trades": 2})
    store = AuditStore(f"sqlite:///{tmp / 'api.db'}", run_id="api")
    m = SyntheticMarket(symbols=cfg.challenge.symbols, start_ts=T0, duration_ms=cfg.duration_ms + 60_000,
                        step_ms=5_000, seed=3)
    h = build_replay(cfg, m.events(), T0, store=store, run_id="api", intelligence=True)
    h.driver.run()
    pushed: list[IntelligenceSignal] = []
    feed = WebhookSignalFeed("alpha", "s3cret-s3cret-s3cret", pushed.append, lambda: T0, ())
    return TestClient(create_app(h.engine, {"alpha": feed})), pushed


def test_api_read_models(api) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    st = client.get("/api/state").json()
    assert st["starting_capital"] == 200.0 and st["generation"] >= 3 and st["ended"]
    lb = client.get("/api/leaderboard").json()
    assert len(lb) >= 10 and {"agent_id", "species", "fitness", "weight", "status"} <= set(lb[0])
    tree = client.get("/api/lineage/tree").json()
    assert any(n["parents"] for n in tree) or any(n["status"] == "dead" for n in tree)
    decisions = client.get("/api/decisions?limit=5").json()
    assert decisions and "outcomes" in decisions[0]
    agent = client.get(f"/api/agents/{lb[0]['agent_id']}").json()
    assert agent["genome"]["terms"]
    assert client.get("/api/agents/NOPE").status_code == 404
    assert client.get("/api/signals/recent").json()


def test_api_explain_and_attribution(api) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    d = client.get("/api/decisions?limit=50").json()
    iid = next(x["intent_id"] for x in d if x["reason"] in ("entry", "exit", "flip", "stop_loss", "take_profit"))
    ex = client.get(f"/api/explain/{iid}").json()
    assert ex["intent"]["intent_id"] == iid and "risk_decisions" in ex
    why = client.get("/api/why", params={"agent": ex["intent"]["agent_id"], "at": ex["intent"]["ts"]}).json()
    assert why["intent"]["agent_id"] == ex["intent"]["agent_id"]
    species = client.get("/api/attribution/species").json()
    assert species and {"species", "regime", "trades", "t_stat"} <= set(species[0])
    sigs = client.get("/api/attribution/signals").json()
    assert sigs and {"provider", "topic", "regime", "horizon_bars", "rank_ic"} <= set(sigs[0])


def test_api_webhook_and_kill_switch(api, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    client, pushed = api
    body = json.dumps({"id": "w1", "value": -0.4, "confidence": 0.7}).encode()
    r = client.post("/api/signals/alpha", content=body,
                    headers={"X-Darwin-Signature": WebhookSignalFeed.sign("s3cret-s3cret-s3cret", body)})
    assert r.status_code == 202 and pushed
    assert client.post("/api/signals/alpha", content=body).status_code == 401
    assert client.post("/api/signals/unknown", content=body).status_code == 404
    assert client.post("/api/kill-switch", json={"engage": True}).status_code == 403  # disabled without token
    monkeypatch.setenv("DARWIN_OPERATOR_TOKEN", "tok")
    assert client.post("/api/kill-switch", json={"engage": True}).status_code == 401
    r = client.post("/api/kill-switch", json={"engage": True}, headers={"Authorization": "Bearer tok"})
    assert r.json() == {"kill_switch": True}
    client.post("/api/kill-switch", json={"engage": False}, headers={"Authorization": "Bearer tok"})


def test_api_stream(api) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    with client.websocket_connect("/api/stream") as ws:
        msg = ws.receive_json()
    assert {"state", "leaderboard", "allocations", "decisions", "lineage", "equity", "positions"} <= set(msg)
