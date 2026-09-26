"""Bybit public/private stream adapters feeding the live driver."""

from __future__ import annotations

from typing import Any, Literal

from darwin.core.events import FeedStatus
from darwin.exchange.bybit.gateway import BybitExecutionGateway
from darwin.exchange.bybit.parser import parse_private, parse_public
from darwin.exchange.bybit.signing import Credentials
from darwin.exchange.bybit.ws import ReconnectingWebSocket
from darwin.runtime.live import LiveDriver

_STATUS: dict[str, Literal["connected", "disconnected", "resync", "gap", "stale"]] = {
    "connected": "connected",
    "disconnected": "disconnected",
    "resync": "resync",
    "stale": "stale",
    "error": "disconnected",
    "auth_failed": "disconnected",
}


class BybitMarketFeed:
    def __init__(
        self, driver: LiveDriver, symbols: tuple[str, ...], url: str, depth: int = 50, **ws_kw: Any
    ) -> None:
        self.driver = driver
        self.depth = depth
        topics: list[str] = []
        for s in symbols:
            topics += [f"orderbook.{depth}.{s}", f"publicTrade.{s}", f"tickers.{s}", f"allLiquidation.{s}"]
        self.ws = ReconnectingWebSocket(
            "public", url, topics, self._on_message, self._on_status, clock_ms=driver.clock.now_ms, **ws_kw
        )

    def _on_message(self, msg: dict[str, Any], recv_ts: int) -> None:
        for ev in parse_public(msg, recv_ts):
            self.driver.push(ev)

    def _on_status(self, status: str, detail: str) -> None:
        self.driver.push(
            FeedStatus(
                ts=self.driver.clock.now_ms(),
                feed="public:bybit",
                status=_STATUS.get(status, "disconnected"),
                detail=f"{status}: {detail}",
            )
        )

    def resync(self, symbol: str) -> None:
        self.ws.request_resync(f"orderbook.{self.depth}.{symbol}")

    async def run(self) -> None:
        await self.ws.run(self.driver.stop)


class BybitPrivateFeed:
    def __init__(
        self,
        driver: LiveDriver,
        gateway: BybitExecutionGateway,
        url: str,
        credentials: Credentials,
        account: str,
        **ws_kw: Any,
    ) -> None:
        self.driver = driver
        self.gateway = gateway
        self.account = account
        self.ws = ReconnectingWebSocket(
            "private",
            url,
            ["order", "execution", "position", "wallet"],
            self._on_message,
            self._on_status,
            credentials=credentials,
            clock_ms=driver.clock.now_ms,
            **ws_kw,
        )

    def _on_message(self, msg: dict[str, Any], recv_ts: int) -> None:
        for ev in parse_private(msg, recv_ts, self.account):
            self.gateway.on_private_event(ev)
            self.driver.push(ev)

    def _on_status(self, status: str, detail: str) -> None:
        self.driver.push(
            FeedStatus(
                ts=self.driver.clock.now_ms(),
                feed="private:bybit",
                status=_STATUS.get(status, "disconnected"),
                detail=f"{status}: {detail}",
            )
        )
        if status == "connected":
            # whatever happened while we were away is recovered from REST
            self.gateway.on_private_reconnect()

    async def run(self) -> None:
        await self.ws.run(self.driver.stop)
