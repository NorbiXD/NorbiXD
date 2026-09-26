"""Async Bybit V5 REST client (signed endpoints used by the execution gateway)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

import httpx

from darwin.config.challenge import InstrumentSpec
from darwin.exchange.bybit.signing import Credentials, rest_headers
from darwin.exchange.bybit.ws import now_ms

log = logging.getLogger(__name__)

# retCodes that mean "the request definitively did not create/change anything"
DEFINITIVE_REJECT_CODES = {
    10001,  # params error
    10003,  # invalid api key
    10004,  # sign error
    10005,  # permission denied
    110003,  # price out of range
    110004,  # insufficient wallet balance
    110007,  # insufficient available balance
    110012,  # insufficient balance
    110017,  # reduce-only would increase
    110020,  # too many active orders
    110094,  # below min notional
}
DUPLICATE_LINK_ID = 110072  # orderLinkId is duplicate => the earlier attempt reached the exchange
LEVERAGE_NOT_MODIFIED = 110043
POSITION_MODE_NOT_MODIFIED = 110025


class BybitApiError(Exception):
    def __init__(self, ret_code: int, ret_msg: str, path: str) -> None:
        super().__init__(f"{path}: {ret_code} {ret_msg}")
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.path = path

    @property
    def definitive(self) -> bool:
        return self.ret_code in DEFINITIVE_REJECT_CODES


class BybitRest:
    def __init__(
        self,
        base_url: str,
        credentials: Credentials | None,
        recv_window_ms: int = 5_000,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock_ms: Callable[[], int] = now_ms,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials
        self.recv_window_ms = recv_window_ms
        self.clock_ms = clock_ms
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s, transport=transport)
        self.rate_limit_status: dict[str, str] = {}

    async def close(self) -> None:
        await self.client.aclose()

    async def _request(
        self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = True
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers: dict[str, str] = {}
        if method == "GET":
            query = urlencode(params)
            if signed:
                if self.credentials is None:
                    raise RuntimeError("signed endpoint requires credentials")
                headers = rest_headers(self.credentials, self.clock_ms(), self.recv_window_ms, query)
            resp = await self.client.get(path + (f"?{query}" if query else ""), headers=headers)
        else:
            body = json.dumps(params, separators=(",", ":"))
            if signed:
                if self.credentials is None:
                    raise RuntimeError("signed endpoint requires credentials")
                headers = rest_headers(self.credentials, self.clock_ms(), self.recv_window_ms, body)
            else:
                headers = {"Content-Type": "application/json"}
            resp = await self.client.post(path, content=body, headers=headers)
        for h in ("X-Bapi-Limit-Status", "X-Bapi-Limit", "X-Bapi-Limit-Reset-Timestamp"):
            if h in resp.headers:
                self.rate_limit_status[h] = resp.headers[h]
        resp.raise_for_status()
        payload = resp.json()
        code = int(payload.get("retCode", -1))
        if code != 0:
            raise BybitApiError(code, str(payload.get("retMsg")), path)
        return payload.get("result")

    # ------------------------------------------------------------------ market metadata
    async def instruments(self, symbols: list[str]) -> dict[str, InstrumentSpec]:
        out: dict[str, InstrumentSpec] = {}
        for sym in symbols:
            res = await self._request(
                "GET", "/v5/market/instruments-info", {"category": "linear", "symbol": sym}, signed=False
            )
            for it in res.get("list", []):
                lot = it.get("lotSizeFilter", {})
                pf = it.get("priceFilter", {})
                lev = it.get("leverageFilter", {})
                out[sym] = InstrumentSpec(
                    symbol=sym,
                    tick_size=float(pf["tickSize"]),
                    qty_step=float(lot["qtyStep"]),
                    min_qty=float(lot["minOrderQty"]),
                    min_notional=float(lot.get("minNotionalValue") or 5.0),
                    max_leverage=float(lev.get("maxLeverage") or 50),
                )
        return out

    async def server_time_ms(self) -> int:
        res = await self._request("GET", "/v5/market/time", signed=False)
        return int(res["timeNano"]) // 1_000_000 if "timeNano" in res else int(res["timeSecond"]) * 1000

    # ------------------------------------------------------------------ trading
    async def place_order(self, **order: Any) -> dict[str, Any]:
        res: dict[str, Any] = await self._request("POST", "/v5/order/create", {"category": "linear", **order})
        return res

    async def cancel_order(self, symbol: str, order_link_id: str) -> dict[str, Any]:
        res: dict[str, Any] = await self._request(
            "POST", "/v5/order/cancel", {"category": "linear", "symbol": symbol, "orderLinkId": order_link_id}
        )
        return res

    async def order_by_link_id(self, symbol: str, order_link_id: str) -> dict[str, Any] | None:
        """Look in open/recent orders first, then order history."""
        for path in ("/v5/order/realtime", "/v5/order/history"):
            res = await self._request(
                "GET", path, {"category": "linear", "symbol": symbol, "orderLinkId": order_link_id}
            )
            rows = res.get("list", [])
            if rows:
                row: dict[str, Any] = rows[0]
                return row
        return None

    async def executions_by_link_id(self, symbol: str, order_link_id: str) -> list[dict[str, Any]]:
        res = await self._request(
            "GET",
            "/v5/execution/list",
            {"category": "linear", "symbol": symbol, "orderLinkId": order_link_id},
        )
        rows: list[dict[str, Any]] = res.get("list", [])
        return rows

    async def open_orders(self) -> list[dict[str, Any]]:
        res = await self._request("GET", "/v5/order/realtime", {"category": "linear", "settleCoin": "USDT"})
        rows: list[dict[str, Any]] = res.get("list", [])
        return rows

    async def positions(self) -> list[dict[str, Any]]:
        res = await self._request("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
        rows: list[dict[str, Any]] = res.get("list", [])
        return rows

    async def wallet(self) -> dict[str, Any] | None:
        res = await self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        rows = res.get("list", [])
        return rows[0] if rows else None

    async def account_info(self) -> dict[str, Any]:
        res: dict[str, Any] = await self._request("GET", "/v5/account/info", {})
        return res

    async def set_one_way_mode(self, coin: str = "USDT") -> None:
        try:
            await self._request(
                "POST", "/v5/position/switch-mode", {"category": "linear", "coin": coin, "mode": 0}
            )
        except BybitApiError as e:
            if e.ret_code != POSITION_MODE_NOT_MODIFIED:
                raise

    async def set_leverage(self, symbol: str, leverage: float) -> None:
        lev = f"{leverage:g}"
        try:
            await self._request(
                "POST",
                "/v5/position/set-leverage",
                {"category": "linear", "symbol": symbol, "buyLeverage": lev, "sellLeverage": lev},
            )
        except BybitApiError as e:
            if e.ret_code != LEVERAGE_NOT_MODIFIED:
                raise
