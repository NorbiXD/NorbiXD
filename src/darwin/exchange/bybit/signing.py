"""Bybit V5 request signing (HMAC-SHA256)."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Credentials:
    api_key: str
    api_secret: str

    def __repr__(self) -> str:  # never leak secrets into logs
        return f"Credentials(api_key={self.api_key[:4]}***)"

    @staticmethod
    def from_env(prefix: str = "BYBIT") -> Credentials | None:
        key = os.environ.get(f"{prefix}_API_KEY")
        secret = os.environ.get(f"{prefix}_API_SECRET")
        if not key or not secret:
            return None
        return Credentials(key, secret)


def rest_signature(creds: Credentials, timestamp_ms: int, recv_window_ms: int, payload: str) -> str:
    """``payload`` is the query string for GET, the exact JSON body for POST."""
    msg = f"{timestamp_ms}{creds.api_key}{recv_window_ms}{payload}"
    return hmac.new(creds.api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def rest_headers(creds: Credentials, timestamp_ms: int, recv_window_ms: int, payload: str) -> dict[str, str]:
    return {
        "X-BAPI-API-KEY": creds.api_key,
        "X-BAPI-TIMESTAMP": str(timestamp_ms),
        "X-BAPI-RECV-WINDOW": str(recv_window_ms),
        "X-BAPI-SIGN": rest_signature(creds, timestamp_ms, recv_window_ms, payload),
        "Content-Type": "application/json",
    }


def ws_auth_args(creds: Credentials, expires_ms: int) -> list[str | int]:
    sig = hmac.new(
        creds.api_secret.encode(), f"GET/realtime{expires_ms}".encode(), hashlib.sha256
    ).hexdigest()
    return [creds.api_key, expires_ms, sig]
