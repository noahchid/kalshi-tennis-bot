"""Signed Kalshi v2 REST client (market data, portfolio, order placement).

Every authenticated request is signed with RSA-PSS over
``timestamp_ms + METHOD + path_without_query`` (path includes /trade-api/v2).
Market-data reads are public and sent without auth headers.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from kalshi_bot.config import (
    KALSHI_API_KEY_ID,
    KALSHI_API_PREFIX,
    KALSHI_BASE_URL,
    KALSHI_BASE_URLS,
    KALSHI_PRIVATE_KEY_PATH,
)


class KalshiAuthError(ValueError):
    """Credentials missing or unreadable."""


class KalshiAPIError(RuntimeError):
    """Non-2xx response from Kalshi."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"Kalshi API {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class KalshiClient:
    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        base_url: str | None = None,
        timeout: int = 30,
        session=None,
    ):
        self.base_url = (base_url or KALSHI_BASE_URL).rstrip("/")
        self.api_key_id = KALSHI_API_KEY_ID if api_key_id is None else api_key_id
        self.private_key_path = (
            KALSHI_PRIVATE_KEY_PATH if private_key_path is None else private_key_path
        )
        self.timeout = timeout
        self.session = session or requests.Session()
        self._private_key = None

    # ── auth ──────────────────────────────────────────────────────────────
    def _key(self):
        if self._private_key is None:
            if not self.api_key_id:
                raise KalshiAuthError("KALSHI_API_KEY_ID is not set")
            if not self.private_key_path:
                raise KalshiAuthError("KALSHI_PRIVATE_KEY_PATH is not set")
            path = Path(self.private_key_path).expanduser()
            if not path.exists():
                raise KalshiAuthError(f"Kalshi private key not found: {path}")
            self._private_key = serialization.load_pem_private_key(
                path.read_bytes(), password=None
            )
        return self._private_key

    def _auth_headers(self, method: str, full_path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method.upper()}{full_path.split('?', 1)[0]}".encode()
        key = self._key()
        # Kalshi picks the verifier from the registered public key: Ed25519 keys
        # sign the pre-sign text directly, RSA keys use RSA-PSS with SHA-256.
        if isinstance(key, ed25519.Ed25519PrivateKey):
            sig = key.sign(msg)
        elif isinstance(key, rsa.RSAPrivateKey):
            sig = key.sign(
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        else:
            raise KalshiAuthError("Kalshi private key must be RSA or Ed25519")
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict | None = None,
        auth: bool = True,
    ) -> dict:
        full_path = KALSHI_API_PREFIX + path
        headers = self._auth_headers(method, full_path) if auth else {}
        resp = self.session.request(
            method.upper(),
            self.base_url + full_path,
            params=params,
            json=json_body,
            headers=headers,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise KalshiAPIError(resp.status_code, resp.text)
        return resp.json()

    def _paginate(self, path: str, key: str, params: dict, auth: bool = True) -> list[dict]:
        out: list[dict] = []
        params = {k: v for k, v in params.items() if v is not None}
        params.setdefault("limit", 1000)
        while True:
            data = self.request("GET", path, params=dict(params), auth=auth)
            out.extend(data.get(key, []) or [])
            cursor = data.get("cursor")
            if not cursor:
                return out
            params["cursor"] = cursor

    # ── public market data ───────────────────────────────────────────────
    def list_markets(
        self, series_ticker: str, status: str = "open", min_close_ts: int | None = None
    ) -> list[dict]:
        return self._paginate(
            "/markets",
            "markets",
            {"series_ticker": series_ticker, "status": status, "min_close_ts": min_close_ts},
            auth=False,
        )

    # ── portfolio ────────────────────────────────────────────────────────
    def get_balance(self, exchange_index: int | None = None) -> float:
        """Available cash in dollars; scoped to one exchange shard when given."""
        params = {"exchange_index": exchange_index} if exchange_index is not None else None
        data = self.request("GET", "/portfolio/balance", params=params)
        if "balance_dollars" in data:
            return float(data["balance_dollars"])
        return float(data.get("balance", 0)) / 100.0

    def get_positions(self, event_ticker: str | None = None) -> list[dict]:
        return self._paginate(
            "/portfolio/positions",
            "market_positions",
            {"event_ticker": event_ticker, "count_filter": "position"},
        )

    def get_orders(self, event_ticker: str | None = None, status: str | None = None) -> list[dict]:
        return self._paginate(
            "/portfolio/orders", "orders", {"event_ticker": event_ticker, "status": status}
        )

    def get_settlements(self, min_ts: int | None = None) -> list[dict]:
        return self._paginate("/portfolio/settlements", "settlements", {"min_ts": min_ts})

    def get_fills(self, min_ts: int | None = None, ticker: str | None = None) -> list[dict]:
        return self._paginate("/portfolio/fills", "fills", {"min_ts": min_ts, "ticker": ticker})

    def create_order(
        self,
        ticker: str,
        price: float,
        count: int,
        client_order_id: str,
        *,
        time_in_force: str = "immediate_or_cancel",
        expiration_time: int | None = None,
        post_only: bool | None = None,
        cancel_order_on_pause: bool | None = None,
        stp: str = "taker_at_cross",
    ) -> dict:
        body = {
            "ticker": ticker,
            "side": "bid",
            "count": f"{int(count):.2f}",
            "price": f"{price:.2f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": stp,
            "client_order_id": client_order_id,
        }
        # Optional maker fields are added only when set, so a plain taker call
        # sends the same body as before.
        if expiration_time is not None:
            body["expiration_time"] = int(expiration_time)
        if post_only is not None:
            body["post_only"] = post_only
        if cancel_order_on_pause is not None:
            body["cancel_order_on_pause"] = cancel_order_on_pause
        data = self.request("POST", "/portfolio/events/orders", json_body=body)
        return data.get("order", data)

    def cancel_order(self, order_id: str) -> dict:
        # Path is unverified until the demo probe (Task 7).
        data = self.request("DELETE", f"/portfolio/orders/{order_id}")
        return data.get("order", data)


def public_client() -> KalshiClient:
    """Unauthenticated prod client for public market data (settled results)."""
    return KalshiClient(api_key_id="", private_key_path="", base_url=KALSHI_BASE_URLS["prod"])
