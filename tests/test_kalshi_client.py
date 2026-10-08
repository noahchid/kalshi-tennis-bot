import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_bot.client import KalshiAPIError, KalshiAuthError, KalshiClient


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "headers": headers})
        return self.responses.pop(0)


@pytest.fixture
def key_path(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    p = tmp_path / "k.pem"
    p.write_bytes(pem)
    return p, key.public_key()


def test_signed_request_signs_full_path_without_query(key_path):
    path, pub = key_path
    s = _Session([_Resp({"balance": 12345})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    assert c.get_balance() == 123.45
    call = s.calls[0]
    h = call["headers"]
    assert call["url"] == "https://x.test/trade-api/v2/portfolio/balance"
    assert h["KALSHI-ACCESS-KEY"] == "kid"
    msg = (h["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/v2/portfolio/balance").encode()
    pub.verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]),
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_balance_accepts_dollar_field(key_path):
    path, _ = key_path
    s = _Session([_Resp({"balance_dollars": "50.25"})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    assert c.get_balance() == 50.25


def test_missing_credentials_raise_auth_error():
    c = KalshiClient("", "", base_url="https://x.test", session=_Session([]))
    with pytest.raises(KalshiAuthError):
        c.get_balance()


def test_public_list_markets_paginates_without_auth():
    s = _Session([
        _Resp({"markets": [{"ticker": "A"}], "cursor": "c1"}),
        _Resp({"markets": [{"ticker": "B"}], "cursor": ""}),
    ])
    c = KalshiClient("", "", base_url="https://x.test", session=s)
    out = c.list_markets("KXATPMATCH", status="settled", min_close_ts=100)
    assert [m["ticker"] for m in out] == ["A", "B"]
    assert s.calls[0]["headers"] == {}
    assert s.calls[0]["params"]["series_ticker"] == "KXATPMATCH"
    assert s.calls[0]["params"]["min_close_ts"] == 100
    assert s.calls[1]["params"]["cursor"] == "c1"


def test_create_order_body(key_path):
    path, _ = key_path
    s = _Session([_Resp({"order": {"order_id": "o1", "fill_count": "3.00"}})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    out = c.create_order("KX-T-SMI", price=0.47, count=3, client_order_id="atpedge-abc")
    assert out["order_id"] == "o1"
    call = s.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://x.test/trade-api/v2/portfolio/events/orders"
    assert call["json"] == {
        "ticker": "KX-T-SMI",
        "side": "bid",
        "count": "3.00",
        "price": "0.47",
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": "atpedge-abc",
    }


def test_http_error_raises_api_error(key_path):
    path, _ = key_path
    s = _Session([_Resp({"error": "bad"}, status=400)])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    with pytest.raises(KalshiAPIError) as e:
        c.get_positions()
    assert e.value.status_code == 400


def test_balance_scoped_to_exchange_index(key_path):
    path, _ = key_path
    s = _Session([_Resp({"balance": 5000})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    assert c.get_balance(exchange_index=3) == 50.0
    assert s.calls[0]["params"] == {"exchange_index": 3}
