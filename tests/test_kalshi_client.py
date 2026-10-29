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


def test_signed_request_with_ed25519_key(tmp_path):
    # Kalshi also issues (and recommends) Ed25519 keys: same headers and pre-sign
    # text, but the key signs the text directly (no PSS padding / hash argument).
    from cryptography.hazmat.primitives.asymmetric import ed25519
    key = ed25519.Ed25519PrivateKey.generate()
    path = tmp_path / "demokey.txt"  # Kalshi's download is PKCS#8 PEM text, whatever the extension
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    s = _Session([_Resp({"balance": 500})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    assert c.get_balance() == 5.0
    h = s.calls[0]["headers"]
    msg = (h["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/v2/portfolio/balance").encode()
    key.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg)


def test_unsupported_key_type_is_an_auth_error(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ec
    key = ec.generate_private_key(ec.SECP256R1())
    path = tmp_path / "k.pem"
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=_Session([_Resp({})]))
    with pytest.raises(KalshiAuthError, match="RSA or Ed25519"):
        c.get_balance()


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


def test_create_order_default_body_unchanged(key_path):
    path, _ = key_path
    s = _Session([_Resp({"order": {"order_id": "o1"}})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    c.create_order("KX-T-SMI", 0.47, 3, "atpedge-abc")
    assert s.calls[0]["json"] == {
        "ticker": "KX-T-SMI",
        "side": "bid",
        "count": "3.00",
        "price": "0.47",
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": "atpedge-abc",
    }


def test_create_order_maker_body(key_path):
    path, _ = key_path
    s = _Session([_Resp({"order": {"order_id": "o2"}})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    out = c.create_order(
        "KX-T-SMI",
        0.42,
        5,
        "atpedge-mk-abc",
        time_in_force="good_till_canceled",
        expiration_time=1793000000,
        post_only=True,
        cancel_order_on_pause=True,
        stp="maker",
    )
    assert out["order_id"] == "o2"
    call = s.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://x.test/trade-api/v2/portfolio/events/orders"
    assert call["json"] == {
        "ticker": "KX-T-SMI",
        "side": "bid",
        "count": "5.00",
        "price": "0.42",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "maker",
        "client_order_id": "atpedge-mk-abc",
        "expiration_time": 1793000000,
        "post_only": True,
        "cancel_order_on_pause": True,
    }


def test_cancel_order_path(key_path):
    path, _ = key_path
    s = _Session([_Resp({"order": {"order_id": "o9", "status": "canceled"}})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    out = c.cancel_order("o9")
    call = s.calls[0]
    assert call["method"] == "DELETE"
    assert call["url"] == "https://x.test/trade-api/v2/portfolio/orders/o9"
    assert call["json"] is None
    assert out["order_id"] == "o9"
    assert call["headers"]["KALSHI-ACCESS-KEY"] == "kid"


def test_get_fills_paginates(key_path):
    path, _ = key_path
    s = _Session([
        _Resp({"fills": [{"fill_id": "f1"}], "cursor": "c1"}),
        _Resp({"fills": [{"fill_id": "f2"}], "cursor": ""}),
    ])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    out = c.get_fills(min_ts=100, ticker="KX-T-SMI")
    assert [f["fill_id"] for f in out] == ["f1", "f2"]
    assert s.calls[0]["method"] == "GET"
    assert s.calls[0]["url"] == "https://x.test/trade-api/v2/portfolio/fills"
    assert s.calls[0]["params"]["min_ts"] == 100
    assert s.calls[0]["params"]["ticker"] == "KX-T-SMI"
    assert s.calls[1]["params"]["cursor"] == "c1"
    s2 = _Session([_Resp({"fills": []})])
    c2 = KalshiClient("kid", str(path), base_url="https://x.test", session=s2)
    assert c2.get_fills() == []
    assert "min_ts" not in s2.calls[0]["params"] and "ticker" not in s2.calls[0]["params"]


def test_rate_limited_request_is_retried_with_backoff(key_path, monkeypatch):
    # Kalshi answers a burst with 429 too_many_requests; the request was not processed,
    # so it is retried (re-signed, fresh timestamp) after a short backoff.
    path, _ = key_path
    slept = []
    monkeypatch.setattr("kalshi_bot.client.time.sleep", lambda s: slept.append(s))
    s = _Session([_Resp({"error": {"code": "too_many_requests"}}, status=429),
                  _Resp({"error": {"code": "too_many_requests"}}, status=429),
                  _Resp({"balance": 100})])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    assert c.get_balance() == 1.0
    assert len(s.calls) == 3 and slept == [1.0, 2.0]


def test_rate_limit_gives_up_after_bounded_retries(key_path, monkeypatch):
    path, _ = key_path
    monkeypatch.setattr("kalshi_bot.client.time.sleep", lambda s: None)
    s = _Session([_Resp({"error": {}}, status=429)] * 5)
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    with pytest.raises(KalshiAPIError) as e:
        c.get_balance()
    assert e.value.status_code == 429 and len(s.calls) == 4


def test_other_errors_are_not_retried(key_path, monkeypatch):
    path, _ = key_path
    monkeypatch.setattr("kalshi_bot.client.time.sleep", lambda s: pytest.fail("no retry on 500"))
    s = _Session([_Resp({"error": {}}, status=500)])
    c = KalshiClient("kid", str(path), base_url="https://x.test", session=s)
    with pytest.raises(KalshiAPIError):
        c.get_balance()
    assert len(s.calls) == 1
