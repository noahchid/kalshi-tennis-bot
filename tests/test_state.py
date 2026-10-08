import base64

import pytest

from kalshi_bot.state import StateError, StateStore


class _Resp:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "headers": headers})
        return self.responses.pop(0)


def _store(responses):
    s = _Session(responses)
    return StateStore("owner/private", "bot-state", "tok", session=s), s


def test_get_small_file_decodes_base64():
    store, s = _store([_Resp(payload={"sha": "abc", "size": 5,
                                      "content": base64.b64encode(b"hello").decode(), "encoding": "base64"})])
    assert store.get("plays.json") == (b"hello", "abc")
    call = s.calls[0]
    assert call["url"] == "https://api.github.com/repos/owner/private/contents/plays.json"
    assert call["params"] == {"ref": "bot-state"}
    assert call["headers"]["Authorization"] == "Bearer tok"


def test_get_missing_file_returns_none():
    store, _ = _store([_Resp(status=404)])
    assert store.get("kalshi_bets.parquet") == (None, None)


def test_get_large_file_falls_back_to_raw():
    store, s = _store([_Resp(payload={"sha": "big", "size": 2_000_000, "content": "", "encoding": "none"}),
                       _Resp(content=b"raw-bytes")])
    assert store.get("kalshi_bets.parquet") == (b"raw-bytes", "big")
    assert s.calls[1]["headers"]["Accept"] == "application/vnd.github.raw+json"


def test_put_sends_sha_and_branch():
    store, s = _store([_Resp(payload={"content": {"sha": "new"}})])
    assert store.put("kalshi_bets.parquet", b"data", sha="old", message="bot: bets") == "new"
    body = s.calls[0]["json"]
    assert s.calls[0]["method"] == "PUT"
    assert body == {"message": "bot: bets", "content": base64.b64encode(b"data").decode(),
                    "branch": "bot-state", "sha": "old"}


def test_put_new_file_omits_sha():
    store, s = _store([_Resp(status=201, payload={"content": {"sha": "first"}})])
    assert store.put("kalshi_bankroll.json", b"{}", sha=None, message="m") == "first"
    assert "sha" not in s.calls[0]["json"]


def test_errors_carry_status_only():
    store, _ = _store([_Resp(status=409, payload={"message": "sha mismatch"})])
    with pytest.raises(StateError) as e:
        store.put("x", b"1", sha="stale", message="m")
    assert e.value.status_code == 409
    assert "sha mismatch" not in str(e.value)


def test_requires_repo_and_token():
    with pytest.raises(StateError):
        StateStore("", "bot-state", "tok")
    with pytest.raises(StateError):
        StateStore("owner/private", "bot-state", "")


def test_check_branch_ok_and_missing():
    store, s = _store([_Resp(status=200, payload={"name": "bot-state"})])
    store.check_branch()
    assert s.calls[0]["url"] == "https://api.github.com/repos/owner/private/branches/bot-state"
    store, _ = _store([_Resp(status=404)])
    with pytest.raises(StateError) as e:
        store.check_branch()
    assert e.value.status_code == 404
