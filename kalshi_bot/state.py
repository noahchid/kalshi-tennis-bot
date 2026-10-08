"""Read/write bot state files on a branch of a private repo (GitHub contents API).

Each file has exactly one writer (the model pipeline writes plays.json; the bot
writes the bet log and bankroll), and every write carries the file's current
sha, so GitHub rejects a write based on a stale read instead of overwriting.
Errors expose only the HTTP status — never response bodies — because this runs
in a public repo's logs.
"""

from __future__ import annotations

import base64

import requests

API = "https://api.github.com"


class StateError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message if status_code is None else f"{message} (HTTP {status_code})")
        self.status_code = status_code


class StateStore:
    def __init__(self, repo: str, branch: str, token: str, session=None, timeout: int = 30):
        if not repo or not token:
            raise StateError("STATE_REPO and BOT_STATE_TOKEN must be set")
        self.repo = repo
        self.branch = branch
        self.token = token
        self.session = session or requests.Session()
        self.timeout = timeout

    def _call(self, method: str, path: str, accept: str = "application/vnd.github+json",
              endpoint: str = "contents", **kw):
        return self.session.request(
            method,
            f"{API}/repos/{self.repo}/{endpoint}/{path}",
            headers={"Authorization": f"Bearer {self.token}", "Accept": accept,
                     "X-GitHub-Api-Version": "2022-11-28"},
            timeout=self.timeout,
            **kw,
        )

    def check_branch(self) -> None:
        """Fail loudly if the state branch is unreachable (missing branch, wrong repo or
        a token without access all return 404) rather than reading it as 'no plays'."""
        resp = self._call("GET", self.branch, endpoint="branches")
        if resp.status_code >= 400:
            raise StateError("state branch unreachable", resp.status_code)

    def get(self, path: str) -> tuple[bytes | None, str | None]:
        """Return (content, sha), or (None, None) if the file doesn't exist."""
        resp = self._call("GET", path, params={"ref": self.branch})
        if resp.status_code == 404:
            return None, None
        if resp.status_code >= 400:
            raise StateError(f"reading {path} failed", resp.status_code)
        meta = resp.json()
        if meta.get("encoding") == "base64" and meta.get("content"):
            return base64.b64decode(meta["content"]), meta["sha"]
        # Files over 1 MB come back without content; fetch the raw bytes.
        raw = self._call("GET", path, accept="application/vnd.github.raw+json", params={"ref": self.branch})
        if raw.status_code >= 400:
            raise StateError(f"reading {path} failed", raw.status_code)
        return raw.content, meta["sha"]

    def put(self, path: str, data: bytes, sha: str | None, message: str) -> str:
        """Create or update a file; returns the new sha."""
        body = {"message": message, "content": base64.b64encode(data).decode(), "branch": self.branch}
        if sha:
            body["sha"] = sha
        resp = self._call("PUT", path, json=body)
        if resp.status_code >= 400:
            raise StateError(f"writing {path} failed", resp.status_code)
        return resp.json()["content"]["sha"]
