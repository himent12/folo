"""The few calls of docs/API.md that run_folo.py makes, with `requests` and a custom User-Agent (RunPod's proxy
blocks python-urllib's).

A request that got no answer (connection refused or reset, a timeout) raises `TransportError`, one the server
answered with an error status raises `ApiError`. Both may be retried for requests that can be repeated (GETs, a
stop); a request that creates something (`POST /api/runs`) is sent once (`retries=1`), since a lost answer or a 5xx
does not say whether the server created it: `ambiguous()` tells such a failure apart, and the caller looks the
thing up before it sends the request again.
"""

from __future__ import annotations

import json
import time

USER_AGENT = "folo-bench-finetune/1"


class ApiError(RuntimeError):
    """The server answered with an error status (`code`: docs/API.md's error code, when the body has one)."""

    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:500]}")
        self.status = status
        try:
            self.code = json.loads(text)["error"]["code"]
        except Exception:
            self.code = None


class TransportError(RuntimeError):
    """No answer from the server: the request may or may not have reached it."""


def ambiguous(e: BaseException) -> bool:
    """A failure after which the request may have taken effect: no answer, or a server error (5xx)."""
    return isinstance(e, TransportError) or (isinstance(e, ApiError) and e.status >= 500)


class Api:
    def __init__(self, base: str, token: str | None = None, timeout_s: float = 60, sleep=time.sleep):
        import requests

        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"
        self.timeout_s = timeout_s
        self.sleep = sleep

    def call(self, method: str, path: str, body=None, ok=(200, 201, 204), retries: int = 5):
        """`method path` with a JSON body; up to `retries` attempts in all for no answer or a 5xx (1: sent once).
        Raises the last failure: `TransportError` or `ApiError`."""
        last: Exception | None = None
        for attempt in range(max(1, retries)):
            if attempt:
                self.sleep(min(30, 2 ** (attempt - 1)))
            try:
                r = self.s.request(method, self.base + path, json=body, timeout=self.timeout_s)
            except Exception as e:  # connection refused or reset, a timeout
                last = TransportError(f"{method} {path}: no answer: {e}")
                continue
            if r.status_code in ok:
                return r.json() if r.content else None
            if r.status_code >= 500:
                last = ApiError(r.status_code, r.text)
                continue
            raise ApiError(r.status_code, r.text)
        assert last is not None
        raise last

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body=None):
        return self.call("POST", path, body)
