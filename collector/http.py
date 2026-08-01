"""A small HTTP client with the two behaviours every API pull needs:
retry on transient failure, and cursor pagination.

Square rate-limits per-endpoint and answers 429 with no Retry-After, so the
backoff is exponential with jitter rather than header-driven.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Iterator

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5


class ApiError(RuntimeError):
    """A non-retryable error response, with Square's detail message unwrapped."""

    def __init__(self, status: int, body: Any, url: str):
        self.status = status
        self.body = body
        detail = ""
        if isinstance(body, dict):
            errors = body.get("errors") or []
            if errors:
                first = errors[0]
                detail = f" {first.get('code', '')}: {first.get('detail', '')}"
            elif body.get("reason"):
                # open-meteo sends errors as {"error": true, "reason": "..."}
                detail = f" {body['reason']}"
        super().__init__(f"HTTP {status} from {url}.{detail}")


class HttpClient:
    """Thin wrapper over httpx.Client. Use as a context manager."""

    def __init__(self, base_url: str = "", headers: dict[str, str] | None = None, timeout: float = 30.0):
        self._client = httpx.Client(base_url=base_url, headers=headers or {}, timeout=timeout)

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.request(method, path, **kwargs)
            except httpx.TransportError as exc:  # DNS, connection reset, read timeout
                last_error = exc
                log.warning("%s %s failed (%s), attempt %d/%d", method, path, exc, attempt, MAX_ATTEMPTS)
                self._sleep(attempt)
                continue

            if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                log.warning(
                    "%s %s -> %d, attempt %d/%d", method, path, response.status_code, attempt, MAX_ATTEMPTS
                )
                self._sleep(attempt)
                continue

            body = self._parse(response)
            if response.status_code >= 400:
                raise ApiError(response.status_code, body, str(response.request.url))
            return body

        raise ApiError(0, {"errors": [{"detail": str(last_error)}]}, path)

    def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("POST", path, **kwargs)

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        try:
            return response.json()
        except ValueError:
            return {"errors": [{"detail": response.text[:400]}]}

    @staticmethod
    def _sleep(attempt: int) -> None:
        time.sleep(min(2**attempt, 30) * (0.5 + random.random() / 2))


def paginate(
    fetch: Callable[[str | None], dict[str, Any]],
    key: str,
    cursor_field: str = "cursor",
) -> Iterator[dict[str, Any]]:
    """Walk a Square cursor-paginated endpoint.

    `fetch` takes a cursor (None on the first call) and returns the parsed body;
    `key` is the array field to yield from ("orders", "payments", ...).
    """
    cursor: str | None = None
    page = 0
    while True:
        body = fetch(cursor)
        page += 1
        rows = body.get(key) or []
        log.debug("page %d: %d %s", page, len(rows), key)
        yield from rows

        cursor = body.get(cursor_field)
        if not cursor:
            return
