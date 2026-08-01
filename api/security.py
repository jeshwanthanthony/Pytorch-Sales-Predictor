"""Signing, OAuth state, and rate limits.

Three problems this solves, all of which the app had:

  1. the session cookie was a plain merchant id, so anyone could type in
     somebody else's and read their sales
  2. the OAuth state lived in a set in memory: not signed, never expired, and
     gone the moment the server restarted
  3. nothing stopped a script hammering the connect or training routes

Everything is signed with HMAC-SHA256 using one app secret. No dependency
needed — hmac and secrets are in the standard library.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import time
from base64 import urlsafe_b64encode
from collections import defaultdict, deque
from hashlib import sha256
from pathlib import Path

log = logging.getLogger("security")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRET_FILE = PROJECT_ROOT / ".app-secret"

# an oauth handshake that takes longer than this is not a real one
OAUTH_STATE_MAX_AGE = 10 * 60
# how long a browser stays signed in as one restaurant
SESSION_MAX_AGE = 30 * 24 * 60 * 60


class SecurityError(RuntimeError):
    """A value failed its signature, age, or single-use check."""


def app_secret() -> bytes:
    """The key everything is signed with.

    From APP_SECRET if set. Otherwise generated once and kept in a gitignored
    file, so restarting the server does not sign everyone out.
    """
    from_env = os.environ.get("APP_SECRET")
    if from_env:
        return from_env.encode()

    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes().strip()

    generated = secrets.token_urlsafe(48).encode()
    SECRET_FILE.write_bytes(generated)
    SECRET_FILE.chmod(0o600)
    log.info("generated a new app secret at %s", SECRET_FILE.name)
    return generated


def _signature(payload: str) -> str:
    digest = hmac.new(app_secret(), payload.encode(), sha256).digest()
    return urlsafe_b64encode(digest).decode().rstrip("=")


def sign(value: str) -> str:
    """value -> 'value.timestamp.signature'"""
    payload = f"{value}.{int(time.time())}"
    return f"{payload}.{_signature(payload)}"


def unsign(signed: str | None, max_age: int | None = None) -> str:
    """Recover the value, or raise. Never returns something unverified."""
    if not signed:
        raise SecurityError("nothing to verify")

    parts = signed.rsplit(".", 2)
    if len(parts) != 3:
        raise SecurityError("malformed signed value")

    value, issued, provided = parts
    payload = f"{value}.{issued}"

    # constant time, so a wrong signature cannot be guessed a byte at a time
    if not hmac.compare_digest(provided, _signature(payload)):
        raise SecurityError("signature does not match")

    try:
        age = time.time() - int(issued)
    except ValueError as exc:
        raise SecurityError("bad timestamp") from exc

    if max_age is not None and age > max_age:
        raise SecurityError("expired")

    return value


# -- oauth state ------------------------------------------------------------

# nonces already spent. an oauth callback may only be used once.
_consumed: set[str] = set()


def new_oauth_state() -> tuple[str, str]:
    """A signed state for Square, plus the nonce to store in a cookie.

    Square echoes the state back. The nonce never leaves this machine except as
    an HttpOnly cookie, so a callback forged by another site cannot match.
    """
    nonce = secrets.token_urlsafe(24)
    return sign(nonce), nonce


def consume_oauth_state(state: str | None, nonce_cookie: str | None) -> None:
    """Check the state came from us, is fresh, matches the cookie, and is unused."""
    nonce = unsign(state, max_age=OAUTH_STATE_MAX_AGE)

    if not nonce_cookie or not hmac.compare_digest(nonce, nonce_cookie):
        raise SecurityError("this callback did not start in this browser")

    if nonce in _consumed:
        raise SecurityError("this callback was already used")
    _consumed.add(nonce)

    # the set would grow forever otherwise, and old nonces have expired anyway
    if len(_consumed) > 1000:
        _consumed.clear()


# -- sessions ---------------------------------------------------------------


def sign_session(merchant_id: str) -> str:
    return sign(merchant_id)


def read_session(cookie: str | None) -> str | None:
    """The merchant id this browser proved it owns, or None."""
    try:
        return unsign(cookie, max_age=SESSION_MAX_AGE)
    except SecurityError:
        return None


# -- rate limits ------------------------------------------------------------

_hits: dict[str, deque[float]] = defaultdict(deque)


def rate_limit(key: str, limit: int, per_seconds: int) -> None:
    """Allow `limit` calls per window, per key. Raises when over."""
    now = time.time()
    window = _hits[key]

    while window and now - window[0] > per_seconds:
        window.popleft()

    if len(window) >= limit:
        raise SecurityError(f"too many attempts, wait {per_seconds}s")
    window.append(now)


def reset_rate_limits() -> None:
    _hits.clear()
