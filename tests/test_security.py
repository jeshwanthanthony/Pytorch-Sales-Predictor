"""Tests for signing, OAuth state, and rate limits.

The forged-cookie test is the important one: before the cookie was signed, one
restaurant could read another's sales by typing in their merchant id.
"""

from __future__ import annotations

import time

import pytest

from api import security
from api.security import (
    SecurityError,
    consume_oauth_state,
    new_oauth_state,
    rate_limit,
    read_session,
    reset_rate_limits,
    sign,
    sign_session,
    unsign,
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_SECRET", "test-secret-for-the-suite")
    security._consumed.clear()
    reset_rate_limits()
    yield
    security._consumed.clear()
    reset_rate_limits()


class TestSigning:
    def test_round_trip(self):
        assert unsign(sign("MERCHANT123")) == "MERCHANT123"

    def test_a_tampered_value_is_rejected(self):
        signed = sign("MERCHANT123")
        forged = signed.replace("MERCHANT123", "SOMEONEELSE")
        with pytest.raises(SecurityError, match="signature"):
            unsign(forged)

    def test_a_made_up_value_is_rejected(self):
        with pytest.raises(SecurityError):
            unsign("SOMEONEELSE.123456.notarealsignature")

    def test_a_signature_from_another_secret_is_rejected(self, monkeypatch):
        signed = sign("MERCHANT123")
        monkeypatch.setenv("APP_SECRET", "a-completely-different-secret")
        with pytest.raises(SecurityError, match="signature"):
            unsign(signed)

    def test_expiry_is_enforced(self, monkeypatch):
        signed = sign("MERCHANT123")
        # capture the real clock first, the lambda would call itself otherwise
        later = time.time() + 10_000
        monkeypatch.setattr(time, "time", lambda: later)
        with pytest.raises(SecurityError, match="expired"):
            unsign(signed, max_age=60)

    def test_nothing_is_not_a_signature(self):
        with pytest.raises(SecurityError):
            unsign(None)
        with pytest.raises(SecurityError):
            unsign("")


class TestSession:
    def test_a_signed_cookie_reads_back(self):
        assert read_session(sign_session("MERCHANT123")) == "MERCHANT123"

    def test_a_hand_typed_merchant_id_is_refused(self):
        """The whole reason the cookie is signed."""
        assert read_session("SOMEONEELSESMERCHANTID") is None

    def test_a_missing_cookie_is_refused(self):
        assert read_session(None) is None


class TestOAuthState:
    def test_a_matching_state_and_nonce_pass(self):
        state, nonce = new_oauth_state()
        consume_oauth_state(state, nonce)  # does not raise

    def test_a_state_can_only_be_used_once(self):
        state, nonce = new_oauth_state()
        consume_oauth_state(state, nonce)
        with pytest.raises(SecurityError, match="already used"):
            consume_oauth_state(state, nonce)

    def test_a_state_without_the_browser_cookie_fails(self):
        """This is the CSRF case: someone else's callback in your browser."""
        state, _ = new_oauth_state()
        with pytest.raises(SecurityError, match="did not start in this browser"):
            consume_oauth_state(state, None)

    def test_a_mismatched_nonce_fails(self):
        state, _ = new_oauth_state()
        _, other_nonce = new_oauth_state()
        with pytest.raises(SecurityError, match="did not start in this browser"):
            consume_oauth_state(state, other_nonce)

    def test_an_unsigned_state_fails(self):
        with pytest.raises(SecurityError):
            consume_oauth_state("just-some-string", "just-some-string")

    def test_a_stale_state_fails(self, monkeypatch):
        state, nonce = new_oauth_state()
        later = time.time() + security.OAUTH_STATE_MAX_AGE + 60
        monkeypatch.setattr(time, "time", lambda: later)
        with pytest.raises(SecurityError, match="expired"):
            consume_oauth_state(state, nonce)


class TestRateLimit:
    def test_allows_up_to_the_limit(self):
        for _ in range(3):
            rate_limit("someone", limit=3, per_seconds=60)

    def test_blocks_past_the_limit(self):
        for _ in range(3):
            rate_limit("someone", limit=3, per_seconds=60)
        with pytest.raises(SecurityError, match="too many"):
            rate_limit("someone", limit=3, per_seconds=60)

    def test_keys_do_not_share_a_budget(self):
        for _ in range(3):
            rate_limit("first", limit=3, per_seconds=60)
        rate_limit("second", limit=3, per_seconds=60)  # unaffected

    def test_the_window_slides(self, monkeypatch):
        for _ in range(3):
            rate_limit("someone", limit=3, per_seconds=60)
        later = time.time() + 120
        monkeypatch.setattr(time, "time", lambda: later)
        rate_limit("someone", limit=3, per_seconds=60)  # old hits aged out


class TestTokenRefresh:
    def test_a_token_expiring_soon_is_refreshed(self):
        from datetime import datetime, timedelta, timezone

        from api.square_oauth import needs_refresh

        soon = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        assert needs_refresh(soon) is True

    def test_a_fresh_token_is_left_alone(self):
        from datetime import datetime, timedelta, timezone

        from api.square_oauth import needs_refresh

        later = (datetime.now(timezone.utc) + timedelta(days=25)).isoformat()
        assert needs_refresh(later) is False

    def test_no_expiry_means_no_refresh(self):
        from api.square_oauth import needs_refresh

        assert needs_refresh(None) is False
        assert needs_refresh("not a date") is False
