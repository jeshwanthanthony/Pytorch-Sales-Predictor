"""The Square connect flow, in Python.

This replaces server.mjs. Same OAuth dance, but living in the same app as the
dashboard so a user only ever visits one address.

The flow, in order:
  1. user clicks Connect        -> we send them to Square with a random state
  2. they approve at Square     -> Square sends them back with a code
  3. we swap the code for a token and save it
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.parse import urlparse

import httpx

from collector.config import REQUIRED_SCOPES, SQUARE_VERSION

log = logging.getLogger("oauth")

# Square serves OAuth and the v2 API from the same host in both environments.
SANDBOX_HOST = "https://connect.squareupsandbox.com"
PRODUCTION_HOST = "https://connect.squareup.com"

# Square access tokens last 30 days. refresh once we are inside this window,
# so a connection never silently dies between daily runs.
REFRESH_BEFORE_DAYS = 7


class OAuthError(RuntimeError):
    """Something went wrong connecting the Square account."""


@dataclass(frozen=True)
class OAuthConfig:
    application_id: str
    application_secret: str
    environment: str
    redirect_url: str

    @property
    def authorize_host(self) -> str:
        return PRODUCTION_HOST if self.environment == "production" else SANDBOX_HOST

    @property
    def api_host(self) -> str:
        return self.authorize_host

    @classmethod
    def from_env(cls) -> OAuthConfig:
        import os

        app_id = (os.environ.get("SQUARE_APPLICATION_ID") or "").strip()
        secret = (os.environ.get("SQUARE_APPLICATION_SECRET") or "").strip()
        if not app_id or not secret:
            raise OAuthError("Add SQUARE_APPLICATION_ID and SQUARE_APPLICATION_SECRET.")

        configured_environment = (os.environ.get("SQUARE_ENVIRONMENT") or "").strip().lower()
        if configured_environment and configured_environment not in {"sandbox", "production"}:
            raise OAuthError("SQUARE_ENVIRONMENT must be sandbox or production.")

        # Render users commonly enter only the two Square credentials. Infer the
        # matching host instead of silently sending a production app to sandbox.
        environment = configured_environment or (
            "sandbox" if app_id.startswith("sandbox-") else "production"
        )

        redirect_url = (os.environ.get("SQUARE_REDIRECT_URL") or "").strip()
        render_url = (os.environ.get("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")
        if not redirect_url and render_url:
            redirect_url = f"{render_url}/api/square/callback"
        if not redirect_url:
            redirect_url = "http://localhost:8080/api/square/callback"

        return cls(
            application_id=app_id,
            application_secret=secret,
            environment=environment,
            redirect_url=redirect_url,
        )

    def validation_errors(self) -> list[str]:
        """Configuration problems that would make Square reject OAuth."""
        errors: list[str] = []
        parsed = urlparse(self.redirect_url)
        app_id_is_sandbox = self.application_id.startswith("sandbox-")
        secret_is_sandbox = self.application_secret.startswith("sandbox-")

        if self.environment == "sandbox":
            if not app_id_is_sandbox or not secret_is_sandbox:
                errors.append("Sandbox mode needs the sandbox Application ID and Application Secret.")
        else:
            if app_id_is_sandbox or secret_is_sandbox:
                errors.append("Production mode needs production Square credentials, not sandbox credentials.")
            if parsed.scheme != "https":
                errors.append("Production Square OAuth needs an HTTPS redirect URL.")

        if parsed.path != "/api/square/callback":
            errors.append("SQUARE_REDIRECT_URL must end with /api/square/callback.")

        return errors

    def validation_warnings(self) -> list[str]:
        """Errors plus useful non-blocking notes for the setup screen."""
        warnings = self.validation_errors()
        if self.environment == "sandbox":
            warnings.append("Sandbox only connects Square sandbox seller test accounts.")

        return warnings


def authorize_url(config: OAuthConfig, state: str) -> str:
    """Where to send the user to approve access. The caller signs the state."""
    params = {
        "client_id": config.application_id,
        "scope": " ".join(REQUIRED_SCOPES),
        "state": state,
        "redirect_uri": config.redirect_url,
    }

    # session=false forces Square's own login page. Sandbox has no such page —
    # sandbox sellers only sign in through the Developer Console — so sending it
    # gets you "first launch the seller test account". Production needs it, so
    # the right seller signs in rather than whoever the browser remembers.
    if config.environment == "production":
        params["session"] = "false"

    return f"{config.authorize_host}/oauth2/authorize?{urlencode(params)}"


def exchange_code(config: OAuthConfig, code: str) -> dict:
    """Swap the one-time code for an access token.

    The state was already verified by the caller, which is where the signature,
    expiry and nonce checks live.
    """
    response = httpx.post(
        f"{config.api_host}/oauth2/token",
        headers={"Square-Version": SQUARE_VERSION},
        json={
            "client_id": config.application_id,
            "client_secret": config.application_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": config.redirect_url,
        },
        timeout=30,
    )
    try:
        body = response.json()
    except ValueError as exc:
        raise OAuthError(f"Square token exchange returned HTTP {response.status_code}.") from exc
    if response.status_code >= 400:
        detail = (body.get("errors") or [{}])[0].get("detail", "token exchange failed")
        raise OAuthError(detail)
    if not body.get("access_token"):
        raise OAuthError("Square did not return an access token.")
    return body


def fetch_locations(config: OAuthConfig, access_token: str) -> list[dict]:
    response = httpx.get(
        f"{config.api_host}/v2/locations",
        headers={"Authorization": f"Bearer {access_token}", "Square-Version": SQUARE_VERSION},
        timeout=30,
    )
    if response.status_code >= 400:
        return []
    return response.json().get("locations") or []


def refresh_token(config: OAuthConfig, refresh: str) -> dict:
    """Get a fresh access token before the old one expires.

    Square access tokens last 30 days. Without this a connection quietly stops
    working a month after it was made, which looks like a bug in the collector.
    """
    response = httpx.post(
        f"{config.api_host}/oauth2/token",
        headers={"Square-Version": SQUARE_VERSION},
        json={
            "client_id": config.application_id,
            "client_secret": config.application_secret,
            "refresh_token": refresh,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    body = response.json()
    if response.status_code >= 400:
        detail = (body.get("errors") or [{}])[0].get("detail", "could not refresh the token")
        raise OAuthError(detail)
    return body


def needs_refresh(expires_at: str | None) -> bool:
    """True when the token expires soon enough to be worth renewing now."""
    if not expires_at:
        return False
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return expiry - datetime.now(timezone.utc) < timedelta(days=REFRESH_BEFORE_DAYS)
