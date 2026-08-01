"""Credentials, paths, and the restaurant's physical location.

Everything the collector needs to know about *where* data comes from lives here,
so no other module reads os.environ or guesses a file path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# where the downloaded files go, one folder per thing we collect
RAW_DIR = ROOT / "data" / "raw"
# remembers the newest record we saw, so we only ask for new stuff next time
STATE_FILE = ROOT / "data" / "collector-state.json"
# CSVs you fill in yourself (events, promos, school breaks)
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

TOKEN_FILE = ROOT / ".square-tokens.json"

# pin the api version so square cannot change the response shape on us
SQUARE_VERSION = "2025-01-23"

# server.mjs must ask for all of these or the matching pull gets a 403
REQUIRED_SCOPES = [
    "MERCHANT_PROFILE_READ",  # locations
    "ORDERS_READ",            # orders + order line items
    "PAYMENTS_READ",          # payments + refunds
    "ITEMS_READ",             # catalog: items, categories, modifiers
    "CUSTOMERS_READ",         # customer directory
    "INVENTORY_READ",         # stock counts
    "EMPLOYEES_READ",         # team members
    "TIMECARDS_READ",         # labor shifts -> hours + labor cost
]


class ConfigError(RuntimeError):
    """Raised when the collector cannot run at all (no token, no location)."""


@dataclass(frozen=True)
class SquareAuth:
    access_token: str
    merchant_id: str | None
    environment: str
    location_ids: list[str]

    @property
    def api_host(self) -> str:
        if self.environment == "production":
            return "https://connect.squareup.com"
        return "https://connect.squareupsandbox.com"


def load_square_auth() -> SquareAuth:
    """Read the token written by the Node connect screen.

    An access token in the environment (SQUARE_ACCESS_TOKEN) wins, which is how
    you run the collector on a server that never served the OAuth page.
    """
    env_token = os.environ.get("SQUARE_ACCESS_TOKEN")
    environment = "production" if os.environ.get("SQUARE_ENVIRONMENT") == "production" else "sandbox"

    if env_token:
        ids = [s for s in os.environ.get("SQUARE_LOCATION_IDS", "").split(",") if s]
        return SquareAuth(env_token, os.environ.get("SQUARE_MERCHANT_ID"), environment, ids)

    if not TOKEN_FILE.exists():
        raise ConfigError(
            f"No Square token. Run `node server.mjs`, click Connect, then retry.\n"
            f"(expected {TOKEN_FILE})"
        )

    data = json.loads(TOKEN_FILE.read_text() or "{}")
    token = data.get("access_token")
    if not token:
        raise ConfigError(f"{TOKEN_FILE} has no access_token — reconnect via `node server.mjs`.")

    return SquareAuth(
        access_token=token,
        merchant_id=data.get("merchant_id"),
        environment=data.get("environment", environment),
        location_ids=[loc["id"] for loc in data.get("locations", []) if loc.get("id")],
    )


@dataclass(frozen=True)
class SiteConfig:
    """Where the restaurant physically is — drives weather and holiday lookups."""

    latitude: float
    longitude: float
    timezone: str
    country: str
    subdivision: str | None  # US state code, for state holidays

    @classmethod
    def from_env(cls) -> SiteConfig | None:
        lat, lon = os.environ.get("SITE_LATITUDE"), os.environ.get("SITE_LONGITUDE")
        if not lat or not lon:
            return None
        return cls(
            latitude=float(lat),
            longitude=float(lon),
            timezone=os.environ.get("SITE_TIMEZONE", "America/New_York"),
            country=os.environ.get("SITE_COUNTRY", "US"),
            subdivision=os.environ.get("SITE_STATE") or None,
        )
