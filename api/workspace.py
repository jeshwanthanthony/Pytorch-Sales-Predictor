"""One folder per restaurant.

Every Square account that connects gets its own workspace, keyed by Square's
merchant id. Nothing is shared between them — separate raw files, separate
database, separate model. That is what makes this multi tenant rather than a
demo with one hardcoded account.

    workspaces/
      ML4X8K2N9P/            <- merchant id from Square
        square.json          <- their access token, mode 600
        raw/                 <- what we downloaded
        forecast.db          <- their warehouse
        features/            <- their training data
        models/              <- their trained model
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from collector.config import SquareAuth

log = logging.getLogger("workspace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACES_DIR = Path(os.environ.get("WORKSPACES_DIR", PROJECT_ROOT / "workspaces"))


class WorkspaceError(RuntimeError):
    """Something is wrong with this restaurant's workspace."""


@dataclass(frozen=True)
class Workspace:
    """Where one restaurant's data lives."""

    merchant_id: str

    @property
    def root(self) -> Path:
        return WORKSPACES_DIR / self.merchant_id

    @property
    def token_file(self) -> Path:
        return self.root / "square.json"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def state_file(self) -> Path:
        return self.root / "collector-state.json"

    @property
    def pipeline_state_file(self) -> Path:
        return self.root / "pipeline-state.json"

    @property
    def db_path(self) -> Path:
        return self.root / "forecast.db"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def dataset_file(self) -> Path:
        return self.features_dir / "dataset.npz"

    @property
    def manifest_file(self) -> Path:
        return self.features_dir / "manifest.json"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def checkpoint(self) -> Path:
        return self.models_dir / "model.pt"

    def ensure(self) -> Workspace:
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    # -- the connected account ---------------------------------------------

    def save_token(self, tokens: dict, locations: list[dict], environment: str) -> None:
        self.ensure()
        self.token_file.write_text(
            json.dumps(
                {
                    **tokens,
                    "environment": environment,
                    "locations": [
                        {
                            "id": loc.get("id"),
                            "name": loc.get("name"),
                            "currency": loc.get("currency"),
                            "timezone": loc.get("timezone"),
                        }
                        for loc in locations
                    ],
                    "business_name": _business_name(locations),
                    "connected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )
        # the token is a credential, keep it to this user
        self.token_file.chmod(0o600)

    def token(self) -> dict:
        if not self.token_file.exists():
            raise WorkspaceError("this restaurant has not connected Square")
        data = json.loads(self.token_file.read_text() or "{}")
        if not data.get("access_token"):
            raise WorkspaceError("no access token stored, connect Square again")
        return data

    def refresh_if_needed(self) -> None:
        """Renew the Square token if it is close to expiring.

        Square tokens last 30 days. Without this the collector starts failing a
        month after connecting, which looks like a bug rather than an expiry.
        """
        from .square_oauth import OAuthConfig, OAuthError, needs_refresh, refresh_token

        data = self.token()
        if not needs_refresh(data.get("expires_at")) or not data.get("refresh_token"):
            return

        try:
            fresh = refresh_token(OAuthConfig.from_env(), data["refresh_token"])
        except OAuthError as exc:
            log.warning("could not refresh %s: %s", self.merchant_id, exc)
            return

        data.update(fresh)
        data["refreshed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.token_file.write_text(json.dumps(data, indent=2))
        self.token_file.chmod(0o600)
        log.info("refreshed the Square token for %s", self.merchant_id)

    def auth(self) -> SquareAuth:
        self.refresh_if_needed()
        data = self.token()
        return SquareAuth(
            access_token=data["access_token"],
            merchant_id=self.merchant_id,
            environment=data.get("environment", "sandbox"),
            location_ids=[loc["id"] for loc in data.get("locations", []) if loc.get("id")],
        )

    def is_connected(self) -> bool:
        try:
            self.token()
            return True
        except WorkspaceError:
            return False

    def has_model(self) -> bool:
        return self.checkpoint.exists()

    def summary(self) -> dict:
        """What we know about this restaurant, for the page header."""
        try:
            data = self.token()
        except WorkspaceError:
            return {"connected": False}

        return {
            "connected": True,
            "merchant_id": self.merchant_id,
            "business_name": data.get("business_name"),
            "environment": data.get("environment"),
            "locations": data.get("locations", []),
            "connected_at": data.get("connected_at"),
            "has_model": self.has_model(),
        }

    def delete(self) -> None:
        """Disconnect and remove everything we hold for this restaurant."""
        if self.root.exists():
            shutil.rmtree(self.root)
        log.info("deleted workspace %s", self.merchant_id)


def _business_name(locations: list[dict]) -> str | None:
    for loc in locations:
        name = loc.get("business_name") or loc.get("name")
        if name:
            return name
    return None


def get(merchant_id: str) -> Workspace:
    if not merchant_id or "/" in merchant_id or ".." in merchant_id:
        raise WorkspaceError(f"bad merchant id: {merchant_id!r}")
    return Workspace(merchant_id)


def list_connected() -> list[str]:
    if not WORKSPACES_DIR.exists():
        return []
    return sorted(
        path.name for path in WORKSPACES_DIR.iterdir()
        if path.is_dir() and (path / "square.json").exists()
    )
