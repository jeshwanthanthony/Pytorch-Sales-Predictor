"""Start the app on the local port used by the Square callback.

    python -m api.serve

Square rejects the connection unless the redirect_uri we send matches the one
registered in the Developer Dashboard byte for byte. Production OAuth needs an
HTTPS redirect URL, which is usually a tunnel during local development. The app
still binds to PORT locally and the tunnel forwards public HTTPS traffic to it.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

import uvicorn
from dotenv import load_dotenv

from .square_oauth import OAuthConfig, OAuthError

DEFAULT_PORT = 8080


def redirect_parts() -> tuple[str, int, str]:
    """host, port and path of the registered redirect URL."""
    load_dotenv()
    url = os.environ.get("SQUARE_REDIRECT_URL", f"http://localhost:{DEFAULT_PORT}/api/square/callback")
    parsed = urlparse(url)
    port = int(os.environ.get("PORT") or parsed.port or DEFAULT_PORT)
    return parsed.hostname or "localhost", port, parsed.path


def main() -> int:
    host, port, path = redirect_parts()

    try:
        config = OAuthConfig.from_env()
    except OAuthError as exc:
        print(f"cannot start: {exc}")
        return 1

    if path != "/api/square/callback":
        print(f"warning: SQUARE_REDIRECT_URL path is {path!r}, the app serves /api/square/callback")

    print(f"\n  environment  {config.environment}")
    print(f"  redirect     {config.redirect_url}")
    print("               ^ this must match the Redirect URL in your Square app, exactly")
    print(f"\n  open         http://localhost:{port}\n")

    # bind locally only, this app has no authentication yet
    uvicorn.run("api.main:app", host="127.0.0.1", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
