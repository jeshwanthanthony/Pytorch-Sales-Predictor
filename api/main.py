"""Connect your Square account, see your data, predict tomorrow's sales.

    uvicorn api.main:app --port 8080
    http://localhost:8080

Multi tenant: every restaurant that connects gets its own workspace, its own
database and its own model. A browser cookie remembers which one you are.

Nothing is shown before you connect. There is no demo data on the page.

No authentication beyond the Square connection yet. The cookie is not signed, so
this is fine on a laptop and not fine on the internet.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import workspace as workspaces
from .pipeline_runner import RunnerRegistry
from .preview import summarize_square
from .schemas import HistoryResponse, MetricsResponse, PredictionResponse
from .service import ModelNotLoaded, ModelService
from .security import (
    OAUTH_STATE_MAX_AGE,
    SESSION_MAX_AGE,
    SecurityError,
    consume_oauth_state,
    new_oauth_state,
    rate_limit,
    read_session,
    sign_session,
)
from .square_oauth import OAuthConfig, OAuthError, authorize_url, exchange_code, fetch_locations
from .workspace import Workspace, WorkspaceError

log = logging.getLogger("api")

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
SESSION_COOKIE = "merchant"
NONCE_COOKIE = "square_oauth"

runners = RunnerRegistry()
# one loaded model per restaurant, so a request never waits for a disk read
_models: dict[str, ModelService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ready. %d restaurant(s) connected", len(workspaces.list_connected()))
    yield
    _models.clear()


app = FastAPI(
    title="Restaurant Sales Forecast",
    description="Connect Square, then predict tomorrow's sales.",
    version="2.0.0",
    lifespan=lifespan,
)


# -- who is asking ----------------------------------------------------------


def current_workspace(merchant: str | None = Cookie(default=None, alias=SESSION_COOKIE)) -> Workspace:
    """The restaurant this browser is signed in as.

    The cookie is signed, so a merchant id typed in by hand does not work. This
    is the only thing standing between one restaurant and another's sales.
    """
    merchant_id = read_session(merchant)
    if not merchant_id:
        raise HTTPException(status_code=401, detail="connect your Square account first")
    try:
        space = workspaces.get(merchant_id)
    except WorkspaceError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if not space.is_connected():
        raise HTTPException(status_code=401, detail="connect your Square account first")
    return space


def model_for(space: Workspace) -> ModelService:
    """Load this restaurant's model, keeping it in memory afterwards."""
    cached = _models.get(space.merchant_id)
    if cached is not None:
        return cached

    if not space.has_model():
        raise HTTPException(status_code=409, detail="no forecast yet, run the setup first")

    try:
        service = ModelService.for_workspace(space)
    except (FileNotFoundError, ModelNotLoaded, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    _models[space.merchant_id] = service
    return service


def _set_session(response: Response, merchant_id: str) -> None:
    """Sign the merchant id into the cookie so it cannot be forged."""
    response.set_cookie(
        SESSION_COOKIE,
        sign_session(merchant_id),
        httponly=True,
        samesite="lax",
        secure=os.environ.get("SQUARE_ENVIRONMENT") == "production",
        max_age=SESSION_MAX_AGE,
    )


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# -- connecting -------------------------------------------------------------


@app.get("/api/session", tags=["square"])
def session(merchant: str | None = Cookie(default=None, alias=SESSION_COOKIE)) -> dict:
    """What this browser is connected to. The page calls this first."""
    merchant_id = read_session(merchant)
    if not merchant_id:
        return {"connected": False}

    try:
        space = workspaces.get(merchant_id)
    except WorkspaceError:
        return {"connected": False}

    if not space.is_connected():
        return {"connected": False}

    runner = runners.for_workspace(space)
    return {
        **space.summary(),
        "setup_running": runner.is_running,
        "setup": runner.state.to_dict(),
    }


@app.get("/api/square/config", tags=["square"])
def square_config() -> dict:
    """Public OAuth setup status for the connect screen.

    This intentionally exposes only the public app id prefix and environment,
    never the application secret.
    """
    try:
        config = OAuthConfig.from_env()
    except OAuthError as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": not config.validation_warnings(),
        "environment": config.environment,
        "application_id_prefix": config.application_id[:16],
        "redirect_url": config.redirect_url,
        "warnings": config.validation_warnings(),
    }


@app.get("/api/accounts", tags=["square"])
def accounts(merchant: str | None = Cookie(default=None, alias=SESSION_COOKIE)) -> dict:
    """The restaurant connected in this browser; never enumerate other sellers."""
    current = read_session(merchant)
    rows = []
    if current:
        try:
            summary = workspaces.get(current).summary()
        except WorkspaceError:
            summary = {"connected": False}
        if summary.get("connected"):
            rows.append(
                {
                    "merchant_id": current,
                    "business_name": summary.get("business_name"),
                    "environment": summary.get("environment"),
                    "has_model": summary.get("has_model"),
                    "current": True,
                }
            )
    return {"accounts": rows}


@app.post("/api/accounts/{merchant_id}/select", tags=["square"])
def select_account(
    merchant_id: str,
    response: Response,
    merchant: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> dict:
    """Only reselect the merchant already authorized in this browser."""
    if read_session(merchant) != merchant_id:
        raise HTTPException(status_code=403, detail="that Square account is not connected here")
    try:
        space = workspaces.get(merchant_id)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not space.is_connected():
        raise HTTPException(status_code=404, detail="that account is not connected")

    _set_session(response, merchant_id)
    return {"ok": True, "merchant_id": merchant_id}


@app.get("/api/square/connect", tags=["square"])
def square_connect(request: Request) -> RedirectResponse:
    """Start the handshake. The state is signed and tied to this browser."""
    try:
        rate_limit(f"connect:{_client(request)}", limit=10, per_seconds=300)
    except SecurityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    try:
        config = OAuthConfig.from_env()
    except OAuthError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    blocking = [
        warning for warning in config.validation_warnings()
        if "needs production Square credentials" in warning or "needs an HTTPS redirect" in warning
    ]
    if blocking:
        raise HTTPException(status_code=400, detail=" ".join(blocking))

    callback = urlparse(config.redirect_url)
    if callback.scheme == "https" and callback.netloc and request.url.hostname != callback.hostname:
        return RedirectResponse(f"{callback.scheme}://{callback.netloc}/api/square/connect", status_code=302)

    state, nonce = new_oauth_state()
    redirect = RedirectResponse(authorize_url(config, state), status_code=302)
    # the nonce never goes to Square, only the browser holds it, so a callback
    # forged from another site cannot match the state we signed
    redirect.set_cookie(
        NONCE_COOKIE,
        nonce,
        httponly=True,
        samesite="lax",
        secure=config.environment == "production",
        max_age=OAUTH_STATE_MAX_AGE,
    )
    return redirect


@app.get("/api/square/callback", include_in_schema=False)
def square_callback(
    code: str | None = None,
    state_token: str | None = Query(None, alias="state"),
    error: str | None = None,
    error_description: str | None = None,
    nonce: str | None = Cookie(default=None, alias=NONCE_COOKIE),
):
    """Square sends the user back here after they approve."""
    if error:
        return RedirectResponse(f"/?error={error_description or error}", status_code=302)
    if not code or not state_token:
        return RedirectResponse("/?error=Square+did+not+send+a+code", status_code=302)

    # signed, unexpired, matches this browser, and not already spent
    try:
        consume_oauth_state(state_token, nonce)
    except SecurityError as exc:
        return RedirectResponse(f"/?error={exc}", status_code=302)

    try:
        config = OAuthConfig.from_env()
        tokens = exchange_code(config, code)
        merchant_id = tokens.get("merchant_id")
        if not merchant_id:
            return RedirectResponse("/?error=Square+did+not+return+a+merchant+id", status_code=302)

        locations = fetch_locations(config, tokens["access_token"])
        workspaces.get(merchant_id).ensure().save_token(tokens, locations, config.environment)
    except (OAuthError, WorkspaceError) as exc:
        return RedirectResponse(f"/?error={exc}", status_code=302)

    redirect = RedirectResponse("/?connected=1", status_code=302)
    _set_session(redirect, merchant_id)
    redirect.delete_cookie(NONCE_COOKIE)
    log.info("connected merchant %s", merchant_id)
    return redirect


@app.delete("/api/account", tags=["square"])
def delete_account(
    response: Response, merchant: str | None = Cookie(default=None, alias=SESSION_COOKIE)
) -> dict:
    """Disconnect and delete everything we hold for this restaurant."""
    merchant_id = read_session(merchant)
    if merchant_id:
        _models.pop(merchant_id, None)
        runners.forget(merchant_id)
        try:
            workspaces.get(merchant_id).delete()
        except WorkspaceError:
            pass
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


# -- what is in their Square ------------------------------------------------


@app.get("/api/data/preview", tags=["data"])
def data_preview(space: Workspace = Depends(current_workspace)) -> dict:
    """Count what is in their account, before we train anything."""
    try:
        return summarize_square(space)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not read Square: {exc}") from exc


# -- building the forecast --------------------------------------------------


@app.post("/api/setup/start", tags=["setup"])
def setup_start(space: Workspace = Depends(current_workspace)) -> dict:
    # training is expensive, do not let it be triggered in a loop
    try:
        rate_limit(f"setup:{space.merchant_id}", limit=5, per_seconds=600)
    except SecurityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    runner = runners.for_workspace(space)

    def reload_model() -> None:
        _models.pop(space.merchant_id, None)

    started = runner.start(on_finish=reload_model)
    return {"started": started, **runner.state.to_dict()}


@app.get("/api/setup/status", tags=["setup"])
def setup_status(space: Workspace = Depends(current_workspace)) -> dict:
    return runners.for_workspace(space).state.to_dict()


# -- the forecast -----------------------------------------------------------


@app.get("/predict", response_model=PredictionResponse, tags=["forecast"])
def predict(space: Workspace = Depends(current_workspace)) -> PredictionResponse:
    try:
        return PredictionResponse(**model_for(space).predict_next())
    except ModelNotLoaded as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/history", response_model=HistoryResponse, tags=["forecast"])
def history(
    days: int = Query(30, ge=1, le=365),
    space: Workspace = Depends(current_workspace),
) -> HistoryResponse:
    points = model_for(space).history(days)
    return HistoryResponse(days=len(points), points=points)


@app.get("/metrics", response_model=MetricsResponse, tags=["status"])
def metrics(space: Workspace = Depends(current_workspace)) -> MetricsResponse:
    try:
        return MetricsResponse(**model_for(space).test_metrics())
    except ModelNotLoaded as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/health", tags=["status"])
def health() -> dict:
    """Is the server up. Says nothing about any particular restaurant."""
    return {"status": "ok", "restaurants_connected": len(workspaces.list_connected())}


# -- the page ---------------------------------------------------------------

if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(DASHBOARD_DIR / "index.html")
