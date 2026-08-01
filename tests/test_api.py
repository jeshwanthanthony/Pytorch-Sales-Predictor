"""Tests for the multi tenant API.

Two restaurants are set up in a temporary folder: one with a trained model, one
that has only just connected. Most of these tests exist to prove two things —
neither can see the other's numbers, and nobody sees anything before connecting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api import workspace as workspaces
from api.security import sign_session
from training.config import TrainConfig
from training.evaluate import evaluate
from training.train import train

from tests.test_training import FEATURES, make_feature_file

WITH_MODEL = "MERCHANTWITHMODEL"
JUST_CONNECTED = "MERCHANTNOMODEL"


@pytest.fixture(scope="module", autouse=True)
def workspaces_dir(tmp_path_factory):
    """Point every workspace at a temp folder, never the real one."""
    directory = tmp_path_factory.mktemp("workspaces")
    original = workspaces.WORKSPACES_DIR
    workspaces.WORKSPACES_DIR = directory
    yield directory
    workspaces.WORKSPACES_DIR = original


@pytest.fixture(scope="module")
def trained_workspace(workspaces_dir, tmp_path_factory):
    """A restaurant that connected and has a model."""
    space = workspaces.get(WITH_MODEL).ensure()
    space.save_token(
        {"access_token": "fake-token-for-tests", "merchant_id": WITH_MODEL},
        [{"id": "LOC1", "name": "Test Diner", "currency": "USD", "business_name": "Test Diner"}],
        "sandbox",
    )

    features_dir = tmp_path_factory.mktemp("features")
    dataset_file, manifest_file = make_feature_file(features_dir)
    space.features_dir.mkdir(parents=True, exist_ok=True)
    space.dataset_file.write_bytes(dataset_file.read_bytes())
    space.manifest_file.write_text(manifest_file.read_text())

    train(
        space.dataset_file, space.manifest_file, space.models_dir,
        TrainConfig(epochs=40, patience=15, seed=42),
    )
    evaluate(space.checkpoint, space.dataset_file, space.manifest_file, space.models_dir)
    return space


@pytest.fixture(scope="module")
def bare_workspace(workspaces_dir):
    """A restaurant that connected but has not built anything yet."""
    space = workspaces.get(JUST_CONNECTED).ensure()
    space.save_token(
        {"access_token": "another-fake-token", "merchant_id": JUST_CONNECTED},
        [{"id": "LOC9", "name": "New Place", "currency": "USD", "business_name": "New Place"}],
        "sandbox",
    )
    return space


@pytest.fixture
def client(trained_workspace, bare_workspace) -> TestClient:
    api_main._models.clear()
    with TestClient(api_main.app) as test_client:
        yield test_client


def cookie(merchant_id: str) -> dict:
    """A properly signed session, the same as the app would set."""
    return {api_main.SESSION_COOKIE: sign_session(merchant_id)}


class TestNothingBeforeConnecting:
    """The whole point: a stranger sees no numbers at all."""

    @pytest.mark.parametrize(
        "path",
        ["/predict", "/metrics", "/history?days=30", "/api/data/preview", "/api/setup/status"],
    )
    def test_data_routes_require_a_connection(self, client, path):
        response = client.get(path)
        assert response.status_code == 401
        assert "connect" in response.json()["detail"].lower()

    def test_session_says_not_connected(self, client):
        assert client.get("/api/session").json() == {"connected": False}

    def test_an_unknown_cookie_is_not_a_connection(self, client):
        response = client.get("/api/session", cookies=cookie("MADEUPMERCHANT"))
        assert response.json() == {"connected": False}

    def test_predict_with_an_unknown_cookie_is_rejected(self, client):
        assert client.get("/predict", cookies=cookie("MADEUPMERCHANT")).status_code == 401

    def test_a_forged_cookie_cannot_read_another_restaurant(self, client):
        """Typing in someone else's merchant id must not work."""
        forged = {api_main.SESSION_COOKIE: WITH_MODEL}
        assert client.get("/predict", cookies=forged).status_code == 401
        assert client.get("/api/session", cookies=forged).json() == {"connected": False}

    def test_a_cookie_signed_with_another_secret_is_rejected(self, client, monkeypatch):
        monkeypatch.setenv("APP_SECRET", "not-the-servers-secret")
        stolen = {api_main.SESSION_COOKIE: sign_session(WITH_MODEL)}
        monkeypatch.delenv("APP_SECRET")
        assert client.get("/predict", cookies=stolen).status_code == 401

    def test_health_says_nothing_about_any_restaurant(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert set(body) == {"status", "restaurants_connected"}


class TestSession:
    def test_reports_the_connected_restaurant(self, client):
        body = client.get("/api/session", cookies=cookie(WITH_MODEL)).json()
        assert body["connected"] is True
        assert body["business_name"] == "Test Diner"
        assert body["has_model"] is True
        assert body["merchant_id"] == WITH_MODEL

    def test_a_new_connection_has_no_model_yet(self, client):
        body = client.get("/api/session", cookies=cookie(JUST_CONNECTED)).json()
        assert body["connected"] is True
        assert body["has_model"] is False


class TestIsolation:
    """Two restaurants, one server, no leaking."""

    def test_each_gets_their_own_answer_or_none(self, client):
        ready = client.get("/predict", cookies=cookie(WITH_MODEL))
        assert ready.status_code == 200
        assert ready.json()["predicted_sales"] > 0

        not_ready = client.get("/predict", cookies=cookie(JUST_CONNECTED))
        assert not_ready.status_code == 409
        assert "no forecast yet" in not_ready.json()["detail"]

    def test_metrics_do_not_cross_over(self, client):
        assert client.get("/metrics", cookies=cookie(WITH_MODEL)).status_code == 200
        assert client.get("/metrics", cookies=cookie(JUST_CONNECTED)).status_code == 409

    def test_workspaces_are_separate_folders(self, trained_workspace, bare_workspace):
        assert trained_workspace.root != bare_workspace.root
        assert trained_workspace.checkpoint.exists()
        assert not bare_workspace.checkpoint.exists()

    def test_both_are_listed_as_connected(self, trained_workspace, bare_workspace):
        assert set(workspaces.list_connected()) >= {WITH_MODEL, JUST_CONNECTED}

    def test_accounts_endpoint_only_lists_this_browsers_merchant(self, client):
        body = client.get("/api/accounts", cookies=cookie(WITH_MODEL)).json()
        assert [account["merchant_id"] for account in body["accounts"]] == [WITH_MODEL]

    def test_accounts_endpoint_lists_nothing_without_a_session(self, client):
        assert client.get("/api/accounts").json() == {"accounts": []}

    def test_cannot_switch_into_another_connected_merchant(self, client):
        response = client.post(
            f"/api/accounts/{JUST_CONNECTED}/select",
            cookies=cookie(WITH_MODEL),
        )
        assert response.status_code == 403


class TestForecast:
    def test_prediction_shape(self, client):
        body = client.get("/predict", cookies=cookie(WITH_MODEL)).json()
        assert body["interval_low"] <= body["predicted_sales"] <= body["interval_high"]
        assert 0 <= body["confidence"] <= 1
        assert body["important_features"]
        assert all(f["name"] in FEATURES for f in body["important_features"])

    def test_history_only_uses_untrained_days(self, client):
        body = client.get("/history?days=10", cookies=cookie(WITH_MODEL)).json()
        assert body["days"] == 10
        assert {p["split"] for p in body["points"]} <= {"val", "test"}

    def test_metrics_include_both_baselines(self, client):
        body = client.get("/metrics", cookies=cookie(WITH_MODEL)).json()
        assert {b["name"] for b in body["baselines"]} == {"last week", "rolling 7"}

    def test_repeated_requests_agree(self, client):
        first = client.get("/predict", cookies=cookie(WITH_MODEL)).json()
        second = client.get("/predict", cookies=cookie(WITH_MODEL)).json()
        assert first["predicted_sales"] == second["predicted_sales"]
        assert first["model_uncertainty"] == second["model_uncertainty"]


class TestAccountRemoval:
    def test_disconnect_deletes_everything(self, client):
        space = workspaces.get("TEMPORARYMERCHANT").ensure()
        space.save_token({"access_token": "x", "merchant_id": "TEMPORARYMERCHANT"}, [], "sandbox")
        assert space.root.exists()

        response = client.delete("/api/account", cookies=cookie("TEMPORARYMERCHANT"))
        assert response.status_code == 200
        # their data is gone from disk, not just hidden
        assert not space.root.exists()


class TestWorkspacePaths:
    def test_rejects_a_path_traversal_id(self):
        for bad in ["../secrets", "a/b", ".."]:
            with pytest.raises(workspaces.WorkspaceError):
                workspaces.get(bad)

    def test_token_file_is_not_world_readable(self, trained_workspace):
        assert trained_workspace.token_file.stat().st_mode & 0o777 == 0o600

    def test_token_round_trips(self, trained_workspace):
        auth = trained_workspace.auth()
        assert auth.access_token == "fake-token-for-tests"
        assert auth.merchant_id == WITH_MODEL
        assert auth.location_ids == ["LOC1"]

    def test_summary_of_an_unconnected_workspace(self):
        assert workspaces.get("NEVERCONNECTED").summary() == {"connected": False}


class TestDashboardFiles:
    def test_index_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Connect your Square account" in response.text

    def test_index_ships_no_real_numbers(self, client):
        """The page contains placeholders only, never a figure."""
        text = client.get("/").text
        assert "$1,2" not in text
        assert "predicted_sales" not in text

    def test_static_assets_are_served(self, client):
        for path in ("/static/app.js", "/static/styles.css"):
            assert client.get(path).status_code == 200

    def test_no_dev_login_helper_shipped(self):
        assert not (Path(api_main.DASHBOARD_DIR) / "_devlogin.html").exists()


class TestAuthorizeUrl:
    """The two things that broke sandbox connect, guarded so they cannot come back."""

    @staticmethod
    def _config(environment: str):
        from api.square_oauth import OAuthConfig

        return OAuthConfig(
            application_id="app-id",
            application_secret="secret",
            environment=environment,
            redirect_url="http://localhost:8080/api/square/callback",
        )

    def test_sandbox_uses_the_connect_host(self):
        from api.square_oauth import authorize_url

        url = authorize_url(self._config("sandbox"), "signed-state")
        assert url.startswith("https://connect.squareupsandbox.com/oauth2/authorize")
        # app.squareupsandbox.com is not an oauth host, it errors immediately
        assert "app.squareupsandbox.com" not in url

    def test_sandbox_omits_session(self):
        """session=false asks for a login page sandbox does not have -> white screen."""
        from api.square_oauth import authorize_url

        assert "session=" not in authorize_url(self._config("sandbox"), "signed-state")

    def test_production_keeps_session_false(self):
        from api.square_oauth import authorize_url

        url = authorize_url(self._config("production"), "signed-state")
        assert "session=false" in url
        assert url.startswith("https://connect.squareup.com/oauth2/authorize")

    def test_redirect_uri_is_sent_exactly(self):
        from urllib.parse import parse_qs, urlparse

        from api.square_oauth import authorize_url

        query = parse_qs(urlparse(authorize_url(self._config("sandbox"), "signed-state")).query)
        assert query["redirect_uri"] == ["http://localhost:8080/api/square/callback"]


class TestSquareConfig:
    def test_config_endpoint_does_not_expose_secret(self, client, monkeypatch):
        monkeypatch.setenv("SQUARE_APPLICATION_ID", "sandbox-app-id")
        monkeypatch.setenv("SQUARE_APPLICATION_SECRET", "sandbox-secret-value")
        monkeypatch.setenv("SQUARE_ENVIRONMENT", "sandbox")
        monkeypatch.setenv("SQUARE_REDIRECT_URL", "http://localhost:8080/api/square/callback")

        body = client.get("/api/square/config").json()

        assert body["environment"] == "sandbox"
        assert body["application_id_prefix"] == "sandbox-app-id"
        assert "secret" not in body
        assert any("sandbox seller test accounts" in warning for warning in body["warnings"])

    def test_production_with_sandbox_credentials_is_blocked(self, client, monkeypatch):
        monkeypatch.setenv("SQUARE_APPLICATION_ID", "sandbox-app-id")
        monkeypatch.setenv("SQUARE_APPLICATION_SECRET", "sandbox-secret-value")
        monkeypatch.setenv("SQUARE_ENVIRONMENT", "production")
        monkeypatch.setenv("SQUARE_REDIRECT_URL", "http://localhost:8080/api/square/callback")

        response = client.get("/api/square/connect")

        assert response.status_code == 400
        assert "production Square credentials" in response.json()["detail"]
        assert "HTTPS redirect" in response.json()["detail"]

    def test_localhost_connect_hops_to_https_redirect_host(self, client, monkeypatch):
        monkeypatch.setenv("SQUARE_APPLICATION_ID", "prod-app-id")
        monkeypatch.setenv("SQUARE_APPLICATION_SECRET", "prod-secret-value")
        monkeypatch.setenv("SQUARE_ENVIRONMENT", "production")
        monkeypatch.setenv(
            "SQUARE_REDIRECT_URL", "https://public.example.com/api/square/callback"
        )

        response = client.get("/api/square/connect", follow_redirects=False)

        assert response.status_code == 302
        assert response.headers["location"] == "https://public.example.com/api/square/connect"
