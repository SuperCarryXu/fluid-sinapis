"""API-key authentication: every /api endpoint requires "Authorization: Bearer <key>".

The console shell (index.html, SPA deep links, /assets) stays public: a browser
navigation cannot attach a Bearer header, and the page itself prompts for the key.

The key comes from the deployment environment (SINAPIS_API_KEY) and is enforced
at WebSettings.from_env() with fail-closed semantics; create_app() installs the
/api path-prefix middleware (ApiKeyMiddleware) whenever the key is non-empty. Tests
build settings directly, so both halves are covered separately here.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.controller import web_controller
from server.controller.web_controller import WebSettings, create_app

KEY = "test-key-abc123"


def bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(WebSettings(api_key=KEY)))


# -- configuration resolution -------------------------------------------------
class TestFromEnvKeyResolution:
    def test_missing_key_refuses_to_start(self):
        """Fail closed: no key and no explicit opt-out must abort the launch."""
        with pytest.raises(ValueError, match="SINAPIS_API_KEY"):
            WebSettings.from_env({})

    def test_key_is_loaded_and_stripped(self):
        settings = WebSettings.from_env({"SINAPIS_API_KEY": f"  {KEY}  "})
        assert settings.api_key == KEY

    def test_explicit_disable_yields_no_key(self):
        settings = WebSettings.from_env({"SINAPIS_AUTH_DISABLED": "true"})
        assert settings.api_key == ""

    def test_key_and_disable_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            WebSettings.from_env({"SINAPIS_API_KEY": KEY, "SINAPIS_AUTH_DISABLED": "true"})

    def test_invalid_disable_value_still_requires_a_key(self):
        with pytest.raises(ValueError, match="SINAPIS_API_KEY"):
            WebSettings.from_env({"SINAPIS_AUTH_DISABLED": "probably"})


# -- request gating ------------------------------------------------------------
class TestRequestGating:
    def test_every_endpoint_rejects_a_missing_key(self, client):
        for path in ("/api/healthz", "/api/overview", "/api/docs", "/api/openapi.json"):
            response = client.get(path)
            assert response.status_code == 401, path
            assert response.headers["www-authenticate"] == "Bearer"

    def test_post_endpoints_reject_a_missing_key(self, client):
        body = {"source": "t", "metrics": [{"name": "m", "value": 1}]}
        assert client.post("/api/user-system/metrics", json=body).status_code == 401
        assert client.post(
            "/api/user-system/events",
            json={"source": "t", "events": [{"name": "e", "labels": {}, "fields": {}}]},
        ).status_code == 401

    def test_wrong_key_is_rejected(self, client):
        assert client.get("/api/healthz", headers=bearer("not-the-key")).status_code == 401

    def test_malformed_header_is_rejected(self, client):
        for value in ("Basic dXNlcjpwYXNz", "Bearer", "Bearer    ", "bearer-without-space",
                      f"Token {KEY}"):
            response = client.get("/api/healthz", headers={"Authorization": value})
            assert response.status_code == 401, value

    def test_rejection_message_does_not_leak_why(self, client):
        """Missing, malformed, and wrong keys all answer identically, so a caller
        cannot probe which check failed."""
        messages = {
            client.get("/api/healthz").json()["detail"],
            client.get("/api/healthz", headers=bearer("nope")).json()["detail"],
            client.get(
                "/api/healthz", headers={"Authorization": "garbage"}
            ).json()["detail"],
        }
        assert messages == {"missing or invalid API key"}

    def test_correct_key_passes_on_get_and_post(self, client):
        assert client.get("/api/healthz", headers=bearer(KEY)).status_code == 200
        assert client.get("/api/overview", headers=bearer(KEY)).status_code == 200
        assert client.get("/api/docs", headers=bearer(KEY)).status_code == 200
        # Auth passed: the request reaches the business layer (200 accepted or
        # 503 config-unavailable), never the auth gate's 401.
        response = client.post(
            "/api/user-system/metrics",
            json={"source": "t", "metrics": [{"name": "m", "value": 1}]},
            headers=bearer(KEY),
        )
        assert response.status_code in (200, 503)

    def test_scheme_match_is_case_insensitive(self, client):
        response = client.get(
            "/api/healthz", headers={"Authorization": f"bearer {KEY}"}
        )
        assert response.status_code == 200

    def test_no_key_settings_serve_openly(self):
        """Programmatic settings without a key keep auth off (tests/dev tooling);
        production cannot reach this state because from_env() fails closed."""
        open_client = TestClient(create_app(WebSettings()))
        assert open_client.get("/api/healthz").status_code == 200


# -- auth scope: console shell is public, the API is not ----------------------
INDEX_HTML = "<!doctype html><title>console</title><div id=app></div>"


@pytest.fixture()
def console_client(tmp_path, monkeypatch) -> TestClient:
    """An authenticated app serving a fake frontend build from tmp_path."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text(INDEX_HTML)
    (tmp_path / "assets" / "app.js").write_text("console.log('ok')")
    (tmp_path / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(web_controller, "FRONTEND_DIST", tmp_path)
    return TestClient(create_app(WebSettings(api_key=KEY)))


class TestAuthScope:
    def test_console_index_loads_without_a_key(self, console_client):
        """Regression: with SINAPIS_API_KEY set, a plain browser visit to / used to
        hit the app-wide auth gate (401) before the key prompt could ever render."""
        response = console_client.get("/")
        assert response.status_code == 200
        assert "console" in response.text

    def test_spa_deep_link_and_assets_load_without_a_key(self, console_client):
        assert console_client.get("/settings/some/page").status_code == 200
        assert console_client.get("/assets/app.js").status_code == 200
        assert console_client.get("/favicon.svg").status_code == 200

    def test_every_documented_api_route_still_requires_the_key(self, console_client):
        schema = console_client.app.openapi()
        endpoints = [
            (method.upper(), path)
            for path, ops in schema["paths"].items()
            for method in ops
        ]
        # Hidden from the schema but still API routes.
        endpoints += [("GET", "/api/docs"), ("GET", "/api/openapi.json")]
        assert all(path.startswith("/api/") for _, path in endpoints), endpoints
        assert len(endpoints) >= 8
        for method, path in endpoints:
            response = console_client.request(method, path)
            assert response.status_code == 401, (method, path)

    @pytest.mark.parametrize("method", ["GET", "POST"])
    def test_route_added_later_is_gated_and_reachable(self, console_client, method):
        """Auth is enforced by path prefix, not per route: a route someone adds
        after create_app() (here hidden from the schema) cannot opt out by mistake,
        and nothing registered earlier shadows it once the key is supplied."""
        app = console_client.app

        @app.api_route("/api/added-later", methods=[method], include_in_schema=False)
        def added_later() -> dict[str, str]:
            return {"reached": "yes"}

        assert console_client.request(method, "/api/added-later").status_code == 401
        response = console_client.request(method, "/api/added-later", headers=bearer(KEY))
        assert response.status_code == 200
        assert response.json() == {"reached": "yes"}

    @pytest.mark.parametrize(
        "method",
        ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS",
         # Not enumerated anywhere in the app: unknown /api handling must not rely
         # on a method list.
         "TRACE", "PROPFIND", "BREW"],
    )
    @pytest.mark.parametrize("path", ["/api", "/api/", "/api/does-not-exist", "/api/a/b"])
    def test_unknown_api_path_is_401_without_key_and_404_with_it(
        self, console_client, method, path
    ):
        """Any method on any unknown /api path: 401 anonymously (never 405, and
        never the SPA page), 404 once authenticated."""
        assert console_client.request(method, path).status_code == 401
        response = console_client.request(method, path, headers=bearer(KEY))
        assert response.status_code == 404
        if method != "HEAD":
            assert response.json()["detail"] == "unknown api route"

    def test_non_get_on_console_paths_is_405(self, console_client):
        response = console_client.post("/settings/page")
        assert response.status_code == 405
        assert response.headers["allow"] == "GET, HEAD"

    def test_cors_preflight_is_answered_without_a_key(self, console_client):
        """Browsers send preflights without credentials; CORS wraps the auth gate,
        so an allowed origin gets its preflight answered, and a 401 still carries
        the CORS headers the page needs to read it."""
        origin = "http://localhost:5173"
        preflight = console_client.options(
            "/api/overview",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == origin

        denied = console_client.get("/api/overview", headers={"Origin": origin})
        assert denied.status_code == 401
        assert denied.headers["access-control-allow-origin"] == origin


# -- auth under a mount prefix / root_path -------------------------------------
class TestAuthUnderPrefix:
    """The router strips root_path before matching; the gate must too, otherwise
    /console/api/... would reach the API handlers without a key."""

    PATHS = [
        ("GET", "/api/healthz"),
        ("GET", "/api/openapi.json"),
        ("GET", "/api/user-system/metrics/schema"),
        ("POST", "/api/user-system/metrics"),
        ("POST", "/api/user-system/events"),
        ("GET", "/api/does-not-exist"),
        ("GET", "/api"),
    ]

    @pytest.fixture()
    def mounted_client(self, console_client) -> TestClient:
        parent = FastAPI()
        parent.mount("/console", console_client.app)
        return TestClient(parent)

    @pytest.mark.parametrize("method,path", PATHS)
    def test_mounted_api_requires_the_key(self, mounted_client, method, path):
        assert mounted_client.request(method, "/console" + path).status_code == 401

    def test_mounted_api_and_console_work_with_and_without_key(self, mounted_client):
        assert mounted_client.get("/console/api/healthz", headers=bearer(KEY)).status_code == 200
        assert mounted_client.get("/console/").status_code == 200
        assert mounted_client.get("/console/assets/app.js").status_code == 200

    @pytest.mark.parametrize("method,path", PATHS)
    def test_root_path_api_requires_the_key(self, console_client, method, path):
        """--root-path deployments (reverse proxy prefix): with root_path set the
        request path may carry the prefix; either way the key is required."""
        client = TestClient(console_client.app, root_path="/console")
        assert client.request(method, path).status_code == 401
        assert client.request(method, "/console" + path).status_code == 401
