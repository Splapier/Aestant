"""Playwright tests verifying the Refresh Models button behavior.

Tests that clicking "Refresh Models" returns the expected status message
(error when the endpoint is unreachable, success when a server responds)
and that a model becomes selectable in the dropdown on success.
"""

import json
import pytest
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


PROVIDERS = [
    {"name": "lmstudio", "test_url": "http://localhost:17861"},
    {"name": "llamacpp", "test_url": "http://localhost:17861"},
]

MOCK_MODELS = ["mock-model-alpha", "mock-model-beta"]


class _ModelHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves a mock OpenAI-compatible /models response."""

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/models", "/v1/models"):
            body = json.dumps({"data": [{"id": m} for m in MOCK_MODELS]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args):
        """Suppress request logging."""


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    """Point CONFIG_DIR to a temporary directory for test isolation."""
    monkeypatch.setattr("chatbot.config_manager.CONFIG_DIR", tmp_path)
    import chatbot.config_manager

    chatbot.config_manager.CONFIG_DIR = tmp_path
    return tmp_path


@pytest.fixture(scope="module")
def mock_server():
    """Start a lightweight HTTP server that mocks the /models endpoint."""
    server = HTTPServer(("127.0.0.1", 17861), _ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield
    server.shutdown()


def _navigate(page, app_url):
    page.goto(app_url, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)


def _select_provider(page, provider_name):
    page.locator("label").filter(has_text=provider_name).first.click()
    page.wait_for_timeout(1500)


class TestRefreshModels:
    """Verify Refresh Models returns expected status and populates the dropdown."""

    @pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p["name"])
    def test_refresh_models_connection_error(self, page, app_url, provider, config_dir):
        """Clicking Refresh Models with an unreachable endpoint shows a connection error.

        Steps:
        1. Navigate and select the provider
        2. Set an endpoint URL that no server listens on
        3. Click Refresh Models
        4. Assert the status contains an error indicator
        """
        _navigate(page, app_url)
        _select_provider(page, provider["name"])

        endpoint = page.get_by_label("Endpoint URL")
        endpoint.fill("http://127.0.0.1:19999")

        page.get_by_role("button", name="Refresh Models").click()
        page.wait_for_timeout(3000)

        # The status should reflect a connection error
        status = page.get_by_text("Connection Error")
        status.wait_for(state="visible", timeout=10000)
        assert status.is_visible(), "Expected a Connection Error status message"

    @pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p["name"])
    def test_refresh_models_success_populates_dropdown(
        self, page, app_url, provider, mock_server, config_dir
    ):
        """Clicking Refresh Models against a working endpoint populates the dropdown.

        Steps:
        1. Navigate and select the provider
        2. Set the endpoint URL to the mock server
        3. Click Refresh Models
        4. Assert the status reports models found
        5. Assert the dropdown contains the expected model choices
        6. Assert a model is selected by default
        """
        _navigate(page, app_url)
        _select_provider(page, provider["name"])

        endpoint = page.get_by_label("Endpoint URL")
        endpoint.fill(provider["test_url"])

        page.get_by_role("button", name="Refresh Models").click()
        page.wait_for_timeout(3000)

        # Status should report success
        status = page.get_by_text("Found")
        status.wait_for(state="visible", timeout=10000)
        assert "model(s)" in status.text_content(), (
            "Expected status to report the number of models found"
        )

        # The dropdown should have a value selected (the first mock model)
        dropdown = page.get_by_label("Select Model")
        dropdown_value = dropdown.input_value()
        assert dropdown_value in MOCK_MODELS, (
            f"Expected dropdown to select one of {MOCK_MODELS}, got '{dropdown_value}'"
        )
