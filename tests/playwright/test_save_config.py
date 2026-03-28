"""Playwright tests verifying that Save Config persists settings to disk.

Tests that modifying the endpoint URL and selecting a model, then clicking
"Save Config", writes the correct values to the provider's config.json file.
"""

import json
import pytest
from pathlib import Path


PROVIDERS = [
    {
        "name": "lmstudio",
        "test_url": "http://localhost:17870/v1",
        "test_model": "custom-lmstudio-model",
        "config_file": "lmstudio_config.json",
    },
    {
        "name": "llamacpp",
        "test_url": "http://localhost:17870",
        "test_model": "custom-llamacpp-model",
        "config_file": "llamacpp_config.json",
    },
]


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    """Point CONFIG_DIR to a temporary directory for test isolation."""
    monkeypatch.setattr("chatbot.config_manager.CONFIG_DIR", tmp_path)
    # Also patch the CONFIG_DIR used by config_manager at module level
    import chatbot.config_manager

    chatbot.config_manager.CONFIG_DIR = tmp_path
    return tmp_path


def _navigate(page, app_url):
    page.goto(app_url, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)


def _select_provider(page, provider_name):
    page.locator("label").filter(has_text=provider_name).first.click()
    page.wait_for_timeout(1500)


class TestSaveConfiguration:
    """Verify Save Config writes endpoint and model to the config file."""

    @pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p["name"])
    def test_save_config_persists_endpoint_and_model(
        self, page, app_url, provider, config_dir
    ):
        """Saving configuration writes endpoint_url and model_name to the JSON file.

        Steps:
        1. Navigate and select the provider
        2. Modify the endpoint URL
        3. Type a model name into the dropdown
        4. Click Save Config
        5. Assert the status confirms the save
        6. Read the config JSON and assert values match
        """
        _navigate(page, app_url)
        _select_provider(page, provider["name"])

        # Modify the endpoint URL
        endpoint = page.get_by_label("Endpoint URL")
        endpoint.fill(provider["test_url"])

        # Wait for any pending auto-fetch to complete
        page.wait_for_timeout(3000)

        # Type a model name into the dropdown (allow_custom_value=True)
        # Use keyboard interaction so Gradio's Svelte component registers the change
        dropdown = page.get_by_label("Select Model")
        dropdown.click()
        page.keyboard.press("Control+a")
        page.keyboard.type(provider["test_model"])
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)

        # Click Save Config
        page.get_by_role("button", name="Save Config").click()
        page.wait_for_timeout(2000)

        # Assert the status shows a save confirmation (use first to avoid
        # strict-mode conflict with the Gradio toast notification)
        saved_status = page.get_by_text("Configuration saved").first
        saved_status.wait_for(state="visible", timeout=10000)
        assert saved_status.is_visible(), "Expected save confirmation in status"

        # Read the config file and verify contents
        config_path = config_dir / provider["config_file"]
        assert config_path.exists(), (
            f"Expected config file {config_path} to exist after save"
        )

        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        assert saved_config["endpoint_url"] == provider["test_url"], (
            f"Expected endpoint_url '{provider['test_url']}' "
            f"but got '{saved_config.get('endpoint_url')}'"
        )
        assert saved_config["model_name"] == provider["test_model"], (
            f"Expected model_name '{provider['test_model']}' "
            f"but got '{saved_config.get('model_name')}'"
        )

    @pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p["name"])
    def test_save_config_creates_correct_file_per_provider(
        self, page, app_url, provider, config_dir
    ):
        """Each provider saves to its own {provider}_config.json file.

        Steps:
        1. Navigate and select the provider
        2. Set endpoint URL and model
        3. Click Save Config
        4. Assert only the expected config file exists in the temp directory
        """
        _navigate(page, app_url)
        _select_provider(page, provider["name"])

        endpoint = page.get_by_label("Endpoint URL")
        endpoint.fill(provider["test_url"])

        # Wait for any pending auto-fetch to complete
        page.wait_for_timeout(3000)

        dropdown = page.get_by_label("Select Model")
        dropdown.click()
        page.keyboard.press("Control+a")
        page.keyboard.type(provider["test_model"])
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)

        page.get_by_role("button", name="Save Config").click()
        page.wait_for_timeout(2000)

        saved_status = page.get_by_text("Configuration saved").first
        saved_status.wait_for(state="visible", timeout=10000)

        # Verify the correct file exists
        expected_path = config_dir / provider["config_file"]
        assert expected_path.exists(), (
            f"Expected {provider['config_file']} to be created"
        )

        # Verify file naming convention: {provider}_config.json
        json_files = sorted(p.name for p in config_dir.glob("*.json"))
        assert provider["config_file"] in json_files, (
            f"Expected '{provider['config_file']}' in {json_files}"
        )
