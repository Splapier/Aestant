"""Playwright tests verifying the provider endpoint URL is editable in the UI.

Tests that for each supported provider (lmstudio, llamacpp), the Endpoint URL
textbox is visible, enabled, and accepts user input — confirming the user can
write to it in the browser.
"""

import pytest

PROVIDERS = [
    {
        "name": "lmstudio",
        "default_url": "http://localhost:1234/v1",
        "test_url": "http://localhost:9999/v1",
    },
    {
        "name": "llamacpp",
        "default_url": "http://localhost:8080",
        "test_url": "http://localhost:9999",
    },
]


def _navigate_to_app(page, app_url):
    """Navigate to the Gradio app and wait for it to load."""
    page.goto(app_url, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)


def _select_provider(page, provider_name):
    """Click the provider radio button and wait for the config panel to render."""
    radio = page.locator("label").filter(has_text=provider_name).first
    radio.click()
    page.wait_for_timeout(1500)


class TestEndpointUrlEditable:
    """Verify the Endpoint URL textbox is writable for each provider."""

    @pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p["name"])
    def test_endpoint_url_is_visible_and_editable(self, page, app_url, provider):
        """The Endpoint URL textbox must be visible and enabled for the user to type into it.

        Steps:
        1. Select the provider radio button
        2. Locate the Endpoint URL textbox in the sidebar
        3. Assert it is visible and not disabled
        4. Clear the field and type a new URL
        5. Assert the textbox value reflects the typed text
        """
        _navigate_to_app(page, app_url)
        _select_provider(page, provider["name"])

        # Locate the Endpoint URL textbox by its label
        textbox = page.get_by_label("Endpoint URL")

        # Assert the textbox is visible and enabled
        textbox.wait_for(state="visible", timeout=5000)
        assert textbox.is_visible(), (
            f"Endpoint URL textbox should be visible for {provider['name']}"
        )
        assert not textbox.is_disabled(), (
            f"Endpoint URL textbox should be enabled for {provider['name']}"
        )

        # Clear any existing value and type a new URL
        textbox.fill("")
        textbox.fill(provider["test_url"])

        # Verify the textbox now contains the typed URL
        assert textbox.input_value() == provider["test_url"], (
            f"Endpoint URL textbox should accept user input for {provider['name']}"
        )

    @pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p["name"])
    def test_endpoint_url_accepts_special_characters(self, page, app_url, provider):
        """The Endpoint URL textbox must accept URLs with special characters like ports and paths.

        Steps:
        1. Select the provider radio button
        2. Locate the Endpoint URL textbox
        3. Type a URL containing colons, slashes, and digits
        4. Assert the value is preserved exactly
        """
        _navigate_to_app(page, app_url)
        _select_provider(page, provider["name"])

        textbox = page.get_by_label("Endpoint URL")
        textbox.wait_for(state="visible", timeout=5000)

        # Type a URL with mixed special characters
        url_with_port = "https://192.168.1.100:8443/api/v2"
        textbox.fill("")
        textbox.fill(url_with_port)

        assert textbox.input_value() == url_with_port, (
            f"Endpoint URL textbox should preserve special characters for {provider['name']}"
        )
