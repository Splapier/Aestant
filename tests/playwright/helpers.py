"""Shared helper functions for Playwright end-to-end tests."""


def navigate(page, app_url):
    """Navigate to the Gradio app and wait for it to load."""
    page.goto(app_url, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)


def select_provider(page, provider_name):
    """Click the provider radio button and wait for the config panel to render."""
    page.locator("label").filter(has_text=provider_name).first.click()
    page.wait_for_timeout(1500)
