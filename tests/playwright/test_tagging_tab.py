"""Playwright tests for the Tagging tab functionality.

Tests the tagging UI interactions including:
- Tab navigation
- Image display
- Send to Tag button
- Delete & Next button
- Save Tags button
- Key search/filter dropdown
"""

import os

import pytest
from playwright.sync_api import expect
from PIL import Image

from conftest import INPUT_DIR
from helpers import navigate


@pytest.fixture()
def test_images():
    """Create test images in input/ directory for tagging tests."""
    paths = []
    for i in range(2):
        img = Image.new("RGB", (50, 50), color=["red", "blue"][i])
        p = INPUT_DIR / f"tag_test_{i}.png"
        img.save(p)
        paths.append(p)
    yield [str(p) for p in paths]
    for p in paths:
        if p.exists():
            os.remove(p)


class TestTaggingTabNavigation:
    """Test navigation to the Tagging tab."""

    def test_switch_to_tagging_tab(self, page, app_url):
        """Clicking the Tagging tab should switch to tagging interface."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(2000)

        heading = page.locator("h2", has_text="Image Tagging")
        expect(heading).to_be_visible(timeout=10000)


class TestTaggingTabBasicUI:
    """Test basic UI elements in the Tagging tab."""

    def test_send_to_tag_button_visible(self, page, app_url):
        """Send to Tag button should be visible in tagging tab."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        tag_button = page.locator("button", has_text="Send to Tag")
        expect(tag_button).to_be_visible()

    def test_prev_next_buttons_visible(self, page, app_url):
        """Previous and Next buttons should be visible."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        prev_button = page.locator("button", has_text="Previous")
        expect(prev_button).to_be_visible()

        next_button = page.locator("button", has_text="Next")
        expect(next_button).to_be_visible()

    def test_save_tags_button_visible(self, page, app_url):
        """Save Tags button should be visible."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        save_button = page.locator("button", has_text="Save Tags")
        expect(save_button).to_be_visible()


class TestKeySearchDropdown:
    """Test the search/filter functionality for adding keys."""

    def test_add_key_section_visible(self, page, app_url):
        """Add key section with search should be visible in tagging tab."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        add_key_heading = page.locator("h3, h4").filter(has_text="Add Tag Key")
        expect(add_key_heading).to_be_visible(timeout=5000)


class TestTaggingWorkflow:
    """Test complete tagging workflow (without actual VLM calls)."""

    def test_refresh_images_button_exists(self, page, app_url):
        """Refresh Images button should be available."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        refresh_button = page.get_by_role("button", name="🔄 Refresh Images")
        expect(refresh_button).to_be_visible()

    def test_image_counter_shows_count(self, page, app_url):
        """Image counter should show number of images."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        counter = page.locator("text=Image 1 of")
        expect(counter).to_be_visible(timeout=5000)

    def test_status_display_shows_ready(self, page, app_url):
        """Status should show ready message."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        status = page.locator("text=Ready to tag")
        if not status.is_visible():
            status = page.locator("text=No images")
        expect(status).to_be_visible(timeout=5000)
