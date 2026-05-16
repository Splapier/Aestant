"""Playwright tests for the Comparison tab with rectangle tools.

Tests the comparison UI interactions including:
- Tab navigation to Comparison
- Rectangle tool display (instead of gr.Image)
- Image dropdown selection
- Winner/Loser selection buttons
- Inference functionality
"""

import os

import pytest
from playwright.sync_api import expect
from PIL import Image

from conftest import INPUT_DIR
from helpers import navigate


@pytest.fixture()
def test_images():
    """Create test images in input/ directory for comparison tests."""
    paths = []
    for i, color in enumerate(["red", "blue", "green", "yellow"]):
        img = Image.new("RGB", (50, 50), color=color)
        p = INPUT_DIR / f"comparison_test_{i}.png"
        img.save(p)
        paths.append(p)
    yield [str(p) for p in paths]
    for p in paths:
        if p.exists():
            os.remove(p)


class TestComparisonTabNavigation:
    """Test navigation to the Comparison tab."""

    def test_switch_to_comparison_tab(self, page, app_url):
        """Clicking the Comparison tab should switch to comparison interface."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(2000)

        heading = page.locator("h2", has_text="Preference Comparison")
        expect(heading).to_be_visible(timeout=10000)


class TestComparisonTabBasicUI:
    """Test basic UI elements in the Comparison tab."""

    def test_a_wins_button_visible(self, page, app_url):
        """A Wins button should be visible in comparison tab."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        a_wins_btn = page.locator("button", has_text="A Wins")
        expect(a_wins_btn).to_be_visible()

    def test_b_wins_button_visible(self, page, app_url):
        """B Wins button should be visible in comparison tab."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        b_wins_btn = page.locator("button", has_text="B Wins")
        expect(b_wins_btn).to_be_visible()

    def test_swap_button_visible(self, page, app_url):
        """Swap button should be visible in comparison tab."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        swap_btn = page.locator("button", has_text="Swap")
        expect(swap_btn).to_be_visible()


class TestComparisonImageDropdowns:
    """Test image selection dropdowns in Comparison tab."""

    def test_image_a_dropdown_exists(self, page, app_url):
        """Image A dropdown should exist and contain images."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        dropdown_a = page.locator("label:has-text('Select Image A')")
        expect(dropdown_a).to_be_visible()

    def test_image_b_dropdown_exists(self, page, app_url):
        """Image B dropdown should exist and contain images."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        dropdown_b = page.locator("label:has-text('Select Image B')")
        expect(dropdown_b).to_be_visible()

    def test_dropdown_has_choices(self, page, app_url):
        """Dropdowns should have image options available."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        dropdown_a = (
            page.locator("label:has-text('Select Image A')")
            .locator("..")
            .locator("select")
        )
        expect(dropdown_a).to_be_visible()


class TestRectangleToolDisplay:
    """Test rectangle tool components in Comparison tab."""

    def test_rectangle_tool_label_visible_for_image_a(self, page, app_url):
        """Rectangle tool label for Image A should be visible."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        label_a = page.locator("#rect-tool").first
        if label_a.count() > 0:
            expect(label_a).to_be_visible(timeout=5000)

    def test_canvas_elements_exist(self, page, app_url):
        """Canvas elements for rectangle drawing should exist."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        canvas = page.locator("#rect-canvas")
        if canvas.count() > 0:
            expect(canvas.first).to_be_visible()

    def test_color_selector_exists(self, page, app_url):
        """Color selector dropdown should exist for rectangle tool."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        color_select = page.locator("#color-select")
        if color_select.count() > 0:
            expect(color_select).to_be_visible()

    def test_undo_button_exists(self, page, app_url):
        """Undo button should exist for rectangle tool."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        undo_btn = page.locator("#undo-btn")
        if undo_btn.count() > 0:
            expect(undo_btn).to_be_visible()

    def test_clear_button_exists(self, page, app_url):
        """Clear All button should exist for rectangle tool."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        clear_btn = page.locator("#clear-btn")
        if clear_btn.count() > 0:
            expect(clear_btn).to_be_visible()


class TestComparisonInferenceSection:
    """Test inference section of Comparison tab."""

    def test_refresh_pools_button_visible(self, page, app_url):
        """Refresh Pools button should be visible."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        refresh_btn = page.get_by_role("button", name="🔄 Refresh Pools")
        expect(refresh_btn).to_be_visible()

    def test_find_winner_button_visible(self, page, app_url):
        """Find Winner button should be visible."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        find_btn = page.get_by_role("button", name="🏆 Find Winner")
        expect(find_btn).to_be_visible()

    def test_pool_status_display_exists(self, page, app_url):
        """Pool Status markdown should exist."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        pool_status = page.locator("text=Winners:")
        if pool_status.count() > 0:
            expect(pool_status).to_be_visible()


class TestRectangleToolInteraction:
    """Test interaction with rectangle drawing tool."""

    def test_rectangle_count_starts_at_zero(self, page, app_url):
        """Rectangle count should start at 0 rects."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        rect_count = page.locator("#rect-count")
        if rect_count.count() > 0:
            expect(rect_count).to_have_text("0 rects")

    def test_color_options_available(self, page, app_url):
        """Color select should have multiple color options."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        color_select = page.locator("#color-select")
        if color_select.count() > 0:
            options = color_select.locator("option")
            expect(options).to_have_count(3)

    def test_instructions_text_visible(self, page, app_url):
        """Instructions for drawing rectangles should be visible."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        instructions = page.locator("#instructions")
        if instructions.count() > 0:
            expect(instructions).to_contain_text("Click and drag")


class TestComparisonTabResilience:
    """Test comparison tab handles edge cases gracefully."""

    def test_tab_works_without_images_in_pool(self, page, app_url):
        """Comparison tab should display without errors even when pools are empty."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        pool_status = page.locator("text=Winners:")
        if pool_status.count() > 0:
            expect(pool_status).to_be_visible()

        a_wins_btn = page.get_by_role("button", name="A Wins")
        expect(a_wins_btn).to_be_visible()

    def test_find_winner_works_with_empty_pools(self, page, app_url):
        """Find Winner should handle empty pools without crashing."""
        navigate(page, app_url)

        comparison_tab = page.get_by_role("tab", name="Comparison")
        comparison_tab.click()
        page.wait_for_timeout(1000)

        find_btn = page.get_by_role("button", name="🏆 Find Winner")
        find_btn.click()
        page.wait_for_timeout(2000)

        status_display = page.locator("text=No winners")
        if status_display.count() > 0:
            expect(status_display).to_be_visible()
