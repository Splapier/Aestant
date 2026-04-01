"""Playwright tests verifying image auto-loading and Refresh Images button behavior.

Tests that images from the input/ directory are displayed automatically on page
load and that clicking "Refresh Images" loads images into the editors on the
first click (not requiring a second click).
"""

import os

import pytest
from PIL import Image

from conftest import INPUT_DIR
from helpers import navigate


@pytest.fixture()
def test_images():
    """Create two small test PNGs in input/ and clean up after the test."""
    paths = []
    for i in range(2):
        img = Image.new("RGB", (50, 50), color=["red", "blue"][i])
        p = INPUT_DIR / f"test_image_{i}.png"
        img.save(p)
        paths.append(p)
    yield [str(p) for p in paths]
    for p in paths:
        if p.exists():
            os.remove(p)


class TestImagesAutoLoaded:
    """Verify images are displayed automatically when the page loads.

    Uses the session-scoped test images created by _test_images_session
    in conftest.py, which ensures images exist before app construction.
    """

    def test_images_displayed_on_initial_load(self, page, app_url):
        """Images in input/ must be visible without user interaction.

        Steps:
        1. Navigate to the app (images already in input/ from session fixture)
        2. Assert the image editors row is visible
        3. Assert each rectangle tool canvas has rendered with non-zero dimensions
        """
        navigate(page, app_url)

        # The rectangle tool canvases should be visible since images were auto-loaded
        canvas1 = page.locator("#rect-tool-1 canvas")
        canvas1.wait_for(state="visible", timeout=10000)
        assert canvas1.is_visible(), "Image 1 canvas should be visible on page load"

        canvas2 = page.locator("#rect-tool-2 canvas")
        canvas2.wait_for(state="visible", timeout=10000)
        assert canvas2.is_visible(), "Image 2 canvas should be visible on page load"

        # Verify canvases have non-zero dimensions
        for i, canvas in enumerate([canvas1, canvas2]):
            bbox = canvas.bounding_box()
            assert bbox is not None, f"Canvas {i} should have a bounding box"
            assert bbox["width"] > 0 and bbox["height"] > 0, (
                f"Canvas {i} should have non-zero dimensions, got {bbox}"
            )


class TestRefreshImagesButton:
    """Verify the Refresh Images button loads images on the first click."""

    def test_first_refresh_click_shows_images(self, page, app_url, test_images):
        """A single click on Refresh Images must render images in the editors.

        Steps:
        1. Place test images in input/
        2. Navigate to the app
        3. Click the Refresh Images button once
        4. Assert the rectangle tool canvases are visible with rendered images
        """
        navigate(page, app_url)

        refresh_btn = page.get_by_role("button", name="Refresh Images")
        refresh_btn.click()
        page.wait_for_timeout(5000)

        # After a single click, canvas should be visible
        canvas1 = page.locator("#rect-tool-1 canvas")
        canvas1.wait_for(state="visible", timeout=10000)
        assert canvas1.is_visible(), (
            "Image 1 canvas should be visible after first refresh click"
        )

        bbox = canvas1.bounding_box()
        assert bbox is not None, (
            "Image 1 canvas should have a bounding box (not stuck in processing)"
        )
        assert bbox["width"] > 0 and bbox["height"] > 0, (
            "Image 1 canvas should have non-zero dimensions after first click"
        )

    def test_first_click_loads_both_editors(self, page, app_url, test_images):
        """A single click must load images into both editors when 2 images exist.

        Steps:
        1. Place 2 test images in input/
        2. Navigate and click Refresh Images once
        3. Assert both editors are visible with rendered canvases
        """
        navigate(page, app_url)

        refresh_btn = page.get_by_role("button", name="Refresh Images")
        refresh_btn.click()
        page.wait_for_timeout(5000)

        canvas1 = page.locator("#rect-tool-1 canvas")
        canvas1.wait_for(state="visible", timeout=10000)

        canvas2 = page.locator("#rect-tool-2 canvas")
        canvas2.wait_for(state="visible", timeout=10000)

        for i, canvas in enumerate([canvas1, canvas2]):
            bbox = canvas.bounding_box()
            assert bbox is not None, f"Canvas {i} should have a bounding box"
            assert bbox["width"] > 0 and bbox["height"] > 0, (
                f"Canvas {i} should have non-zero dimensions"
            )
