"""Playwright tests for navigation changes and embedding UI.

Tests:
- Previous/Next button navigation
- Default loading to first untagged image
- Embedding controls (dropdown, buttons, status)
"""

import os

import pytest
from playwright.sync_api import expect
from PIL import Image

from conftest import INPUT_DIR
from helpers import navigate


@pytest.fixture()
def multiple_test_images():
    """Create multiple test images for navigation testing."""
    paths = []
    colors = ["red", "blue", "green", "yellow", "purple"]
    for i, color in enumerate(colors):
        img = Image.new("RGB", (50, 50), color=color)
        p = INPUT_DIR / f"nav_test_{i}.png"
        img.save(p)
        paths.append(p)
    yield [str(p) for p in paths]
    for p in paths:
        if p.exists():
            os.remove(p)


@pytest.fixture()
def some_tagged_images(multiple_test_images):
    """Create some pre-tagged images to test 'skip to first untagged'."""
    import yaml
    from pathlib import Path

    dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    tagged = []
    for i in range(2):
        yaml_path = dataset_dir / f"nav_test_{i}.png.yaml"
        data = {"image_path": str(INPUT_DIR / f"nav_test_{i}.png"), "tags": {}}
        yaml_path.write_text(yaml.dump(data))
        tagged.append(yaml_path)

    yield multiple_test_images

    for p in tagged:
        if p.exists():
            p.unlink()


class TestNavigationButtons:
    """Test Previous/Next button navigation."""

    def test_previous_button_visible(self, page, app_url):
        """Previous button should be visible in tagging tab."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        prev_button = page.locator("button", has_text="Previous")
        expect(prev_button).to_be_visible()

    def test_next_button_visible(self, page, app_url):
        """Next button should be visible in tagging tab."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        next_button = page.locator("button", has_text="Next")
        expect(next_button).to_be_visible()

    def test_delete_and_next_renamed_to_next(self, page, app_url):
        """Old 'Delete & Next' should be renamed to just 'Next'."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        old_button = page.locator("button", has_text="Delete & Next")
        expect(old_button).to_have_count(0)


class TestDefaultNavigation:
    """Test that app loads first untagged image by default."""

    def test_loads_first_image_when_none_tagged(
        self, page, app_url, multiple_test_images
    ):
        """When no images are tagged, should start at image 1."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        counter = page.locator("text=Image 1 of")
        expect(counter).to_be_visible(timeout=5000)

    def test_loads_first_untagged_when_some_tagged(
        self, page, app_url, some_tagged_images
    ):
        """When some images are tagged, should jump to first untagged."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(2000)

        counter = page.locator("text=Image 3 of")
        expect(counter).to_be_visible(timeout=5000)


class TestEmbeddingUI:
    """Test embedding controls in tagging tab."""

    def test_embedding_controls_visible(self, page, app_url):
        """Embedding controls should be visible in tagging tab."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        embedding_dropdown = page.locator("text=Embed")
        expect(embedding_dropdown).to_be_visible(timeout=5000)

    def test_embedding_options_available(self, page, app_url):
        """Embedding dropdown should have Image, Tags, Both options."""
        navigate(page, app_url)

        tagging_tab = page.get_by_role("tab", name="Tagging")
        tagging_tab.click()
        page.wait_for_timeout(1000)

        embed_button = page.get_by_role("button", name="Create Embeddings")
        embed_button.click()
        page.wait_for_timeout(500)

        expect(page.get_by_role("option", name="Embed Image")).to_be_visible()
        expect(page.get_by_role("option", name="Embed Tags")).to_be_visible()
        expect(page.get_by_role("option", name="Embed Both")).to_be_visible()
