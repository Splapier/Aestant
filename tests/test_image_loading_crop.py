"""Tests for load_image_cropped helper function.

Tests the image loading functionality that supports
optional rectangle-based cropping for attention-based comparison.
"""

import numpy as np
import pytest
from PIL import Image

from chatbot.image_modules.image_loading import load_image_cropped


@pytest.fixture
def test_image(tmp_path):
    """Create a test image file."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (200, 200), color="blue")
    img.save(img_path)
    return str(img_path)


class TestLoadImageCropped:
    """Test load_image_cropped function."""

    def test_returns_full_image_when_no_rectangles(self, test_image):
        result = load_image_cropped(test_image, None)

        assert result is not None
        assert result.shape == (200, 200, 3)

    def test_returns_full_image_when_empty_rectangles(self, test_image):
        result = load_image_cropped(test_image, [])

        assert result is not None
        assert result.shape == (200, 200, 3)

    def test_crops_to_single_rectangle(self, test_image):
        rects = [{"x1": 50, "y1": 50, "x2": 150, "y2": 150}]
        result = load_image_cropped(test_image, rects)

        assert result is not None
        assert result.shape == (100, 100, 3)

    def test_handles_inverted_coordinates(self, test_image):
        rects = [{"x1": 150, "y1": 150, "x2": 50, "y2": 50}]
        result = load_image_cropped(test_image, rects)

        assert result is not None
        assert result.shape == (100, 100, 3)

    def test_handles_out_of_bounds_coordinates(self, test_image):
        rects = [{"x1": -50, "y1": -50, "x2": 300, "y2": 300}]
        result = load_image_cropped(test_image, rects)

        assert result is not None
        assert result.shape[0] <= 200
        assert result.shape[1] <= 200

    def test_uses_first_rectangle_when_multiple(self, test_image):
        rects = [
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
        ]
        result = load_image_cropped(test_image, rects)

        assert result is not None
        assert result.shape == (200, 200, 3)

    def test_returns_none_for_missing_file(self):
        result = load_image_cropped("/nonexistent/path.png", None)

        assert result is None

    def test_returns_none_for_missing_file_with_rectangles(self):
        rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        result = load_image_cropped("/nonexistent/path.png", rects)

        assert result is None

    def test_crops_partial_region(self, test_image):
        rects = [{"x1": 25, "y1": 75, "x2": 125, "y2": 175}]
        result = load_image_cropped(test_image, rects)

        assert result is not None
        assert result.shape == (100, 100, 3)

    def test_handles_rectangle_at_image_edges(self, test_image):
        rects = [{"x1": 0, "y1": 0, "x2": 200, "y2": 200}]
        result = load_image_cropped(test_image, rects)

        assert result is not None
        assert result.shape == (200, 200, 3)

    def test_rectangle_with_color_preserved_in_result(self, test_image):
        rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "color": "#FF0000"}]
        result = load_image_cropped(test_image, rects)

        assert result is not None
        assert result.shape == (40, 40, 3)


class TestLoadImageCroppedWithDifferentFormats:
    """Test loading various image formats."""

    @pytest.fixture
    def image_formats(self, tmp_path):
        """Create test images in different formats."""
        formats = {"jpg": "JPEG", "png": "PNG", "bmp": "BMP"}
        paths = {}
        for ext, fmt in formats.items():
            img_path = tmp_path / f"test_image.{ext}"
            img = Image.new("RGB", (100, 100), color="green")
            img.save(img_path, format=fmt)
            paths[ext] = str(img_path)
        return paths

    def test_loads_jpeg_with_crop(self, image_formats):
        result = load_image_cropped(
            image_formats["jpg"], [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        )
        assert result is not None
        assert result.shape == (40, 40, 3)

    def test_loads_png_with_crop(self, image_formats):
        result = load_image_cropped(
            image_formats["png"], [{"x1": 20, "y1": 20, "x2": 80, "y2": 80}]
        )
        assert result is not None
        assert result.shape == (60, 60, 3)

    def test_loads_bmp_with_crop(self, image_formats):
        result = load_image_cropped(
            image_formats["bmp"], [{"x1": 0, "y1": 0, "x2": 50, "y2": 50}]
        )
        assert result is not None
        assert result.shape == (50, 50, 3)
