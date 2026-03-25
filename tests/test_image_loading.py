"""Tests for image loading functionality.

This module provides unit tests for the image_loading module,
covering directory scanning, image loading as numpy arrays,
and the combined scan-and-load workflow using temp directories.
"""

import pytest
import os
from pathlib import Path

import numpy as np
from PIL import Image

from chatbot.image_modules.image_loading import (
    scan_input_directory,
    load_image_as_numpy,
    load_images_from_directory,
    SUPPORTED_EXTENSIONS,
    DEFAULT_INPUT_DIR,
    MAX_IMAGES_DEFAULT,
)


@pytest.fixture
def image_dir(tmp_path):
    """Create a temporary directory with test images."""
    # Create a PNG image
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(tmp_path / "image_a.png")

    # Create a JPG image
    img = Image.new("RGB", (10, 10), color=(0, 255, 0))
    img.save(tmp_path / "image_b.jpg")

    # Create a BMP image
    img = Image.new("RGB", (10, 10), color=(0, 0, 255))
    img.save(tmp_path / "image_c.bmp")

    # Create a non-image file (should be ignored)
    (tmp_path / "readme.txt").write_text("not an image")

    return tmp_path


class TestScanInputDirectory:
    """Tests for the scan_input_directory function."""

    def test_nonexistent_dir_returns_empty(self):
        """Test that a non-existent directory returns empty list."""
        result = scan_input_directory("/nonexistent/path")
        assert result == []

    def test_file_path_returns_empty(self):
        """Test that a file path (not a directory) returns empty list."""
        with pytest.MonkeyPatch.context() as m:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                filepath = f.name
            try:
                result = scan_input_directory(filepath)
                assert result == []
            finally:
                os.unlink(filepath)

    def test_empty_dir_returns_empty(self, tmp_path):
        """Test that an empty directory returns empty list."""
        result = scan_input_directory(tmp_path)
        assert result == []

    def test_finds_supported_images(self, image_dir):
        """Test that all supported image formats are found."""
        result = scan_input_directory(image_dir, max_count=10)
        names = [Path(p).name for p in result]
        assert "image_a.png" in names
        assert "image_b.jpg" in names
        assert "image_c.bmp" in names

    def test_ignores_unsupported_files(self, image_dir):
        """Test that non-image files are ignored."""
        result = scan_input_directory(image_dir, max_count=10)
        for path in result:
            assert Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def test_sorted_by_filename(self, image_dir):
        """Test that results are sorted alphabetically by filename."""
        result = scan_input_directory(image_dir, max_count=10)
        names = [Path(p).name for p in result]
        assert names == sorted(names)

    def test_respects_max_count(self, image_dir):
        """Test that max_count limits the number of results."""
        result = scan_input_directory(image_dir, max_count=2)
        assert len(result) == 2

    def test_default_max_count_is_2(self, image_dir):
        """Test that default max_count limits to 2 images."""
        result = scan_input_directory(image_dir)
        assert len(result) == 2

    def test_returns_absolute_paths(self, image_dir):
        """Test that returned paths are absolute."""
        result = scan_input_directory(image_dir, max_count=1)
        assert len(result) == 1
        assert os.path.isabs(result[0])

    def test_case_insensitive_extensions(self, tmp_path):
        """Test that uppercase extensions are recognized."""
        img = Image.new("RGB", (10, 10), color=(128, 128, 128))
        img.save(tmp_path / "IMAGE.PNG")
        img.save(tmp_path / "PHOTO.JPEG")
        result = scan_input_directory(tmp_path, max_count=10)
        names = [Path(p).name for p in result]
        assert "IMAGE.PNG" in names
        assert "PHOTO.JPEG" in names


class TestLoadImageAsNumpy:
    """Tests for the load_image_as_numpy function."""

    def test_loads_valid_image_as_rgb(self, tmp_path):
        """Test that a valid image is loaded as RGB numpy array."""
        img = Image.new("RGB", (20, 30), color=(100, 150, 200))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        result = load_image_as_numpy(str(img_path))
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (30, 20, 3)
        assert result.dtype == np.uint8
        # Verify pixel color
        assert result[0, 0, 0] == 100
        assert result[0, 0, 1] == 150
        assert result[0, 0, 2] == 200

    def test_rgba_converted_to_rgb(self, tmp_path):
        """Test that RGBA images are converted to RGB."""
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        img_path = tmp_path / "rgba.png"
        img.save(img_path)

        result = load_image_as_numpy(str(img_path))
        assert result is not None
        assert result.shape == (10, 10, 3)

    def test_invalid_path_returns_none(self):
        """Test that a non-existent file returns None."""
        result = load_image_as_numpy("/nonexistent/image.png")
        assert result is None

    def test_corrupted_file_returns_none(self, tmp_path):
        """Test that a corrupted file returns None."""
        bad_file = tmp_path / "bad.png"
        bad_file.write_text("this is not an image")
        result = load_image_as_numpy(str(bad_file))
        assert result is None

    def test_loads_jpg_image(self, tmp_path):
        """Test loading a JPG image."""
        img = Image.new("RGB", (15, 25), color=(50, 100, 150))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        result = load_image_as_numpy(str(img_path))
        assert result is not None
        assert result.shape[0] == 25
        assert result.shape[1] == 15
        assert result.shape[2] == 3

    def test_returns_numpy_array(self, tmp_path):
        """Test that return type is always a numpy array when successful."""
        img = Image.new("RGB", (5, 5), color=(0, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        result = load_image_as_numpy(str(img_path))
        assert isinstance(result, np.ndarray)


class TestLoadImagesFromDirectory:
    """Tests for the load_images_from_directory function."""

    def test_nonexistent_dir_returns_empty(self):
        """Test that non-existent directory returns empty list."""
        result = load_images_from_directory("/nonexistent/path")
        assert result == []

    def test_loads_all_valid_images(self, image_dir):
        """Test that all valid images are loaded as numpy arrays."""
        result = load_images_from_directory(image_dir, max_count=10)
        assert len(result) > 0
        for img in result:
            assert isinstance(img, np.ndarray)
            assert img.ndim == 3
            assert img.shape[2] == 3

    def test_skips_failed_loads(self, tmp_path):
        """Test that failed loads are excluded from results."""
        # Create a valid image
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img.save(tmp_path / "good.png")

        # Create a corrupted file with a supported extension
        (tmp_path / "bad.png").write_text("not an image")

        result = load_images_from_directory(tmp_path, max_count=10)
        assert len(result) == 1

    def test_respects_max_count(self, tmp_path):
        """Test that max_count limits the number of loaded images."""
        for i in range(5):
            img = Image.new("RGB", (10, 10), color=(i * 50, 0, 0))
            img.save(tmp_path / f"img_{i}.png")

        result = load_images_from_directory(tmp_path, max_count=3)
        assert len(result) == 3

    def test_returns_empty_for_non_image_dir(self, tmp_path):
        """Test that directory with only non-image files returns empty."""
        (tmp_path / "file.txt").write_text("text")
        (tmp_path / "file.csv").write_text("a,b,c")
        result = load_images_from_directory(tmp_path, max_count=10)
        assert result == []


class TestConstants:
    """Tests for module-level constants."""

    def test_supported_extensions_contains_common_formats(self):
        """Test that common image formats are in supported extensions."""
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".gif" in SUPPORTED_EXTENSIONS
        assert ".bmp" in SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS

    def test_default_input_dir(self):
        """Test default input directory value."""
        assert DEFAULT_INPUT_DIR == "input"

    def test_max_images_default(self):
        """Test default max images value."""
        assert MAX_IMAGES_DEFAULT == 2
