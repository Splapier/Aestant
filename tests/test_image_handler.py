"""Tests for image handling functionality.

This module provides unit tests for the image_handler module,
covering annotation processing, multimodal payload preparation,
and base64 conversion.
"""

import pytest
import numpy as np

from chatbot.image_handler import (
    process_image_annotations,
    prepare_multimodal_payload,
    image_to_base64,
)


class TestProcessImageAnnotations:
    """Tests for the process_image_annotations function."""

    def test_none_background_returns_none_and_empty(self):
        """Test that None background returns (None, [])."""
        img, rects = process_image_annotations(None)
        assert img is None
        assert rects == []

    def test_none_rects_returns_base_image(self):
        """Test that None rects returns base image unchanged."""
        bg = np.ones((50, 50, 3), dtype=np.uint8) * 128
        img, rects = process_image_annotations(bg, None)
        assert img is not None
        assert np.array_equal(img, bg)
        assert rects == []

    def test_empty_rects_returns_base_image(self):
        """Test that empty rects returns base image unchanged."""
        bg = np.ones((50, 50, 3), dtype=np.uint8) * 128
        img, rects = process_image_annotations(bg, [])
        assert img is not None
        assert np.array_equal(img, bg)
        assert rects == []

    def test_with_rectangles_returns_annotated_image(self):
        """Test that valid rectangles produce a different image."""
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 200
        rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        img, returned_rects = process_image_annotations(bg, rects)
        assert img is not None
        assert len(returned_rects) == 1
        assert returned_rects[0]["x1"] == 10
        assert not np.array_equal(img, bg)

    def test_with_hex_color_rectangles(self):
        """Test that hex color rectangles are processed correctly."""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        rects = [{"x1": 20, "y1": 20, "x2": 80, "y2": 80, "color": "#00FF00"}]
        img, returned_rects = process_image_annotations(bg, rects)
        assert img is not None
        # Green pixels should be on the outline
        assert img[20, 50][1] > 0  # Green channel on top edge

    def test_preserves_rectangles_list(self):
        """Test that the returned rectangles list matches input."""
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 128
        rects = [
            {"x1": 5, "y1": 5, "x2": 30, "y2": 30, "color": "#FF0000"},
            {"x1": 40, "y1": 40, "x2": 90, "y2": 90, "color": "#0000FF"},
        ]
        img, returned_rects = process_image_annotations(bg, rects)
        assert len(returned_rects) == 2
        assert returned_rects[0]["color"] == "#FF0000"
        assert returned_rects[1]["color"] == "#0000FF"


class TestPrepareMultimodalPayload:
    """Tests for the prepare_multimodal_payload function."""

    def test_text_unchanged(self):
        """Test that text is returned unchanged."""
        text, images = prepare_multimodal_payload("Hello world", [])
        assert text == "Hello world"

    def test_empty_editor_list(self):
        """Test that empty editor list produces no images."""
        text, images = prepare_multimodal_payload("Test", [])
        assert text == "Test"
        assert images == []

    def test_none_editors_skipped(self):
        """Test that None entries in editor list are skipped."""
        text, images = prepare_multimodal_payload("Test", [None, None])
        assert images == []

    def test_valid_editors_produce_images(self):
        """Test that valid editors produce annotated images."""
        bg = np.ones((50, 50, 3), dtype=np.uint8) * 150
        rects = [{"x1": 10, "y1": 10, "x2": 30, "y2": 30}]
        editor = {"background": bg, "rects": rects}
        text, images = prepare_multimodal_payload("Describe this", [editor])
        assert text == "Describe this"
        assert len(images) == 1
        assert isinstance(images[0], np.ndarray)
        assert images[0].shape == (50, 50, 3)

    def test_mixed_none_and_valid_editors(self):
        """Test that mix of None and valid editors only includes valid ones."""
        bg = np.ones((20, 20, 3), dtype=np.uint8) * 100
        rects = [{"x1": 5, "y1": 5, "x2": 15, "y2": 15}]
        editor = {"background": bg, "rects": rects}
        text, images = prepare_multimodal_payload("Test", [None, editor, None])
        assert len(images) == 1

    def test_editor_without_background_skipped(self):
        """Test that editors with no background are skipped."""
        editor = {"background": None, "rects": [{"x1": 5, "y1": 5, "x2": 15, "y2": 15}]}
        text, images = prepare_multimodal_payload("Test", [editor])
        assert len(images) == 0

    def test_editor_without_rects_still_included(self):
        """Test that editors with background but no rects still produce an image."""
        bg = np.ones((30, 30, 3), dtype=np.uint8) * 128
        editor = {"background": bg, "rects": []}
        text, images = prepare_multimodal_payload("Test", [editor])
        assert len(images) == 1
        assert np.array_equal(images[0], bg)


class TestImageToBase64:
    """Tests for the image_to_base64 function."""

    def test_returns_data_url_prefix(self):
        """Test that return value has correct data URL prefix."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = image_to_base64(img)
        assert result.startswith("data:image/png;base64,")

    def test_valid_base64_content(self):
        """Test that base64 content is decodable."""
        import base64

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = image_to_base64(img)
        b64_data = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0

    def test_different_sizes(self):
        """Test conversion with different image sizes."""
        img_small = np.zeros((5, 5, 3), dtype=np.uint8)
        img_large = np.zeros((200, 200, 3), dtype=np.uint8)
        result_small = image_to_base64(img_small)
        result_large = image_to_base64(img_large)
        assert result_small.startswith("data:image/png;base64,")
        assert result_large.startswith("data:image/png;base64,")
        assert len(result_large) > len(result_small)

    def test_roundtrip_preserves_dimensions(self):
        """Test that decoding the base64 back to an image preserves dimensions."""
        import base64
        from io import BytesIO
        from PIL import Image

        img = np.full((30, 40, 3), 128, dtype=np.uint8)
        result = image_to_base64(img)
        b64_data = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        restored = np.array(Image.open(BytesIO(decoded)))
        assert restored.shape == (30, 40, 3)
