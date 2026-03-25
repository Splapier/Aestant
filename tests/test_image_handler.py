"""Tests for image handling functionality.

This module provides unit tests for the image_handler module,
covering image editor data parsing, annotation processing,
multimodal payload preparation, and base64 conversion.
"""

import pytest
import numpy as np

from chatbot.image_handler import (
    parse_image_editor_data,
    process_image_annotations,
    prepare_multimodal_payload,
    image_to_base64,
)


class TestParseImageEditorData:
    """Tests for the parse_image_editor_data function."""

    def test_none_returns_none_and_empty_list(self):
        """Test that None input returns (None, [])."""
        base, layers = parse_image_editor_data(None)
        assert base is None
        assert layers == []

    def test_extracts_background_and_layers(self):
        """Test that background and layers are extracted from editor dict."""
        bg = np.zeros((50, 50, 3), dtype=np.uint8)
        layer = np.ones((50, 50, 4), dtype=np.uint8)
        editor = {"background": bg, "layers": [layer]}
        base, layers = parse_image_editor_data(editor)
        assert base is bg
        assert len(layers) == 1
        assert np.array_equal(layers[0], layer)

    def test_single_array_layer_wrapped_in_list(self):
        """Test that a single numpy array layer is wrapped in a list."""
        bg = np.zeros((10, 10, 3), dtype=np.uint8)
        layer = np.ones((10, 10, 4), dtype=np.uint8)
        editor = {"background": bg, "layers": layer}
        base, layers = parse_image_editor_data(editor)
        assert base is bg
        assert len(layers) == 1
        assert isinstance(layers, list)

    def test_missing_layers_returns_empty_list(self):
        """Test that missing layers key returns empty list."""
        bg = np.zeros((10, 10, 3), dtype=np.uint8)
        editor = {"background": bg}
        base, layers = parse_image_editor_data(editor)
        assert base is bg
        assert layers == []

    def test_none_background(self):
        """Test that None background is returned as-is."""
        editor = {"background": None, "layers": [np.ones((10, 10, 4), dtype=np.uint8)]}
        base, layers = parse_image_editor_data(editor)
        assert base is None
        assert len(layers) == 1

    def test_empty_layers_list(self):
        """Test that empty layers list is returned as-is."""
        bg = np.zeros((10, 10, 3), dtype=np.uint8)
        editor = {"background": bg, "layers": []}
        base, layers = parse_image_editor_data(editor)
        assert base is bg
        assert layers == []


class TestProcessImageAnnotations:
    """Tests for the process_image_annotations function."""

    def test_none_editor_returns_none_and_empty(self):
        """Test that None editor returns (None, [])."""
        img, rects = process_image_annotations(None)
        assert img is None
        assert rects == []

    def test_no_base_image_returns_none(self):
        """Test that editor with no background returns (None, [])."""
        layer = np.zeros((10, 10, 4), dtype=np.uint8)
        editor = {"background": None, "layers": [layer]}
        img, rects = process_image_annotations(editor)
        assert img is None
        assert rects == []

    def test_no_rectangles_returns_base_image(self):
        """Test that no annotations returns base image with empty rects."""
        bg = np.ones((50, 50, 3), dtype=np.uint8) * 128
        layer = np.zeros((50, 50, 4), dtype=np.uint8)  # No alpha => no rectangles
        editor = {"background": bg, "layers": [layer]}
        img, rects = process_image_annotations(editor)
        assert img is not None
        assert np.array_equal(img, bg)
        assert rects == []

    def test_with_annotations_returns_annotated_image(self):
        """Test that valid annotations produce a different image with rectangles."""
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 200
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        layer[10:30, 10:30, 3] = 255  # Draw rectangle in alpha channel
        editor = {"background": bg, "layers": [layer]}
        img, rects = process_image_annotations(editor)
        assert img is not None
        assert len(rects) == 1
        assert rects[0]["x1"] == 10
        assert rects[0]["y1"] == 10
        assert rects[0]["x2"] == 29
        assert rects[0]["y2"] == 29

    def test_custom_min_area(self):
        """Test that min_area parameter filters small annotations."""
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 200
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        # Small 3x3 rect => area=9, below min_area=20
        layer[5:8, 5:8, 3] = 255
        editor = {"background": bg, "layers": [layer]}
        img, rects = process_image_annotations(editor, min_area=20)
        assert np.array_equal(img, bg)
        assert rects == []


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
        layer = np.zeros((50, 50, 4), dtype=np.uint8)
        layer[10:30, 10:30, 3] = 255
        editor = {"background": bg, "layers": [layer]}
        text, images = prepare_multimodal_payload("Describe this", [editor])
        assert text == "Describe this"
        assert len(images) == 1
        assert isinstance(images[0], np.ndarray)
        assert images[0].shape == (50, 50, 3)

    def test_mixed_none_and_valid_editors(self):
        """Test that mix of None and valid editors only includes valid ones."""
        bg = np.ones((20, 20, 3), dtype=np.uint8) * 100
        layer = np.zeros((20, 20, 4), dtype=np.uint8)
        layer[5:15, 5:15, 3] = 255
        editor = {"background": bg, "layers": [layer]}
        text, images = prepare_multimodal_payload("Test", [None, editor, None])
        assert len(images) == 1

    def test_editor_without_annotations_excluded(self):
        """Test that editors with no annotations produce no image."""
        bg = np.ones((30, 30, 3), dtype=np.uint8) * 128
        layer = np.zeros((30, 30, 4), dtype=np.uint8)  # No alpha
        editor = {"background": bg, "layers": [layer]}
        text, images = prepare_multimodal_payload("Test", [editor])
        # No annotations => base image returned but still included
        assert len(images) == 1


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
        # Extract base64 part after prefix
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
        # Larger image should produce longer base64 string
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
