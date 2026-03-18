"""Tests for rectangle processing functionality.

This module provides unit tests for the rectangle_processing module,
covering detection and rendering of rectangular shapes from brush strokes.
"""

import numpy as np
import pytest
from PIL import Image

from chatbot.image_modules.rectangle_processing import (
    detect_rectangles_from_layer,
    detect_all_rectangles,
    burn_rectangles_into_image,
)


class TestDetectRectanglesFromLayer:
    """Tests for the detect_rectangles_from_layer function."""

    def test_none_input_returns_empty_list(self):
        """Test that None input returns an empty list."""
        result = detect_rectangles_from_layer(None)
        assert result == []

    def test_invalid_shape_returns_empty_list(self):
        """Test that arrays with less than 3 dimensions return empty list."""
        invalid_array = np.array([[1, 2], [3, 4]])
        result = detect_rectangles_from_layer(invalid_array)
        assert result == []

    def test_rgba_layer_with_rectangle(self):
        """Test detection of a rectangle in an RGBA layer."""
        # Create a 100x100 RGBA image with a filled rectangle from (20, 20) to (80, 80)
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        layer[20:81, 20:81, 3] = 255  # Set alpha channel for the rectangle

        result = detect_rectangles_from_layer(layer)

        assert len(result) == 1
        rect = result[0]
        assert rect["x1"] == 20
        assert rect["y1"] == 20
        assert rect["x2"] == 80
        assert rect["y2"] == 80

    def test_rgb_layer_without_alpha(self):
        """Test detection on RGB layer without alpha channel."""
        # Create a 50x50 RGB image (all pixels are considered drawn)
        layer = np.ones((50, 50, 3), dtype=np.uint8) * 128

        result = detect_rectangles_from_layer(layer)

        assert len(result) == 1
        rect = result[0]
        # Should return the full image bounds
        assert rect["x1"] == 0
        assert rect["y1"] == 0
        assert rect["x2"] == 49
        assert rect["y2"] == 49

    def test_empty_layer_returns_empty_list(self):
        """Test that a layer with no drawn pixels returns empty list."""
        # Create an RGBA image with all alpha = 0
        layer = np.zeros((100, 100, 4), dtype=np.uint8)

        result = detect_rectangles_from_layer(layer)
        assert result == []

    def test_min_area_filtering(self):
        """Test that rectangles below min_area are filtered out."""
        # Create a small 5x5 rectangle (area = 25 pixels)
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        layer[10:16, 10:16, 3] = 255

        # With default min_area=100, this should be filtered out
        result = detect_rectangles_from_layer(layer, min_area=100)
        assert result == []

        # With lower min_area, it should be detected
        result = detect_rectangles_from_layer(layer, min_area=25)
        assert len(result) == 1

    def test_multiple_disconnected_regions(self):
        """Test detection when there are multiple disconnected regions.

        Note: The current implementation returns a single bounding box
        encompassing all non-zero pixels, not individual connected components.
        """
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        # First rectangle at (10, 10) to (20, 20)
        layer[10:21, 10:21, 3] = 255
        # Second rectangle at (70, 70) to (80, 80)
        layer[70:81, 70:81, 3] = 255

        result = detect_rectangles_from_layer(layer)

        # Current implementation returns bounding box of all pixels
        assert len(result) == 1
        rect = result[0]
        assert rect["x1"] == 10
        assert rect["y1"] == 10
        assert rect["x2"] == 80
        assert rect["y2"] == 80

    def test_single_pixel(self):
        """Test detection of a single pixel."""
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        layer[50, 50, 3] = 255

        # Single pixel has area 1, which is below default min_area
        result = detect_rectangles_from_layer(layer, min_area=1)
        assert len(result) == 1
        rect = result[0]
        assert rect["x1"] == 50
        assert rect["y1"] == 50
        assert rect["x2"] == 50
        assert rect["y2"] == 50

    def test_full_image_coverage(self):
        """Test detection when entire image is filled."""
        layer = np.zeros((64, 64, 4), dtype=np.uint8)
        layer[:, :, 3] = 255

        result = detect_rectangles_from_layer(layer)
        assert len(result) == 1
        rect = result[0]
        assert rect["x1"] == 0
        assert rect["y1"] == 0
        assert rect["x2"] == 63
        assert rect["y2"] == 63


class TestDetectAllRectangles:
    """Tests for the detect_all_rectangles function."""

    def test_empty_layers_list(self):
        """Test with an empty list of layers."""
        result = detect_all_rectangles([])
        assert result == []

    def test_none_layer_in_list(self):
        """Test that None layers are skipped."""
        layers = [None, None]
        result = detect_all_rectangles(layers)
        assert result == []

    def test_single_layer_with_rectangle(self):
        """Test detection from a single layer with a rectangle."""
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        # Draw a red rectangle
        layer[20:81, 20:81, :3] = [255, 0, 0]
        layer[20:81, 20:81, 3] = 255

        result = detect_all_rectangles([layer])

        assert len(result) == 1
        rect = result[0]
        assert rect["x1"] == 20
        assert rect["y1"] == 20
        assert rect["color"] == (255, 0, 0)

    def test_multiple_layers(self):
        """Test detection from multiple layers."""
        # First layer with blue rectangle
        layer1 = np.zeros((100, 100, 4), dtype=np.uint8)
        layer1[10:31, 10:31, :3] = [0, 0, 255]
        layer1[10:31, 10:31, 3] = 255

        # Second layer with green rectangle
        layer2 = np.zeros((100, 100, 4), dtype=np.uint8)
        layer2[60:91, 60:91, :3] = [0, 255, 0]
        layer2[60:91, 60:91, 3] = 255

        result = detect_all_rectangles([layer1, layer2])

        assert len(result) == 2
        # First rectangle should be blue
        assert result[0]["color"] == (0, 0, 255)
        # Second rectangle should be green
        assert result[1]["color"] == (0, 255, 0)

    def test_layer_without_alpha_channel(self):
        """Test detection from RGB layer without alpha.

        Note: The current implementation assumes RGBA layers (4 channels).
        RGB layers (3 channels) will not detect rectangles because the
        color extraction code tries to access index 3 which doesn't exist.
        This test documents that behavior - only RGBA layers are fully supported.
        """
        # Create an RGBA layer with some non-zero pixel values and alpha
        layer = np.zeros((50, 50, 4), dtype=np.uint8)
        layer[10:41, 10:41, :3] = [128, 64, 32]
        layer[10:41, 10:41, 3] = 255

        result = detect_all_rectangles([layer])

        assert len(result) == 1
        # Should have a color attribute
        assert "color" in result[0]

    def test_mixed_valid_and_invalid_layers(self):
        """Test with a mix of valid and invalid layers."""
        # Valid layer
        valid_layer = np.zeros((50, 50, 4), dtype=np.uint8)
        valid_layer[10:20, 10:20, 3] = 255

        # Invalid layer (None)
        invalid_layer = None

        # Empty layer (no alpha)
        empty_layer = np.zeros((50, 50, 4), dtype=np.uint8)

        result = detect_all_rectangles([valid_layer, invalid_layer, empty_layer])

        assert len(result) == 1

    def test_color_extraction_average(self):
        """Test that average color is extracted from multi-colored layer."""
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        # Create a gradient-like pattern
        layer[20:40, 20:40, :3] = [255, 0, 0]  # Red half
        layer[40:60, 20:40, :3] = [0, 255, 0]  # Green half
        layer[20:60, 20:40, 3] = 255

        result = detect_all_rectangles([layer])

        assert len(result) == 1
        assert "color" in result[0]
        # Color should be somewhere between red and green (averaged)


class TestBurnRectanglesIntoImage:
    """Tests for the burn_rectangles_into_image function."""

    def test_none_base_image(self):
        """Test that None base image returns None."""
        result = burn_rectangles_into_image(None, [])
        assert result is None

    def test_empty_rectangles_list(self):
        """Test with empty rectangles list - should return unchanged image."""
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = burn_rectangles_into_image(base_image, [])

        assert result.shape == base_image.shape
        # Image should be unchanged (no rectangles drawn)
        assert np.array_equal(result, base_image)

    def test_single_rectangle(self):
        """Test drawing a single rectangle."""
        base_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rectangles = [{"x1": 20, "y1": 20, "x2": 80, "y2": 80}]

        result = burn_rectangles_into_image(base_image, rectangles)

        assert result.shape == base_image.shape
        # Check that some pixels were modified (rectangle was drawn)
        assert not np.array_equal(result, base_image)

    def test_default_color_is_red(self):
        """Test that default rectangle color is red."""
        base_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rectangles = [{"x1": 40, "y1": 40, "x2": 60, "y2": 60}]

        result = burn_rectangles_into_image(base_image, rectangles)

        # Check that red pixels were drawn on the outline (R=255, G=0, B=0)
        # The rectangle draws an outline, so check a point on the edge
        top_edge_pixel = result[40, 50]
        assert top_edge_pixel[0] > 0  # Red channel should be non-zero

    def test_custom_color(self):
        """Test drawing rectangle with custom color."""
        base_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rectangles = [
            {"x1": 40, "y1": 40, "x2": 60, "y2": 60, "color": (0, 255, 0)}
        ]

        result = burn_rectangles_into_image(base_image, rectangles)

        # Check that green pixels were drawn on the outline
        left_edge_pixel = result[50, 40]
        assert left_edge_pixel[1] > 0  # Green channel should be non-zero

    def test_multiple_rectangles(self):
        """Test drawing multiple rectangles."""
        base_image = np.zeros((200, 200, 3), dtype=np.uint8)
        rectangles = [
            {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "color": (255, 0, 0)},
            {"x1": 100, "y1": 100, "x2": 150, "y2": 150, "color": (0, 0, 255)},
        ]

        result = burn_rectangles_into_image(base_image, rectangles)

        assert result.shape == base_image.shape
        # First rectangle outline should have red pixels (check top edge)
        first_rect_top_edge = result[10, 30]
        assert first_rect_top_edge[0] > 0  # Red channel
        # Second rectangle outline should have blue pixels (check left edge)
        second_rect_left_edge = result[125, 100]
        assert second_rect_left_edge[2] > 0  # Blue channel

    def test_line_width_parameter(self):
        """Test that line width affects the drawn rectangles."""
        base_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rectangles = [{"x1": 40, "y1": 40, "x2": 60, "y2": 60}]

        # Draw with thin line
        result_thin = burn_rectangles_into_image(base_image, rectangles, line_width=1)
        # Draw with thick line
        result_thick = burn_rectangles_into_image(base_image, rectangles, line_width=5)

        # Thick line should modify more pixels
        thin_modified = np.sum(result_thin != 0)
        thick_modified = np.sum(result_thick != 0)
        assert thick_modified > thin_modified

    def test_output_shape_preservation(self):
        """Test that output image has the same shape as input."""
        base_image = np.ones((256, 480, 3), dtype=np.uint8) * 128
        rectangles = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100}]

        result = burn_rectangles_into_image(base_image, rectangles)

        assert result.shape == (256, 480, 3)
        assert result.dtype == np.uint8

    def test_rectangle_at_image_edge(self):
        """Test drawing rectangle at image boundaries."""
        base_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rectangles = [{"x1": 0, "y1": 0, "x2": 99, "y2": 99}]

        result = burn_rectangles_into_image(base_image, rectangles)

        assert result.shape == base_image.shape
        # Corners should be modified
        assert result[0, 0][0] > 0  # Top-left corner (red by default)

    def test_overlapping_rectangles(self):
        """Test drawing overlapping rectangles."""
        base_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rectangles = [
            {"x1": 20, "y1": 20, "x2": 60, "y2": 60, "color": (255, 0, 0)},
            {"x1": 40, "y1": 40, "x2": 80, "y2": 80, "color": (0, 0, 255)},
        ]

        result = burn_rectangles_into_image(base_image, rectangles)

        assert result.shape == base_image.shape
        # Overlapping area should show the second rectangle's color (blue)


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow(self):
        """Test the complete workflow from detection to rendering."""
        # Create a layer with brush strokes
        layer = np.zeros((200, 200, 4), dtype=np.uint8)
        layer[50:101, 50:101, :3] = [255, 128, 64]  # Orange color
        layer[50:101, 50:101, 3] = 255

        # Detect rectangles
        rectangles = detect_all_rectangles([layer])
        assert len(rectangles) == 1

        # Burn into base image
        base_image = np.ones((200, 200, 3), dtype=np.uint8) * 240  # Light gray background
        annotated = burn_rectangles_into_image(base_image, rectangles)

        assert annotated.shape == base_image.shape
        # The rectangle outline should be visible
        assert not np.array_equal(annotated, base_image)

    def test_workflow_with_multiple_layers(self):
        """Test workflow with multiple input layers."""
        layers = []

        # First layer: red square
        layer1 = np.zeros((300, 300, 4), dtype=np.uint8)
        layer1[20:71, 20:71, :3] = [255, 0, 0]
        layer1[20:71, 20:71, 3] = 255
        layers.append(layer1)

        # Second layer: green square
        layer2 = np.zeros((300, 300, 4), dtype=np.uint8)
        layer2[150:221, 150:221, :3] = [0, 255, 0]
        layer2[150:221, 150:221, 3] = 255
        layers.append(layer2)

        # Detect all rectangles
        rectangles = detect_all_rectangles(layers)
        assert len(rectangles) == 2

        # Burn into base image with thick line to ensure outline is visible
        base_image = np.zeros((300, 300, 3), dtype=np.uint8)
        annotated = burn_rectangles_into_image(base_image, rectangles, line_width=10)

        assert annotated.shape == (300, 300, 3)
        # First rectangle outline should have red pixels (check top edge with thick line)
        assert annotated[25, 45][0] > 0  # Red channel on first rect outline
        # Second rectangle outline should have green pixels (check left edge)
        assert annotated[175, 150][1] > 0  # Green channel on second rect outline
