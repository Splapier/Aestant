"""Rectangle processing functionality for the chatbot application.

This module provides functions for:
- Detecting rectangular shapes from brush strokes (layer alpha channels)
- Burning rectangles into images using PIL.ImageDraw

Usage Example:
    >>> from chatbot.image_modules.rectangle_processing import detect_all_rectangles, burn_rectangles_into_image
    >>> rectangles = detect_all_rectangles(layers)
    >>> annotated = burn_rectangles_into_image(base_image, rectangles)
"""

from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def detect_rectangles_from_layer(
    layer_array: np.ndarray, min_area: int = 100
) -> list[dict[str, int]]:
    """Detect rectangular shapes from a brush stroke layer.

    This function analyzes the alpha channel of a layer to find non-zero pixels,
    then computes bounding boxes around connected regions that exceed a minimum area.

    Args:
        layer_array: RGBA numpy array (H x W x 4) containing brush strokes.
        min_area: Minimum pixel area for a region to be considered a rectangle.

    Returns:
        List of dictionaries with keys 'x1', 'y1', 'x2', 'y2' representing
        bounding box coordinates (top-left and bottom-right corners).

    Example:
        >>> rectangles = detect_rectangles_from_layer(layer_array)
        >>> for rect in rectangles:
        ...     print(f"Box: ({rect['x1']}, {rect['y1']}) to ({rect['x2']}, {rect['y2']})")
    """
    if layer_array is None or len(layer_array.shape) < 3:
        return []

    # Extract alpha channel (4th channel in RGBA)
    if layer_array.shape[2] == 4:
        alpha = layer_array[:, :, 3]
    else:
        # If no alpha, assume all pixels are drawn
        alpha = np.ones(layer_array.shape[:2], dtype=np.uint8) * 255

    # Threshold to get binary mask of drawn pixels
    mask = alpha > 0

    if not np.any(mask):
        return []

    # Find connected components using simple flood-fill approach
    # For simplicity, we'll compute the bounding box of all non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return []

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Calculate area
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    area = height * width

    if area < min_area:
        return []

    return [{"x1": int(x_min), "y1": int(y_min), "x2": int(x_max), "y2": int(y_max)}]


def detect_all_rectangles(
    layers: list[np.ndarray], min_area: int = 100
) -> list[dict[str, Any]]:
    """Detect rectangles from all layers in an ImageEditor output.

    Args:
        layers: List of RGBA numpy arrays representing drawn layers.
        min_area: Minimum pixel area for detection threshold.

    Returns:
        List of rectangle dictionaries with coordinates and optional color info.

    Example:
        >>> rects = detect_all_rectangles(layers)
        >>> print(f"Found {len(rects)} rectangles")
    """
    all_rectangles = []

    for i, layer in enumerate(layers):
        if layer is None or len(layer.shape) < 3:
            continue

        rectangles = detect_rectangles_from_layer(layer, min_area)

        # Extract dominant color from the layer (for colored rectangles)
        if layer.shape[2] >= 3 and np.any(layer[:, :, :3].flatten()):
            # Get average color of non-alpha pixels
            mask = layer[:, :, 3] > 0
            if np.any(mask):
                avg_color_arr = np.mean(layer[mask], axis=0)[:3]
                avg_color = tuple(int(c) for c in avg_color_arr)
            else:
                avg_color = (255, 0, 0)  # Default red
        else:
            avg_color = (255, 0, 0)

        for rect in rectangles:
            rect["color"] = avg_color
            all_rectangles.append(rect)

    return all_rectangles


def burn_rectangles_into_image(
    base_image: np.ndarray,
    rectangles: list[dict[str, Any]],
    line_width: int = 3,
) -> np.ndarray:
    """Draw (burn) rectangles directly onto an image using PIL.

    This function takes a base image and draws the specified rectangles
    as outlines on top of it, returning the modified image.

    Args:
        base_image: RGB numpy array (H x W x 3) representing the base image.
        rectangles: List of rectangle dicts with 'x1', 'y1', 'x2', 'y2' keys.
                    Optional 'color' key for custom colors (default: red).
                    Color can be a hex string (e.g. "#FF0000") or RGB tuple.
        line_width: Width of the rectangle outline in pixels.

    Returns:
        Modified numpy array with rectangles drawn on top.

    Example:
        >>> annotated = burn_rectangles_into_image(img, [{"x1": 0, "y1": 0, "x2": 100, "y2": 100}])
        >>> assert annotated.shape == img.shape
    """
    if base_image is None:
        return None

    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(base_image)
    draw = ImageDraw.Draw(pil_img)

    for rect in rectangles:
        x1, y1 = rect["x1"], rect["y1"]
        x2, y2 = rect["x2"], rect["y2"]
        color = rect.get("color", (255, 0, 0))

        # Normalize hex color strings to RGB tuples
        if isinstance(color, str) and color.startswith("#"):
            color = hex_to_rgb(color)

        # Ensure coordinates are ordered (top-left to bottom-right)
        rx1, ry1 = min(x1, x2), min(y1, y2)
        rx2, ry2 = max(x1, x2), max(y1, y2)

        # Draw rectangle outline
        draw.rectangle([rx1, ry1, rx2, ry2], outline=color, width=line_width)

    return np.array(pil_img)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple.

    Args:
        hex_color: Color in "#RRGGBB" format.

    Returns:
        Tuple of (R, G, B) integers.

    Example:
        >>> hex_to_rgb("#FF0000")
        (255, 0, 0)
    """
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def burn_rectangles_direct(
    base_image: np.ndarray,
    rectangles: list[dict[str, Any]],
    line_width: int = 3,
) -> np.ndarray:
    """Burn rectangle outlines directly from coordinate data onto an image.

    This is a convenience wrapper around burn_rectangles_into_image() for
    use with the rectangle drawing tool, which provides coordinates directly
    rather than requiring detection from brush stroke layers.

    Args:
        base_image: RGB numpy array (H x W x 3).
        rectangles: List of dicts with 'x1', 'y1', 'x2', 'y2' and optional 'color'.
        line_width: Outline width in pixels.

    Returns:
        Annotated numpy array with rectangle outlines drawn.
    """
    return burn_rectangles_into_image(base_image, rectangles, line_width)


__all__ = [
    "detect_rectangles_from_layer",
    "detect_all_rectangles",
    "burn_rectangles_into_image",
    "burn_rectangles_direct",
    "hex_to_rgb",
]
