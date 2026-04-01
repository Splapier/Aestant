"""Image handling functionality for the chatbot application.

This module provides functions for:
- Scanning and loading images from a designated input directory
- Processing rectangle annotations from the drawing tool
- Burning rectangles into images using PIL.ImageDraw
- Preparing multimodal payloads for LLM submission

Usage Example:
    >>> from chatbot.image_handler import scan_input_directory, process_image_annotations
    >>> image_paths = scan_input_directory("./input", max_count=2)
    >>> annotated_img, rects = process_image_annotations(base_image, [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}])
"""

from typing import Any

import numpy as np

# Import functions from submodules
from chatbot.image_modules.image_loading import (
    scan_input_directory,
    load_image_as_numpy,
    load_images_from_directory,
    SUPPORTED_EXTENSIONS,
    DEFAULT_INPUT_DIR,
    MAX_IMAGES_DEFAULT,
)

from chatbot.image_modules.rectangle_processing import (
    detect_rectangles_from_layer,
    detect_all_rectangles,
    burn_rectangles_into_image,
    burn_rectangles_direct,
)

from chatbot.image_modules.rectangle_tool import (
    RectangleTool,
    RectangleToolConfig,
)


def process_image_annotations(
    background: np.ndarray | None,
    rectangles: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    """Process rectangle annotations to produce an annotated image.

    This is the main entry point for processing user annotations:
    Takes a background image and rectangle coordinate data, then burns
    the rectangle outlines directly into the image.

    Args:
        background: Base image as RGB numpy array (H x W x 3), or None.
        rectangles: List of rectangle dicts with 'x1', 'y1', 'x2', 'y2'
                    and optional 'color' key. Empty list if none drawn.

    Returns:
        Tuple of (annotated_image, rectangles) where:
        - annotated_image: Numpy array with rectangles burned in (or None)
        - rectangles: The rectangle list that was passed in

    Example:
        >>> rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "color": "#FF0000"}]
        >>> annotated_img, rects = process_image_annotations(base_image, rects)
        >>> if annotated_img is not None:
        ...     print(f"Drew {len(rects)} rectangles")
    """
    if background is None:
        return None, []

    if not rectangles:
        return background, []

    # Burn rectangles directly into the image
    annotated_image = burn_rectangles_direct(background, rectangles)

    return annotated_image, rectangles


def prepare_multimodal_payload(
    text: str,
    editor_values: list[dict | None],
) -> tuple[str, list[np.ndarray]]:
    """Prepare a multimodal payload for LLM submission.

    Processes rectangle tool outputs and combines them with text into a
    format suitable for multimodal LLM providers.

    Args:
        text: The user's text message.
        editor_values: List of editor output dicts (one per image).
                       Each dict has: {"background": np.ndarray, "rects": [...]}

    Returns:
        Tuple of (text, annotated_images) where:
        - text: Original user text (unchanged)
        - annotated_images: List of processed numpy arrays with burned-in rectangles

    Example:
        >>> text, images = prepare_multimodal_payload("Describe these", [editor1, editor2])
        >>> print(f"Text: {text}, Images: {len(images)}")
    """
    annotated_images = []

    for editor_value in editor_values:
        if editor_value is None:
            continue

        background = editor_value.get("background")
        if background is None:
            continue

        rectangles = editor_value.get("rects", [])
        annotated_img, _ = process_image_annotations(background, rectangles)
        if annotated_img is not None:
            annotated_images.append(annotated_img)

    return text, annotated_images


def image_to_base64(image_array: np.ndarray) -> str:
    """Convert a numpy image array to a base64-encoded data URL.

    Args:
        image_array: RGB numpy array (H x W x 3).

    Returns:
        Base64 data URL string suitable for multimodal LLM APIs.

    Example:
        >>> data_url = image_to_base64(img_array)
        >>> assert data_url.startswith("data:image/png;base64,")
    """
    import base64
    from io import BytesIO

    from PIL import Image

    # Convert to PIL Image
    pil_img = Image.fromarray(image_array)

    # Save to bytes buffer as PNG
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode to base64
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/png;base64,{base64_str}"


__all__ = [
    "scan_input_directory",
    "load_image_as_numpy",
    "load_images_from_directory",
    "detect_rectangles_from_layer",
    "detect_all_rectangles",
    "burn_rectangles_into_image",
    "burn_rectangles_direct",
    "process_image_annotations",
    "prepare_multimodal_payload",
    "image_to_base64",
    "RectangleTool",
    "RectangleToolConfig",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_INPUT_DIR",
    "MAX_IMAGES_DEFAULT",
]
