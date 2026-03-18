"""Image handling functionality for the chatbot application.

This module provides functions for:
- Scanning and loading images from a designated input directory
- Parsing annotation data from Gradio ImageEditor components
- Detecting rectangular shapes from brush strokes (layer alpha channels)
- Burning rectangles into images using PIL.ImageDraw
- Preparing multimodal payloads for LLM submission

Usage Example:
    >>> from chatbot.image_handler import scan_input_directory, process_image_annotations
    >>> image_paths = scan_input_directory("./input", max_count=2)
    >>> annotated_images = process_image_annotations(editor_outputs)
"""

from typing import Any

import numpy as np

# Import functions from the new modules for backward compatibility
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
)


def parse_image_editor_data(editor_value: dict | None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Parse the output from a Gradio ImageEditor component.

    The ImageEditor returns a dictionary with keys: 'background', 'layers', and 'composite'.
    This function extracts the base image and any drawn layers for further processing.

    Args:
        editor_value: The value returned by gr.ImageEditor, or None if empty.
                      Expected format: {"background": np.ndarray, "layers": [np.ndarray], ...}

    Returns:
        Tuple of (base_image, layers) where:
        - base_image: The background image as numpy array (or None)
        - layers: List of layer arrays representing user drawings (empty list if none)

    Example:
        >>> base, layers = parse_image_editor_data(editor_output)
        >>> assert len(layers) >= 0  # May be empty if no drawing
    """
    if editor_value is None:
        return None, []

    background = editor_value.get("background")
    layers = editor_value.get("layers", [])

    # Handle case where layers might be a single array instead of list
    if isinstance(layers, np.ndarray):
        layers = [layers]

    return background, list(layers)


def process_image_annotations(
    editor_value: dict | None, min_area: int = 100
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    """Process ImageEditor output to produce an annotated image.

    This is the main entry point for processing user annotations:
    1. Parse the editor value to extract base image and layers
    2. Detect rectangles from drawn layers
    3. Burn rectangles into the base image

    Args:
        editor_value: Output from gr.ImageEditor component.
        min_area: Minimum area threshold for rectangle detection.

    Returns:
        Tuple of (annotated_image, detected_rectangles) where:
        - annotated_image: Numpy array with rectangles burned in (or None)
        - detected_rectangles: List of rectangle dicts that were drawn

    Example:
        >>> annotated_img, rects = process_image_annotations(editor_output)
        >>> if annotated_img is not None:
        ...     print(f"Found {len(rects)} annotations")
    """
    base_image, layers = parse_image_editor_data(editor_value)

    if base_image is None:
        return None, []

    # Detect rectangles from all layers
    rectangles = detect_all_rectangles(layers, min_area)

    if not rectangles:
        # No annotations, return original image
        return base_image, []

    # Burn rectangles into the image
    annotated_image = burn_rectangles_into_image(base_image, rectangles)

    return annotated_image, rectangles


def prepare_multimodal_payload(
    text: str,
    editor_values: list[dict | None],
    min_area: int = 100,
) -> tuple[str, list[np.ndarray]]:
    """Prepare a multimodal payload for LLM submission.

    This function processes multiple ImageEditor outputs and combines them
    with text into a format suitable for multimodal LLM providers.

    Args:
        text: The user's text message.
        editor_values: List of ImageEditor output dictionaries (one per image).
        min_area: Minimum area threshold for rectangle detection.

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

        annotated_img, _ = process_image_annotations(editor_value, min_area)
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
    "parse_image_editor_data",
    "detect_rectangles_from_layer",
    "detect_all_rectangles",
    "burn_rectangles_into_image",
    "process_image_annotations",
    "prepare_multimodal_payload",
    "image_to_base64",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_INPUT_DIR",
    "MAX_IMAGES_DEFAULT",
]
