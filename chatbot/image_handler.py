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

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# Default input directory path relative to project root
DEFAULT_INPUT_DIR = "input"

# Maximum number of images to load by default
MAX_IMAGES_DEFAULT = 2


def scan_input_directory(
    base_path: str | Path = DEFAULT_INPUT_DIR, max_count: int = MAX_IMAGES_DEFAULT
) -> list[str]:
    """Scan the input directory for valid image files.

    This function searches the specified directory (non-recursively) for files
    with supported image extensions and returns their absolute paths.

    Args:
        base_path: Path to the input directory. Defaults to "./input".
        max_count: Maximum number of images to return. Defaults to 2.

    Returns:
        List of absolute file paths to valid images, sorted by filename.
        Empty list if no images found or directory doesn't exist.

    Example:
        >>> paths = scan_input_directory("./input")
        >>> assert all(os.path.exists(p) for p in paths)
    """
    input_dir = Path(base_path).resolve()

    if not input_dir.exists():
        return []

    if not input_dir.is_dir():
        return []

    # Get all files with supported extensions
    image_files = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    # Sort by filename for consistent ordering
    image_files.sort(key=lambda x: x.name.lower())

    # Return absolute paths as strings, limited to max_count
    return [str(f.absolute()) for f in image_files[:max_count]]


def load_image_as_numpy(image_path: str | Path) -> np.ndarray | None:
    """Load an image file and convert it to a numpy array.

    Args:
        image_path: Path to the image file.

    Returns:
        Numpy array in RGB format (H x W x 3), or None if loading fails.

    Example:
        >>> img_array = load_image_as_numpy("./test.png")
        >>> assert img_array.shape[2] == 3  # RGB channels
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB (handles various input formats)
            rgb_img = img.convert("RGB")
            return np.array(rgb_img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_images_from_directory(
    base_path: str | Path = DEFAULT_INPUT_DIR, max_count: int = MAX_IMAGES_DEFAULT
) -> list[np.ndarray]:
    """Load images from the input directory as numpy arrays.

    Args:
        base_path: Path to the input directory. Defaults to "./input".
        max_count: Maximum number of images to load. Defaults to 2.

    Returns:
        List of numpy arrays (RGB format), excluding any failed loads.

    Example:
        >>> images = load_images_from_directory("./input")
        >>> assert all(isinstance(img, np.ndarray) for img in images)
    """
    image_paths = scan_input_directory(base_path, max_count)
    loaded_images = []

    for path in image_paths:
        img_array = load_image_as_numpy(path)
        if img_array is not None:
            loaded_images.append(img_array)

    return loaded_images


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
        color = rect.get("color", (255, 0, 0))  # Default to red

        # Draw rectangle outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    return np.array(pil_img)


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
