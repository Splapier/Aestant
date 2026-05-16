"""Image loading functionality for the chatbot application.

This module provides functions for:
- Scanning and loading images from a designated input directory
- Converting image files to numpy arrays
- Loading images with optional rectangle-based cropping for attention masking

Usage Example:
    >>> from chatbot.image_modules.image_loading import scan_input_directory, load_image_as_numpy, load_image_cropped
    >>> image_paths = scan_input_directory("./input", max_count=2)
    >>> img_array = load_image_as_numpy(image_paths[0])
    >>> cropped = load_image_cropped(image_paths[0], [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}])
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image


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


def load_image_cropped(
    image_path: str | Path,
    rectangles: list[dict] | None,
) -> np.ndarray | None:
    """Load an image and optionally crop to a bounding box region.

    This function loads an image from disk and, if rectangles are provided
    and valid, crops the image to the first rectangle's bounding box.
    This is used for attention-based comparison where only a specific
    region of the image should be considered.

    Args:
        image_path: Path to the image file.
        rectangles: Optional list of rectangle dicts with x1, y1, x2, y2 keys.
                   If provided and contains exactly one rectangle, the image
                   is cropped to that region. Otherwise, returns full image.

    Returns:
        Numpy array (RGB format), either full image or cropped region.
        Returns None if loading or cropping fails.

    Example:
        >>> rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        >>> cropped = load_image_cropped("./image.png", rects)
        >>> print(cropped.shape)  # (40, 40, 3)
    """
    img = load_image_as_numpy(image_path)
    if img is None:
        return None

    if not rectangles:
        return img

    if len(rectangles) != 1:
        return img

    rect = rectangles[0]
    x1 = min(rect.get("x1", 0), rect.get("x2", 0))
    y1 = min(rect.get("y1", 0), rect.get("y2", 0))
    x2 = max(rect.get("x1", 0), rect.get("x2", 0))
    y2 = max(rect.get("y1", 0), rect.get("y2", 0))

    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return img

    return img[y1:y2, x1:x2]


__all__ = [
    "scan_input_directory",
    "load_image_as_numpy",
    "load_images_from_directory",
    "load_image_cropped",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_INPUT_DIR",
    "MAX_IMAGES_DEFAULT",
]
