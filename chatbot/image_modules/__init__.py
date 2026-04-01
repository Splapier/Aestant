"""Image handling modules for the chatbot application.

This package provides:
- image_loading: Functions for scanning and loading images from directories
- rectangle_processing: Functions for detecting and drawing rectangles on images
- rectangle_tool: Custom HTML5 Canvas rectangle drawing tool for Gradio

Usage Example:
    >>> from chatbot.image_modules import image_loading, rectangle_processing
    >>> paths = image_loading.scan_input_directory("./input")
"""

from .image_loading import (
    scan_input_directory,
    load_image_as_numpy,
    load_images_from_directory,
    SUPPORTED_EXTENSIONS,
    DEFAULT_INPUT_DIR,
    MAX_IMAGES_DEFAULT,
)

from .rectangle_processing import (
    detect_rectangles_from_layer,
    detect_all_rectangles,
    burn_rectangles_into_image,
    burn_rectangles_direct,
    hex_to_rgb,
)

from .rectangle_tool import (
    RectangleTool,
    RectangleToolConfig,
)


__all__ = [
    # From image_loading
    "scan_input_directory",
    "load_image_as_numpy",
    "load_images_from_directory",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_INPUT_DIR",
    "MAX_IMAGES_DEFAULT",
    # From rectangle_processing
    "detect_rectangles_from_layer",
    "detect_all_rectangles",
    "burn_rectangles_into_image",
    "burn_rectangles_direct",
    "hex_to_rgb",
    # From rectangle_tool
    "RectangleTool",
    "RectangleToolConfig",
]
