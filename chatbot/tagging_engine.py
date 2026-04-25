"""Tagging engine for single-image tagging via VLM.

This module handles:
- Concurrent VLM requests per level 0 schema key
- Bounding box cropping
- YAML structured output parsing
- Dataset file generation in YAML format for fine-tuning

Usage:
    >>> from chatbot.tagging_engine import tag_image_concurrent, save_tagged_dataset
    >>> result = tag_image_concurrent(image, rects, "lmstudio", config)
    >>> save_tagged_dataset(image_path, result["tags"], result["raw_responses"], result["prompts"])
"""

import concurrent.futures
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from chatbot.schema_manager import load_master_schema, get_descriptions_from_yaml
from chatbot.providers import get_provider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
INPUT_DIR = PROJECT_ROOT / "input"

try:
    from ruamel.yaml import YAML as _YAML

    def _create_yaml() -> Any:
        yaml = _YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        return yaml

    _ruamel_yaml = _create_yaml()
    RUAMEL_AVAILABLE = True
except ImportError:
    RUAMEL_AVAILABLE = False
    _ruamel_yaml = None


def _load_config() -> dict[str, Any]:
    """Load prompts configuration."""
    prompts_path = PROJECT_ROOT / "prompts_config.json"
    with open(prompts_path, "r") as f:
        return json.load(f)


def ensure_dataset_dir() -> None:
    """Create dataset directory if it doesn't exist."""
    DATASET_DIR.mkdir(exist_ok=True)


def get_level_0_keys_with_subtrees(schema: dict) -> list[tuple[str, dict]]:
    """Extract level 0 keys and their subtrees.

    Agnostic - works with any schema structure (head, upper_body, custom, etc.)

    Args:
        schema: The master schema dictionary.

    Returns:
        List of (key_name, subtree_dict) tuples.
    """
    return [(k, v) for k, v in schema.items() if isinstance(v, dict)]


def _flatten_subtree_to_paths(
    subtree: dict,
    prefix: str = "",
    descriptions: dict | None = None,
) -> list[tuple[str, dict, str]]:
    """Flatten subtree to leaf paths with field definitions and descriptions.

    Args:
        subtree: The subtree dictionary.
        prefix: Current path prefix.
        descriptions: Dict mapping dot-paths to YAML comments.

    Returns:
        List of (dot_path, field_def, description) tuples.
    """
    descriptions = descriptions or {}
    result = []

    for key, value in subtree.items():
        new_prefix = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            if "type" in value:
                desc = descriptions.get(new_prefix, value.get("description", ""))
                result.append((new_prefix, value, desc))
            else:
                result.extend(
                    _flatten_subtree_to_paths(value, new_prefix, descriptions)
                )

    return result


def build_field_definitions(subtree: dict, prefix: str, descriptions: dict) -> str:
    """Build field definitions string with comments as help text.

    Args:
        subtree: The subtree for this category.
        prefix: Current category prefix (e.g., "head").
        descriptions: Dict mapping paths to YAML comments.

    Returns:
        Formatted string of field definitions.
    """
    paths = _flatten_subtree_to_paths(subtree, prefix, descriptions)
    lines = []

    for dot_path, field_def, desc in paths:
        f_type = field_def.get("type", "string")
        if desc:
            lines.append(f"{dot_path}: {desc} (type: {f_type})")
        else:
            lines.append(f"{dot_path}: (type: {f_type})")

    return "\n".join(lines)


def _build_tagging_prompt(category_name: str, subtree: dict, descriptions: dict) -> str:
    """Build the prompt for tagging a specific category.

    Args:
        category_name: Name of the category (e.g., "head", "upper_body").
        subtree: The subtree dictionary for this category.
        descriptions: Dict mapping paths to YAML comments.

    Returns:
        Formatted prompt string.
    """
    config = _load_config()
    template = config.get("tagging_region_prompt", "")

    field_defs = build_field_definitions(subtree, category_name, descriptions)

    return template.format(
        category_name=category_name,
        field_definitions=field_defs,
    )


def parse_yaml_response(response: str) -> dict:
    """Parse YAML response from VLM to Python dict.

    Args:
        response: Raw YAML string from VLM.

    Returns:
        Parsed dictionary.
    """
    if not response or not response.strip():
        return {}

    response = response.strip()

    if response.startswith("```yaml"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    if RUAMEL_AVAILABLE:
        try:
            return _ruamel_yaml.load(response) or {}
        except Exception:
            pass

    try:
        import yaml

        return yaml.safe_load(response) or {}
    except ImportError:
        pass

    return {}


def dict_to_yaml_string(data: dict) -> str:
    """Convert dict to YAML string.

    Args:
        data: Dictionary to convert.

    Returns:
        YAML string.
    """
    if RUAMEL_AVAILABLE:
        from io import StringIO

        stream = StringIO()
        _ruamel_yaml.dump(data, stream)
        return stream.getvalue()

    try:
        import yaml

        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    except ImportError:
        return json.dumps(data, indent=2)


def crop_to_bounding_box(image: np.ndarray, rectangles: list[dict]) -> np.ndarray:
    """Crop image to single bounding box region if exactly one rectangle is drawn.

    Args:
        image: Original image as RGB numpy array.
        rectangles: List of rectangle dicts with x1, y1, x2, y2.

    Returns:
        Cropped image, or original if no rectangles or multiple.
    """
    if not rectangles or len(rectangles) != 1:
        return image

    rect = rectangles[0]
    x1 = min(rect.get("x1", 0), rect.get("x2", 0))
    y1 = min(rect.get("y1", 0), rect.get("y2", 0))
    x2 = max(rect.get("x1", 0), rect.get("x2", 0))
    y2 = max(rect.get("y1", 0), rect.get("y2", 0))

    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return image

    return image[y1:y2, x1:x2]


def _fetch_category_tags(
    category_name: str,
    subtree: dict,
    cropped_image: np.ndarray,
    provider_type: str,
    config: dict,
    descriptions: dict,
) -> tuple[str, dict, str, str]:
    """Fetch tags for a single category from VLM.

    Args:
        category_name: Name of the category.
        subtree: Subtree dictionary for this category.
        cropped_image: Image to analyze (possibly cropped).
        provider_type: Provider type string.
        config: Provider configuration.
        descriptions: Dict mapping paths to descriptions.

    Returns:
        Tuple of (category_name, tags_dict, raw_yaml_response, prompt_sent).
    """
    prompt = _build_tagging_prompt(category_name, subtree, descriptions)

    provider = get_provider(provider_type, config)

    messages = [
        {
            "role": "system",
            "content": "You are an expert at describing visual features in images. Respond ONLY with valid YAML.",
        },
        {"role": "user", "content": prompt},
    ]

    full_response = ""
    for chunk in provider.stream_chat(messages, [cropped_image]):
        if chunk:
            full_response += chunk

    tags = parse_yaml_response(full_response)

    return category_name, tags, full_response, prompt


def tag_image_concurrent(
    image: np.ndarray,
    rectangles: list[dict] | None,
    provider_type: str,
    config: dict,
) -> dict[str, Any]:
    """Tag image with concurrent VLM requests (one per level 0 key).

    This is the main entry point for tagging an image. It:
    1. Loads master schema (read-only)
    2. Crops image if single bounding box is drawn
    3. Sends concurrent VLM requests for each level 0 category
    4. Returns combined tags, raw responses, and prompts

    Args:
        image: Image as RGB numpy array.
        rectangles: Optional list of rectangle dicts for cropping.
        provider_type: Provider type (e.g., "lmstudio", "llamacpp").
        config: Provider configuration dict.

    Returns:
        Dict with keys: tags, raw_responses, prompts

    Example:
        >>> result = tag_image_concurrent(img, rects, "lmstudio", {})
        >>> result["tags"]["head"]["hair"]["style"]
        ['bob_cut', 'bangs']
    """
    ensure_dataset_dir()

    schema = load_master_schema()
    descriptions = get_descriptions_from_yaml()

    if rectangles:
        cropped_image = crop_to_bounding_box(image, rectangles)
    else:
        cropped_image = image

    categories = get_level_0_keys_with_subtrees(schema)

    results: dict[str, dict] = {}
    raw_responses: dict[str, str] = {}
    prompts: dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                _fetch_category_tags,
                cat_name,
                subtree,
                cropped_image,
                provider_type,
                config,
                descriptions,
            ): cat_name
            for cat_name, subtree in categories
        }

        for future in concurrent.futures.as_completed(futures):
            cat_name, tags, raw_response, prompt = future.result()
            results[cat_name] = tags
            raw_responses[cat_name] = raw_response
            prompts[cat_name] = prompt

    return {
        "tags": results,
        "raw_responses": raw_responses,
        "prompts": prompts,
    }


def save_tagged_dataset(
    image_path: str,
    tags: dict,
    raw_responses: dict,
    prompts: dict,
) -> str:
    """Save tagged image data to dataset directory as YAML.

    Args:
        image_path: Path to the original image.
        tags: Parsed tags dictionary.
        raw_responses: Raw YAML strings from VLM responses.
        prompts: Prompts sent to VLM for each category.

    Returns:
        Path to the saved YAML file.
    """
    ensure_dataset_dir()

    original_name = Path(image_path).name
    yaml_path = DATASET_DIR / f"{original_name}.yaml"

    data = {
        "image_path": image_path,
        "tagged_at": datetime.now().isoformat(),
        "tags": tags,
        "raw_responses": raw_responses,
        "prompts": prompts,
    }

    yaml_content = dict_to_yaml_string(data)

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return str(yaml_path)


def scan_untagged_images() -> list[str]:
    """Scan input directory for images not yet tagged.

    Returns:
        List of image paths that don't have corresponding dataset files.
    """
    ensure_dataset_dir()

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    input_images = [
        f.absolute()
        for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    tagged_names = {f.stem for f in DATASET_DIR.iterdir() if f.suffix == ".yaml"}

    untagged = [str(f.absolute()) for f in input_images if f.stem not in tagged_names]

    return sorted(untagged)


__all__ = [
    "tag_image_concurrent",
    "save_tagged_dataset",
    "scan_untagged_images",
    "crop_to_bounding_box",
    "parse_yaml_response",
    "dict_to_yaml_string",
    "get_level_0_keys_with_subtrees",
]
