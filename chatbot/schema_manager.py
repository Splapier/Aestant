"""Schema management for tagging system.

This module provides functionality for:
- Loading and saving the master schema JSON
- Validating schema modifications against protected keys
- Checking whether changes are allowed

Protection is determined dynamically based on schema depth:
- Level 0: All top-level keys in the schema (protected)
- Level 1: Keys directly under level 0 with dict values (protected)
- Level 2+: User-modifiable keys

Usage Example:
    >>> from chatbot.schema_manager import load_master_schema, validate_schema_change
    >>> schema = load_master_schema()
    >>> allowed, msg = validate_schema_change(schema, schema, "delete", "any_key")
    >>> print(allowed)  # False if level 0 or level 1 key
"""

import json
from pathlib import Path
from typing import Any
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_SCHEMA_PATH = PROJECT_ROOT / "master_schema.json"


def get_default_schema() -> dict[str, Any]:
    """Return the default blank master schema structure.

    Returns:
        Dictionary containing schema from file, or empty dict if file doesn't exist.
    """
    if MASTER_SCHEMA_PATH.exists():
        with open(MASTER_SCHEMA_PATH, "r") as f:
            return json.load(f)
    return {}


def load_master_schema() -> dict[str, Any]:
    """Load the master schema from JSON file.

    If the file doesn't exist, creates it with the default schema.

    Returns:
        Master schema dictionary.

    Example:
        >>> schema = load_master_schema()
        >>> "body_regions" in schema
        True
    """
    if not MASTER_SCHEMA_PATH.exists():
        default_schema = get_default_schema()
        save_master_schema(default_schema)
        return default_schema

    with open(MASTER_SCHEMA_PATH, "r") as f:
        return json.load(f)


def save_master_schema(schema: dict[str, Any]) -> None:
    """Save the master schema to JSON file.

    Args:
        schema: The schema dictionary to save.

    Example:
        >>> schema = load_master_schema()
        >>> save_master_schema(schema)
    """
    with open(MASTER_SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=4)


def get_protected_keys_at_level(
    schema: dict[str, Any],
    level: int,
) -> set[str]:
    """Get the set of protected key names at a specific level.

    Args:
        schema: The schema dictionary to inspect.
        level: Level of keys to retrieve (0 = top level, 1 = second level, etc.).

    Returns:
        Set of protected key names at the specified level.

    Example:
        >>> schema = load_master_schema()
        >>> get_protected_keys_at_level(schema, 0)
        {'body_regions', 'upper_extremities', 'lower_extremities'}
    """
    if level == 0:
        return set(schema.keys())

    if level == 1:
        level_1_keys = set()
        for key in schema:
            if isinstance(schema[key], dict):
                level_1_keys.update(
                    k for k in schema[key].keys() if isinstance(schema[key][k], dict)
                )
        return level_1_keys

    return set()


def find_key_path(
    schema: dict[str, Any],
    key_name: str,
    current_path: str = "",
) -> list[str] | None:
    """Find the dot-separated path to a key in the schema.

    Args:
        schema: The schema dictionary to search.
        key_name: The key name to find.
        current_path: Current path traversal (for recursion).

    Returns:
        List of path components, or None if not found.

    Example:
        >>> schema = load_master_schema()
        >>> path = find_key_path(schema, "color")
        >>> print(path)  # ['body_regions', 'head', 'hair', 'color']
    """
    for key, value in schema.items():
        if key == key_name:
            if current_path:
                return current_path.split(".")
            return []

        if isinstance(value, dict):
            new_path = f"{current_path}.{key}" if current_path else key
            result = find_key_path(value, key_name, new_path)
            if result is not None:
                path_parts = [key] + result
                return path_parts

    return None


def parse_key_path(path: str) -> list[str]:
    """Parse a dot-separated key path into components.

    Args:
        path: Dot-separated path (e.g., "body_regions.head.hair.color").

    Returns:
        List of path components.

    Example:
        >>> parse_key_path("body_regions.head.hair.color")
        ['body_regions', 'head', 'hair', 'color']
    """
    if not path:
        return []
    return path.split(".")


def get_key_at_path(
    schema: dict[str, Any],
    path: str | list[str],
) -> dict[str, Any] | None:
    """Get the value at a specific path in the schema.

    Args:
        schema: The schema dictionary.
        path: Dot-separated path or list of path components.

    Returns:
        The value at the path, or None if not found.

    Example:
        >>> schema = load_master_schema()
        >>> get_key_at_path(schema, "body_regions.head.hair.color")
        {'type': 'string', 'default': None, ...}
    """
    if isinstance(path, str):
        path = parse_key_path(path)

    current = schema
    for component in path:
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]

    return current


def set_key_at_path(
    schema: dict[str, Any],
    path: str | list[str],
    value: dict[str, Any],
) -> dict[str, Any]:
    """Set a value at a specific path in the schema (mutates in place).

    Args:
        schema: The schema dictionary to modify.
        path: Dot-separated path or list of path components.
        value: The value to set.

    Returns:
        The modified schema (same object).

    Example:
        >>> schema = load_master_schema()
        >>> new_field = {"type": "string", "default": None, "confidence_score": 0.0, "description": "New field"}
        >>> set_key_at_path(schema, "body_regions.head.hair.length", new_field)
    """
    if isinstance(path, str):
        path = parse_key_path(path)

    current = schema
    for component in path[:-1]:
        if component not in current:
            current[component] = {}
        current = current[component]

    current[path[-1]] = value
    return schema


def delete_key_at_path(
    schema: dict[str, Any],
    path: str | list[str],
) -> bool:
    """Delete a key at a specific path in the schema.

    Args:
        schema: The schema dictionary to modify.
        path: Dot-separated path or list of path components.

    Returns:
        True if deleted, False if not found.

    Example:
        >>> schema = load_master_schema()
        >>> delete_key_at_path(schema, "body_regions.head.hair.length")
        True
    """
    if isinstance(path, str):
        path = parse_key_path(path)

    current = schema
    for component in path[:-1]:
        if not isinstance(current, dict) or component not in current:
            return False
        current = current[component]

    if path[-1] in current:
        del current[path[-1]]
        return True
    return False


def validate_schema_change(
    old_schema: dict[str, Any],
    new_schema: dict[str, Any],
    operation: str,
    path: str,
) -> tuple[bool, str]:
    """Validate if a schema modification is allowed.

    Args:
        old_schema: The original schema.
        new_schema: The modified schema.
        operation: One of "add", "delete", "modify", "rename".
        path: Dot-separated path to the key being changed.

    Returns:
        Tuple of (allowed: bool, message: str).
        If not allowed, message explains why.

    Example:
        >>> old = load_master_schema()
        >>> new = deepcopy(old)
        >>> validate_schema_change(old, new, "delete", "body_regions")
        (False, "Cannot delete level 0 key: body_regions")
    """
    path_parts = parse_key_path(path)
    if not path_parts:
        return False, "Invalid path"

    num_parts = len(path_parts)

    if operation == "delete":
        if num_parts == 1:
            return False, f"Cannot delete level 0 key: {path_parts[0]}"
        elif num_parts == 2:
            return False, f"Cannot delete level 1 key: {path_parts[1]}"

    if operation in ("add", "rename"):
        if num_parts == 1:
            return False, "Cannot add/rename any level 0 key"
        if num_parts == 2:
            return False, f"Cannot add/rename level 1 key: {path_parts[1]}"

    return True, "Allowed"


def flatten_schema_keys(
    schema: dict[str, Any],
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    """Flatten the schema into dot-separated paths with their field definitions.

    Args:
        schema: The master schema dictionary.
        prefix: Current path prefix (for recursion).

    Returns:
        Dictionary mapping dot-separated paths to field definitions.

    Example:
        >>> schema = load_master_schema()
        >>> fields = flatten_schema_keys(schema)
        >>> "body_regions.head.hair.color" in fields
        True
    """
    result = {}

    for key, value in schema.items():
        new_prefix = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            if "type" in value and "description" in value:
                result[new_prefix] = value
            else:
                result.update(flatten_schema_keys(value, new_prefix))

    return result


def get_field_descriptions(schema: dict[str, Any]) -> str:
    """Generate a prompt-friendly description of all fields in the schema.

    Args:
        schema: The master schema dictionary.

    Returns:
        String with field paths and their descriptions.

    Example:
        >>> schema = load_master_schema()
        >>> desc = get_field_descriptions(schema)
        >>> print(desc[:200])
    """
    fields = flatten_schema_keys(schema)
    lines = []

    for path, field_def in sorted(fields.items()):
        desc = field_def.get("description", "No description")
        f_type = field_def.get("type", "string")
        lines.append(f"- {path} ({f_type}): {desc}")

    return "\n".join(lines)


def init_schema_schema() -> dict[str, Any]:
    """Initialize schema if not present.

    Returns:
        The loaded or created master schema.
    """
    return load_master_schema()


__all__ = [
    "load_master_schema",
    "save_master_schema",
    "get_protected_keys_at_level",
    "find_key_path",
    "parse_key_path",
    "get_key_at_path",
    "set_key_at_path",
    "delete_key_at_path",
    "validate_schema_change",
    "flatten_schema_keys",
    "get_field_descriptions",
    "init_schema_schema",
]
