"""Configuration management for LLM providers.

This module provides functionality to load and save provider configurations
to JSON files in the config/ directory at the project root.

Usage Example:
    >>> from chatbot.config_manager import load_provider_config, save_provider_config
    >>> config = load_provider_config("lmstudio")
    >>> save_provider_config("lmstudio", {"endpoint_url": "http://localhost:1234/v1"})
"""

import json
import os
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).parent.parent / "config"


def _ensure_config_dir() -> None:
    """Ensure the config directory exists.

    Creates the config/ directory at the project root if it doesn't exist.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def get_config_file_path(provider_type: str) -> Path:
    """Get the path to the configuration file for a provider.

    Args:
        provider_type: The provider type identifier (e.g., "lmstudio", "llamacpp").

    Returns:
        Path object pointing to the provider's config file.
    """
    return CONFIG_DIR / f"{provider_type}_config.json"


def load_provider_config(
    provider_type: str, defaults: dict[str, Any]
) -> dict[str, Any]:
    """Load saved configuration for a provider, falling back to defaults.

    This function attempts to load the saved configuration from disk. If no
    saved config exists or it's invalid, returns the provided defaults merged
    with any valid fields from the saved file.

    Args:
        provider_type: The provider type identifier (e.g., "lmstudio").
        defaults: Dictionary of default values to use if no saved config exists.

    Returns:
        Merged configuration dictionary with saved values taking precedence
        over defaults for existing keys.

    Example:
        >>> config = load_provider_config("lmstudio", {"endpoint_url": "http://localhost:1234/v1"})
        >>> assert "endpoint_url" in config
    """
    _ensure_config_dir()

    config_path = get_config_file_path(provider_type)

    if not config_path.exists():
        return defaults.copy()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        if not isinstance(saved_config, dict):
            return defaults.copy()

        merged = defaults.copy()
        merged.update(saved_config)
        return merged

    except (json.JSONDecodeError, IOError):
        return defaults.copy()


def save_provider_config(provider_type: str, config: dict[str, Any]) -> bool:
    """Save configuration for a provider to disk.

    Args:
        provider_type: The provider type identifier (e.g., "lmstudio").
        config: Configuration dictionary to save.

    Returns:
        True if save was successful, False otherwise.

    Example:
        >>> success = save_provider_config("lmstudio", {"endpoint_url": "http://localhost:1234/v1"})
        >>> assert success is True
    """
    _ensure_config_dir()

    config_path = get_config_file_path(provider_type)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True

    except IOError:
        return False


def revert_field_to_default(
    provider_type: str,
    field_name: str,
    current_config: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Revert a specific field to its default value.

    This function returns a new config dictionary with the specified field
    reset to its built-in default value from get_config_defaults().

    Args:
        provider_type: The provider type identifier (e.g., "lmstudio").
        field_name: The name of the field to revert.
        current_config: Current configuration dictionary.
        defaults: Dictionary of default values.

    Returns:
        New configuration dictionary with the reverted field.

    Example:
        >>> config = {"endpoint_url": "http://custom-url", "model_name": "my-model"}
        >>> defaults = {"endpoint_url": "http://localhost:1234/v1", "model_name": ""}
        >>> new_config = revert_field_to_default("lmstudio", "endpoint_url", config, defaults)
        >>> assert new_config["endpoint_url"] == "http://localhost:1234/v1"
    """
    new_config = current_config.copy()

    if field_name in defaults:
        new_config[field_name] = defaults[field_name]

    return new_config


def has_saved_config(provider_type: str) -> bool:
    """Check if a saved configuration exists for a provider.

    Args:
        provider_type: The provider type identifier (e.g., "lmstudio").

    Returns:
        True if a config file exists, False otherwise.

    Example:
        >>> has_saved_config("lmstudio")  # Returns True/False based on file existence
    """
    config_path = get_config_file_path(provider_type)
    return config_path.exists()


__all__ = [
    "load_provider_config",
    "save_provider_config",
    "revert_field_to_default",
    "has_saved_config",
    "get_config_file_path",
]
