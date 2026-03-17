"""Model fetching functionality for the chatbot application.

This module provides functions for fetching available models from provider endpoints,
validating endpoint URLs, and managing model selection state. It separates the model
fetching logic from the UI layer for better testability and maintainability.
"""

from typing import Tuple

import gradio as gr

from chatbot.providers import get_provider


def validate_endpoint_url(endpoint_url: str) -> Tuple[bool, str]:
    """Validate an endpoint URL for format correctness.

    This function checks that the provided URL is non-empty and starts with
    either 'http://' or 'https://'.

    Args:
        endpoint_url: The URL string to validate.

    Returns:
        A tuple of (is_valid, error_message). If valid, error_message is empty.
        If invalid, is_valid is False and error_message describes the issue.

    Example:
        >>> is_valid, error = validate_endpoint_url("http://localhost:1234/v1")
        >>> assert is_valid and not error
        >>> is_valid, error = validate_endpoint_url("")
        >>> assert not is_valid
    """
    if not endpoint_url or not endpoint_url.strip():
        return False, "Please enter a valid endpoint URL"

    url = endpoint_url.strip()
    if not url.startswith(("http://", "https://")):
        return (
            False,
            "Invalid URL format. Must start with http:// or https://",
        )

    return True, ""


def fetch_models_from_endpoint(
    selected_provider: str,
    endpoint_url: str,
    current_models: list[str],
    progress: gr.Progress = gr.Progress(),
) -> Tuple[list[str], str, gr.update]:
    """Fetch models from the configured provider endpoint.

    This function queries the provider's API to retrieve available models
    and updates the dropdown component with the results. It handles various
    error conditions gracefully, preserving the current model list on failure.

    Args:
        selected_provider: The currently selected provider type (e.g., "lmstudio").
        endpoint_url: The endpoint URL to query for models.
        current_models: Current list of models (preserved on error).
        progress: Gradio progress indicator for loading states.

    Returns:
        A tuple of (updated_model_list, status_message, dropdown_update).
        - updated_model_list: List of available model names or current_models on error.
        - status_message: Human-readable status message with emoji prefix.
        - dropdown_update: gr.update object for updating the model dropdown component.

    Example:
        >>> models, status, update = fetch_models_from_endpoint(
        ...     "lmstudio", "http://localhost:1234/v1", []
        ... )
        >>> print(f"Found {len(models)} models")
    """
    # Set initial loading state
    progress(0.1, desc="Connecting to provider...")

    # Validate endpoint URL
    is_valid, error_msg = validate_endpoint_url(endpoint_url)
    if not is_valid:
        progress(1.0, desc="Error!")
        return current_models, f"⚠️ {error_msg}", gr.update(choices=current_models)

    try:
        # Create provider instance with endpoint configuration
        config = {"endpoint_url": endpoint_url.strip()}
        provider = get_provider(selected_provider, config)

        progress(0.5, desc="Fetching model list...")

        # Fetch models from provider
        available_models = provider.fetch_models()

        progress(1.0, desc="Complete!")

        if not available_models:
            status_msg = "ℹ️ No models found at this endpoint"
        else:
            status_msg = f"✅ Found {len(available_models)} model(s)"

        # Update dropdown with new models, selecting first one by default
        return (
            available_models,
            status_msg,
            gr.update(
                choices=available_models,
                value=available_models[0] if available_models else None,
            ),
        )

    except ConnectionError as e:
        progress(1.0, desc="Error!")
        return (
            current_models,  # Preserve existing models on error
            f"❌ Connection Error: {str(e)}",
            gr.update(choices=current_models),
        )

    except RuntimeError as e:
        progress(1.0, desc="Error!")
        return current_models, f"❌ Error: {str(e)}", gr.update(choices=current_models)

    except Exception as e:
        progress(1.0, desc="Error!")
        return (
            current_models,
            f"❌ Unexpected error: {type(e).__name__}",
            gr.update(choices=current_models),
        )


def refresh_models_manually(
    selected_provider: str,
    endpoint_url: str,
    current_models: list[str],
    current_selected_model: str | None,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[list[str], str, gr.update]:
    """Manually refresh model list while preserving selection.

    This function fetches models using the same logic as auto-fetch but
    preserves the currently selected model if it's still available in the
    updated list.

    Args:
        selected_provider: Currently selected provider type.
        endpoint_url: Current endpoint URL.
        current_models: Existing model list.
        current_selected_model: Currently selected model value.
        progress: Gradio progress indicator.

    Returns:
        A tuple of (updated_model_list, status_message, dropdown_update).
        The dropdown update preserves the current selection if possible.

    Example:
        >>> models, status, update = refresh_models_manually(
        ...     "lmstudio", "http://localhost:1234/v1", ["model-1"], "model-1"
        ... )
    """
    # Fetch models using the same logic as auto-fetch
    updated_models, status_msg, dropdown_update = fetch_models_from_endpoint(
        selected_provider, endpoint_url, current_models, progress
    )

    # Preserve current selection if it's still in the updated list
    if current_selected_model and current_selected_model in updated_models:
        dropdown_update["value"] = current_selected_model

    return updated_models, status_msg, dropdown_update


__all__ = [
    "validate_endpoint_url",
    "fetch_models_from_endpoint",
    "refresh_models_manually",
]
