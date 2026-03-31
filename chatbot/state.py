"""State management classes for the chatbot application.

This module provides dataclass-based state containers for managing:
- Session-level configuration (provider type, endpoint URL, model name)
- Model selection and available models tracking

These state classes are used throughout the application to maintain
consistent state across different components and handlers.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatSessionState:
    """Manages session-level configuration for the chatbot.

    This class holds the current provider selection and its associated
    configuration parameters. It provides sensible defaults that can be
    overridden by user input.

    Attributes:
        provider_type: The selected LLM provider (e.g., "lmstudio", "llamacpp").
        endpoint_url: The URL of the provider's API endpoint.
        model_name: The name/identifier of the model to use for chat completions.

    Example:
        >>> session = ChatSessionState()
        >>> session.provider_type = "llamacpp"
        >>> session.endpoint_url = "http://localhost:8080"
    """

    provider_type: str = "lmstudio"
    endpoint_url: str = "http://localhost:1234/v1"
    model_name: str = ""


@dataclass
class ModelState:
    """Tracks available models and the current selection.

    This class manages the list of models fetched from a provider endpoint
    and tracks which model is currently selected by the user.

    Attributes:
        available_models: List of model identifiers available from the provider.
        selected_model: The currently selected model, or None if not set.

    Example:
        >>> state = ModelState()
        >>> state.available_models = ["model-1", "model-2"]
        >>> state.selected_model = "model-1"
    """

    available_models: list[str] = field(default_factory=list)
    selected_model: Optional[str] = None


__all__ = [
    "ChatSessionState",
    "ModelState",
]
