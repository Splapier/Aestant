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


@dataclass
class ChatHistoryState:
    """Manages conversation history for the chat session.

    This class wraps the conversation history as a list of message dictionaries,
    providing helper methods for common operations like adding messages or clearing.

    Attributes:
        messages: List of message dictionaries with 'role' and 'content' keys.

    Example:
        >>> history = ChatHistoryState()
        >>> history.add_user_message("Hello!")
        >>> history.add_assistant_message("Hi there!")
    """

    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The text content of the user's message.
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history.

        Args:
            content: The text content of the assistant's response.
        """
        self.messages.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages.clear()

    def get_last_assistant_message_index(self) -> Optional[int]:
        """Get the index of the last assistant message.

        Returns:
            The index of the last assistant message, or None if no assistant
            messages exist in the history.
        """
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                return i
        return None

    def update_last_assistant_message(self, content: str) -> bool:
        """Update the content of the last assistant message.

        Args:
            content: The new content for the last assistant message.

        Returns:
            True if an assistant message was found and updated, False otherwise.
        """
        index = self.get_last_assistant_message_index()
        if index is not None:
            self.messages[index]["content"] = content
            return True
        return False


__all__ = [
    "ChatSessionState",
    "ModelState",
    "ChatHistoryState",
]
