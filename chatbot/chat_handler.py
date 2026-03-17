"""Chat message handling functionality for the chatbot application.

This module provides functions for processing user messages, streaming responses
from LLM providers, and managing session state updates. It separates the chat
handling logic from the UI layer for better testability and maintainability.
"""

from typing import Generator, Tuple, Any
import numpy as np

import gradio as gr

from chatbot.providers import get_provider
from chatbot.image_handler import process_image_annotations, prepare_multimodal_payload


def update_session_on_provider_change(
    selected_provider: str, current_session: dict | None
) -> dict:
    """Update the session state with the newly selected provider.

    This function updates the session dictionary to reflect a change in
    the selected LLM provider type.

    Args:
        selected_provider: The newly selected provider type (e.g., "lmstudio").
        current_session: Current session state dictionary, or None if new session.

    Returns:
        Updated session state with new provider selection.

    Example:
        >>> session = update_session_on_provider_change("llamacpp", {})
        >>> assert session["provider_type"] == "llamacpp"
    """
    if current_session is None:
        current_session = {}
    current_session["provider_type"] = selected_provider
    return current_session


def process_user_message(
    prompt_text: str,
    current_history: list[dict[str, str]] | None,
    provider_type: str,
    endpoint_url: str,
    model_name: str,
    image_editors_value: list[Any] | None = None,
) -> Generator[Tuple[str, list[dict[str, str]]], None, None]:
    """Process a user message by adding it to history and generating a response.

    This function handles the complete chat flow:
    1. Clears the input textbox first for immediate feedback
    2. Adds the user's message to the conversation history
    3. Processes any image annotations from ImageEditor components
    4. Creates or updates the provider with current configuration
    5. Streams the model's response back to the interface (with images if provided)

    Args:
        prompt_text: The text input from the user.
        current_history: Current conversation history list, or None if empty.
        provider_type: Currently selected provider type (e.g., "lmstudio").
        endpoint_url: Configured endpoint URL for the provider.
        model_name: Optional model name for the provider.
        image_editors_value: Optional list of ImageEditor output dictionaries.
                           Each dict has format: {"background": np.ndarray, 
                                                "layers": [np.ndarray], ...}

    Yields:
        Tuple of (cleared_prompt_text, updated_chat_history) after each update.
        The prompt text is always empty (cleared), and history accumulates messages.

    Example:
        >>> generator = process_user_message("Hello", [], "lmstudio", 
        ...                                  "http://localhost:1234/v1", "")
        >>> for prompt, history in generator:
        ...     print(f"History length: {len(history)}")
    """
    # Clear the input textbox first for immediate feedback
    yield "", current_history if current_history else []

    # Skip processing if no message was entered
    if not prompt_text.strip():
        return

    # Initialize history if empty or None
    updated_history = current_history.copy() if current_history else []

    # Process image annotations if provided
    annotated_images: list[np.ndarray] = []
    
    if image_editors_value:
        # Prepare multimodal payload - this processes all editor outputs
        _, annotated_images = prepare_multimodal_payload(
            prompt_text, 
            image_editors_value, 
            min_area=100
        )

    # Add user message to history
    updated_history.append({"role": "user", "content": prompt_text})
    yield "", updated_history

    # Prepare provider configuration
    config: dict[str, str] = {
        "endpoint_url": endpoint_url.strip(),
        "model_name": model_name.strip() if model_name else "",
    }

    try:
        # Create the provider instance with current configuration
        provider = get_provider(provider_type, config)

        # Add assistant message placeholder to history
        updated_history.append({"role": "assistant", "content": ""})

        # Stream the response from the provider (with images if available)
        full_response = ""
        
        # Call stream_chat with optional images parameter
        for chunk in provider.stream_chat(updated_history, annotated_images):
            if chunk:
                full_response += chunk
                # Update the last message (assistant's) with accumulated content
                updated_history[-1]["content"] = full_response
                yield "", updated_history

    except ConnectionError as e:
        # Handle connection errors gracefully
        error_message = f"❌ Connection Error: {str(e)}"
        updated_history.append({"role": "assistant", "content": error_message})
        yield "", updated_history

    except ValueError as e:
        # Handle configuration/validation errors
        error_message = f"⚠️ Configuration Error: {str(e)}"
        updated_history.append({"role": "assistant", "content": error_message})
        yield "", updated_history

    except RuntimeError as e:
        # Handle runtime/inference errors
        error_message = f"🔧 Runtime Error: {str(e)}"
        updated_history.append({"role": "assistant", "content": error_message})
        yield "", updated_history

    except Exception as e:
        # Catch-all for unexpected errors
        error_message = f"💥 Unexpected Error: {type(e).__name__} - {str(e)}"
        updated_history.append({"role": "assistant", "content": error_message})
        yield "", updated_history


def clear_chat() -> Tuple[str, list]:
    """Clear all conversation history.

    This function returns empty values for both the prompt input textbox
    and the chat history, effectively clearing the entire conversation.

    Returns:
        A tuple of (empty_string_for_textbox, empty_list_for_history).

    Example:
        >>> prompt, history = clear_chat()
        >>> assert prompt == "" and history == []
    """
    return "", []


__all__ = [
    "update_session_on_provider_change",
    "process_user_message",
    "clear_chat",
]
