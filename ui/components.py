"""Reusable Gradio component builders for the chat application.

This module provides factory functions for creating individual Gradio components
used throughout the chat interface. Each function returns a configured component
ready to be used in the UI layout.
"""

import gradio as gr
from typing import Any

from chatbot import get_available_providers


def create_provider_selector() -> gr.Radio:
    """Create provider selection radio button.

    Returns:
        gr.Radio: Configured radio button for selecting LLM provider type.

    Example:
        >>> selector = create_provider_selector()
        >>> assert "lmstudio" in selector.choices
    """
    return gr.Radio(
        choices=get_available_providers(),
        label="Select LLM Provider",
        value="lmstudio",  # Default selection
        interactive=True,
    )


def create_endpoint_input(default_url: str = "") -> gr.Textbox:
    """Create endpoint URL input textbox.

    Args:
        default_url: Default URL to pre-fill in the textbox. If empty, no default is set.

    Returns:
        gr.Textbox: Configured textbox for entering provider endpoint URL.

    Example:
        >>> input_box = create_endpoint_input("http://localhost:1234/v1")
    """
    return gr.Textbox(
        label="Endpoint URL",
        value=default_url if default_url else None,
        placeholder="e.g., http://localhost:1234/v1",
        info="Model list refreshes automatically on change",
    )


def create_model_dropdown() -> gr.Dropdown:
    """Create model selection dropdown.

    Returns:
        gr.Dropdown: Configured dropdown for selecting LLM models.
                     Initially empty, populated via fetch_models_from_endpoint.

    Example:
        >>> dropdown = create_model_dropdown()
        >>> assert dropdown.allow_custom_value is True
    """
    return gr.Dropdown(
        choices=[],  # Populated dynamically via fetch_models_from_endpoint
        label="Select Model",
        interactive=True,
        allow_custom_value=True,  # Allow manual entry if needed
        info="Leave empty to use default loaded model",
    )


def create_status_display() -> gr.Markdown:
    """Create status message display component.

    Returns:
        gr.Markdown: Configured markdown component for displaying status messages.

    Example:
        >>> status = create_status_display()
        >>> assert "Status" in str(label) if (label := status.label) else False
    """
    return gr.Markdown(
        label="Status",
        value="ℹ️ Configure endpoint and press Refresh or modify URL",
    )


def create_refresh_button() -> gr.Button:
    """Create manual refresh button for model list.

    Returns:
        gr.Button: Configured secondary variant button for refreshing models.

    Example:
        >>> button = create_refresh_button()
        >>> assert button.variant == "secondary"
    """
    return gr.Button("🔄 Refresh Models", variant="secondary")


def create_chatbot_component() -> gr.Chatbot:
    """Create chatbot display component.

    Returns:
        gr.Chatbot: Configured chatbot component for displaying conversation history.

    Example:
        >>> chatbot = create_chatbot_component()
        >>> assert chatbot.height == 500
    """
    return gr.Chatbot(label="Conversation", height=500)


def create_prompt_input() -> gr.Textbox:
    """Create user message input textbox.

    Returns:
        gr.Textbox: Configured multiline textbox for entering user messages.

    Example:
        >>> prompt = create_prompt_input()
        >>> assert prompt.lines == 2
    """
    return gr.Textbox(
        label="Your Message",
        lines=2,
        placeholder="Type your message here...",
        container=True,
    )


def create_send_button() -> gr.Button:
    """Create send button for submitting messages.

    Returns:
        gr.Button: Configured primary variant button for sending messages.

    Example:
        >>> button = create_send_button()
        >>> assert button.variant == "primary"
    """
    return gr.Button("Send", variant="primary")


def create_clear_button() -> gr.Button:
    """Create clear chat button.

    Returns:
        gr.Button: Configured secondary variant button for clearing conversation history.

    Example:
        >>> button = create_clear_button()
        >>> assert button.variant == "secondary"
    """
    return gr.Button("Clear Chat", variant="secondary")


def create_action_buttons() -> tuple[gr.Button, gr.Button]:
    """Create action buttons row with send and clear buttons.

    Returns:
        Tuple of (send_button, clear_button): Both configured button components.

    Example:
        >>> send_btn, clear_btn = create_action_buttons()
        >>> assert send_btn.value == "Send"
        >>> assert clear_btn.value == "Clear Chat"
    """
    return create_send_button(), create_clear_button()


def create_image_editor_row(
    image_arrays: list[Any] | None = None, 
    max_images: int = 2
) -> tuple[gr.ImageEditor, ...]:
    """Create a row of ImageEditor components for user annotation.

    This function creates one or more Gradio ImageEditor components configured
    for drawing rectangles on images using the brush tool. Each editor is set up
    with type="numpy" for direct array manipulation and includes brush tools
    with predefined colors (red, green, blue).

    Args:
        image_arrays: Optional list of numpy arrays to use as initial backgrounds.
                     If None or empty, editors will be created without images.
        max_images: Maximum number of editors to display (default: 2).
                   
    Returns:
        Tuple of gr.ImageEditor components configured for annotation.

    Example:
        >>> editors = create_image_editor_row([img1, img2])
        >>> assert len(editors) == 2
    """
    # Determine how many editors to create
    num_editors = min(
        max_images, 
        len(image_arrays) if image_arrays else 0,
        max_images  # Always respect max limit
    )
    
    # If no images provided but we want to show placeholders, use max_images
    if not image_arrays or len(image_arrays) == 0:
        num_editors = 0
    
    editors = []
    
    for i in range(num_editors):
        editor = gr.ImageEditor(
            label=f"Image {i + 1} (Draw rectangles)",
            type="numpy",
            tools=["crop", "brush", "eraser"],
            brush=gr.Brush(
                default_color="#FF0000", 
                colors=["#FF0000", "#00FF00", "#0000FF"]
            ),
            interactive=True,
            value={
                "background": image_arrays[i] if i < len(image_arrays) else None,
                "layers": [],
                "composite": image_arrays[i] if i < len(image_arrays) else None,
            } if image_arrays and i < len(image_arrays) else None,
        )
        editors.append(editor)
    
    return tuple(editors) if editors else ()


def create_image_refresh_button() -> gr.Button:
    """Create refresh button for reloading images from input directory.

    Returns:
        gr.Button: Configured secondary variant button for refreshing images.

    Example:
        >>> button = create_image_refresh_button()
        >>> assert "Refresh" in button.value
    """
    return gr.Button("🖼️ Refresh Images", variant="secondary")


__all__ = [
    "create_provider_selector",
    "create_endpoint_input",
    "create_model_dropdown",
    "create_status_display",
    "create_refresh_button",
    "create_chatbot_component",
    "create_prompt_input",
    "create_send_button",
    "create_clear_button",
    "create_action_buttons",
    "create_image_editor_row",
    "create_image_refresh_button",
]
