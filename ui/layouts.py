"""Layout composition functions for the chat application UI.

This module provides functions for composing Gradio Blocks layouts, including
the sidebar with provider configuration and the main chat area. It uses the
@gr.render decorator for dynamic content based on provider selection.
"""

import gradio as gr

from ui.components import (
    create_endpoint_input,
    create_model_dropdown,
    create_refresh_button,
    create_status_display,
)


def render_provider_config_panel(
    selected_provider: str,
    models_state: list[str],
    endpoint_state: str,
    model_state: str,
) -> tuple[gr.Textbox, gr.Dropdown, gr.Markdown, gr.Button]:
    """Render configuration fields specific to the selected provider.

    This function uses the @gr.render decorator to dynamically update
    the UI based on the provider selection. Each provider type shows
    its relevant configuration options including endpoint URL, model
    dropdown with dynamic fetching, and status display.

    Args:
        selected_provider: The currently selected provider type string.
        models_state: Current list of available models (from gr.State).
        endpoint_state: Current endpoint URL state value.
        model_state: Currently selected model state value.

    Returns:
        Tuple of (endpoint_input, model_dropdown, status_display, refresh_button)
        for use in event wiring outside this function.

    Note:
        This function is designed to be used with @gr.render(inputs=provider_selector)
        and should not be called directly outside that context.
    """
    # Get default config values based on provider type
    default_endpoint = ""
    if selected_provider == "lmstudio":
        default_endpoint = "http://localhost:1234/v1"
        gr.Markdown(
            "Configure your local LM Studio server connection.",
            key="lmstudio-desc",
        )
    elif selected_provider == "llamacpp":
        default_endpoint = "http://localhost:8080"
        gr.Markdown(
            "Configure your local llama.cpp server connection.",
            key="llamacpp-desc",
        )
    else:
        # Fallback for unknown providers
        gr.Markdown(f"Configure {selected_provider.title()} provider.")

    gr.Markdown(
        f"### {selected_provider.replace('cpp', '.cpp').title()} Configuration",
        key=f"{selected_provider}-config-title",
    )

    # Create endpoint input with provider-specific default
    endpoint_input = create_endpoint_input(default_endpoint)
    endpoint_input._metadata["key"] = f"{selected_provider}-endpoint"  # type: ignore[index]

    # Model selection dropdown (dynamically populated)
    model_dropdown = create_model_dropdown()
    model_dropdown._metadata["key"] = f"{selected_provider}-model-dropdown"  # type: ignore[index]

    # Status message display
    status_display = create_status_display()
    status_display._metadata["key"] = f"{selected_provider}-status"  # type: ignore[index]

    # Refresh button for manual update
    refresh_button = create_refresh_button()
    refresh_button._metadata["key"] = f"{selected_provider}-refresh"  # type: ignore[index]

    return endpoint_input, model_dropdown, status_display, refresh_button


def create_sidebar_layout(
    provider_selector: gr.Radio,
) -> tuple[gr.Textbox, gr.Dropdown, gr.Markdown, gr.Button]:
    """Create sidebar layout with provider configuration panel.

    This function creates the complete sidebar including the provider selector
    and a dynamically rendered configuration panel that changes based on
    the selected provider type.

    Args:
        provider_selector: The provider selection radio button component.

    Returns:
        Tuple of (endpoint_input, model_dropdown, status_display, refresh_button)
        for use in event wiring outside this function.

    Note:
        This function must be called within a gr.Blocks context manager.
    """
    with gr.Sidebar(open=True):
        gr.Markdown("## Provider Selection")

        # Dynamic configuration panel based on selected provider
        @gr.render(inputs=provider_selector)
        def render_config(selected_provider: str):
            """Render the config panel for the selected provider."""
            # These are placeholder state references - actual states come from app_builder
            models_state = gr.State([])
            endpoint_state = gr.State("")
            model_state = gr.State("")

            return render_provider_config_panel(
                selected_provider, models_state, endpoint_state, model_state
            )

    # Return the components rendered by the last @gr.render call
    # Note: In actual usage, these are accessed via the render context
    # This is a simplified representation for documentation purposes
    return (
        gr.Textbox(),  # type: ignore[return-value]
        gr.Dropdown(),  # type: ignore[return-value]
        gr.Markdown(),  # type: ignore[return-value]
        gr.Button(),  # type: ignore[return-value]
    )


def create_main_chat_area() -> tuple[gr.Chatbot, gr.Textbox, gr.Button, gr.Button]:
    """Create the main chat interface area.

    This function creates the primary chat components including the chatbot
    display, user input textbox, and action buttons (send/clear).

    Returns:
        Tuple of (chatbot_component, prompt_input, send_button, clear_button)
        for use in event wiring outside this function.

    Note:
        This function must be called within a gr.Blocks context manager.

    Example:
        >>> with gr.Blocks() as demo:
        ...     chatbot, prompt, send_btn, clear_btn = create_main_chat_area()
    """
    with gr.Column(scale=3):
        # Chatbot component for displaying conversation history
        chatbot_component = gr.Chatbot(label="Conversation", height=500)

        # Input textbox for user messages
        prompt_input = gr.Textbox(
            label="Your Message",
            lines=2,
            placeholder="Type your message here...",
            container=True,
        )

        # Action buttons row
        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear Chat", variant="secondary")

    return chatbot_component, prompt_input, send_button, clear_button


__all__ = [
    "render_provider_config_panel",
    "create_sidebar_layout",
    "create_main_chat_area",
]
