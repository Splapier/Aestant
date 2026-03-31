"""Main application builder for the LLM chat interface.

This module provides the factory function for creating the complete Gradio Blocks
application, composing all UI components and wiring up event handlers.
"""

import gradio as gr

from ui.state_setup import create_app_state
from ui.sidebar import create_sidebar
from ui.chat_area import create_chat_area
from ui.event_handlers import wire_events


def create_chat_app() -> gr.Blocks:
    """Create and configure the complete LLM chat application.

    This function builds a Gradio Blocks application featuring:
    - A persistent sidebar for selecting LLM providers
    - Dynamic configuration fields that change based on provider type
    - Automatic model fetching from configured endpoints
    - Manual refresh button for updating model lists
    - A chatbot component supporting streaming responses
    - State management for session persistence
    - Image handling with interactive annotation via ImageEditor components

    Returns:
        gr.Blocks: The configured Gradio application interface.

    Example:
        >>> demo = create_chat_app()
        >>> demo.launch(server_name="127.0.0.1", server_port=7860)
    """
    with gr.Blocks(title="LLM Chat Interface") as demo:
        state = create_app_state()

        gr.Markdown("# 🤖 Modular LLM Chat Application")

        with gr.Row():
            # Chat area created first so prompt_input exists for sidebar wiring
            chat = create_chat_area(
                image_paths_state=state.image_paths_state,
            )

            provider_selector = create_sidebar(
                models_state=state.models_state,
                endpoint_state=state.endpoint_state,
                model_state=state.model_state,
                prompt_input=chat.prompt_input,
            )

        wire_events(
            provider_selector=provider_selector,
            state=state,
            chat=chat,
        )

    return demo


__all__ = ["create_chat_app"]
