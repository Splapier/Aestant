"""State initialization for the LLM chat application.

This module provides the container for all Gradio State objects used
throughout the application for session persistence.
"""

import gradio as gr
from dataclasses import dataclass


@dataclass
class AppState:
    """Container for all Gradio state components.

    Attributes:
        session_state: Stores provider configuration dict.
        chat_history: Stores conversation history as list of message dicts.
        models_state: Available models list from provider endpoint.
        endpoint_state: Current endpoint URL string.
        model_state: Currently selected model name.
        image_paths_state: List of loaded image file paths.
    """

    session_state: gr.State
    chat_history: gr.State
    models_state: gr.State
    endpoint_state: gr.State
    model_state: gr.State
    image_paths_state: gr.State


def create_app_state() -> AppState:
    """Create all Gradio state objects for the application.

    Must be called within a gr.Blocks context.

    Returns:
        AppState containing all state components.
    """
    return AppState(
        session_state=gr.State({}),
        chat_history=gr.State([]),
        models_state=gr.State([]),
        endpoint_state=gr.State(value="http://localhost:1234/v1"),
        model_state=gr.State(value=""),
        image_paths_state=gr.State(value=[]),
    )
