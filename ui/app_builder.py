"""Main application builder for the LLM chat interface.

This module provides the factory function for creating the complete Gradio Blocks
application, composing all UI components and wiring up event handlers.
"""

import gradio as gr

from ui.state_setup import create_app_state
from ui.sidebar import create_sidebar
from ui.chat_area import create_chat_area
from ui.event_handlers import wire_events
from ui.comparison_tab import create_comparison_tab
from ui.schema_viewer import create_schema_viewer_tab
from ui.tagging_tab import create_tagging_tab


def create_chat_app() -> gr.Blocks:
    """Create and configure the complete LLM chat application.

    Layout:
    ┌─────────────┬──────────────────────┬──────────────┐
    │  Provider   │     Main (Tabs)      │  Schema /    │
    │  Selection  │                      │  Tagging     │
    │             │  [Annotation] [Comp] │              │
    └─────────────┴──────────────────────┴──────────────┘

    Returns:
        gr.Blocks: The configured Gradio application interface.
    """
    with gr.Blocks(title="LLM Chat Interface") as demo:
        state = create_app_state()

        gr.Markdown("# 🤖 Modular LLM Chat Application")

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                provider_selector = create_sidebar(
                    models_state=state.models_state,
                    endpoint_state=state.endpoint_state,
                    model_state=state.model_state,
                )

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Annotation"):
                        chat = create_chat_area(
                            image_paths_state=state.image_paths_state,
                        )

                    create_comparison_tab()

            with gr.Column(scale=1, min_width=280):
                with gr.Tabs():
                    with gr.Tab("Schema"):
                        gr.Markdown("### Master Schema")
                        gr.Markdown(
                            "View and edit schema keys with their parent hierarchy. "
                            "Users can add keys under level 1+ parents and delete their own additions."
                        )
                        create_schema_viewer_tab()

                    with gr.Tab("Tagging"):
                        gr.Markdown("### Image Tagging")
                        gr.Markdown(
                            "Tag images for dataset creation. Tagging sends images to VLM for automatic "
                            "annotation. Tagged images are saved as YAML files in the dataset/ directory."
                        )
                        create_tagging_tab(
                            provider_selector=provider_selector,
                            endpoint_state=state.endpoint_state,
                            model_state=state.model_state,
                        )

        wire_events(
            provider_selector=provider_selector,
            state=state,
            chat=chat,
        )

    return demo


__all__ = ["create_chat_app"]
