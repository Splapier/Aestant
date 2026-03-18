"""Main application builder for the LLM chat interface.

This module provides the factory function for creating the complete Gradio Blocks
application, composing all UI components and wiring up event handlers from the
chatbot package.
"""

import gradio as gr
from typing import Any

from chatbot import get_available_providers, get_provider
from chatbot.chat_handler import (
    clear_chat,
    process_user_message,
    update_session_on_provider_change,
)
from chatbot.model_fetcher import fetch_models_from_endpoint, refresh_models_manually
from chatbot.image_handler import scan_input_directory, load_images_from_directory


def create_chat_app() -> gr.Blocks:
    """Create and configure the complete LLM chat application.

    This function builds a Gradio Blocks application featuring:
    - A persistent sidebar for selecting LLM providers
    - Dynamic configuration fields that change based on provider type
    - Automatic model fetching from configured endpoints
    - Manual refresh button for updating model lists
    - Loading indicators and status messages
    - A chatbot component supporting streaming responses
    - State management for session persistence
    - Image handling with interactive annotation via ImageEditor components

    Returns:
        gr.Blocks: The configured Gradio application interface.

    Example:
        >>> demo = create_chat_app()
        >>> demo.launch(server_name="127.0.0.1", server_port=7860)
    """
    
    def get_initial_images():
        """Load initial images from input directory."""
        return load_images_from_directory(max_count=2)
    
    with gr.Blocks(title="LLM Chat Interface") as demo:
        # Initialize state for session persistence (defined inside Blocks context)
        session_state = gr.State({})  # Stores provider configuration
        chat_history = gr.State([])   # Stores conversation history
        models_state = gr.State([])   # Available models list
        
        # State components to capture current config values from render context
        endpoint_state = gr.State(value="http://localhost:1234/v1")  # Current endpoint URL
        model_state = gr.State(value="")  # Currently selected model
        
        # Image state for tracking loaded images and editor outputs
        image_paths_state = gr.State(value=[])  # List of loaded image paths
        image_editors_state = gr.State(value=[])  # List of ImageEditor output values

        gr.Markdown("# 🤖 Modular LLM Chat Application")

        with gr.Row():
            # Persistent sidebar for provider selection and configuration
            with gr.Sidebar(open=True):
                gr.Markdown("## Provider Selection")

                # Radio button for selecting the LLM provider
                provider_selector = gr.Radio(
                    choices=get_available_providers(),
                    label="Select LLM Provider",
                    value="lmstudio",  # Default selection
                    interactive=True,
                )

                # Dynamic configuration panel based on selected provider
                @gr.render(inputs=provider_selector)
                def render_provider_config(selected_provider: str):
                    """Render configuration fields specific to the selected provider.

                    This function uses the @gr.render decorator to dynamically update
                    the UI based on the provider selection. Each provider type shows
                    its relevant configuration options including endpoint URL, model
                    dropdown with dynamic fetching, and status display.

                    Args:
                        selected_provider: The currently selected provider type string.
                    """
                    # Get default config values for the selected provider
                    try:
                        provider_instance = get_provider(selected_provider, {})
                        defaults = provider_instance.get_config_defaults()
                    except ValueError:
                        # If we can't create a provider (missing required fields),
                        # show basic fields with empty values
                        defaults = {"endpoint_url": "", "model_name": ""}

                    gr.Markdown(
                        f"### {selected_provider.replace('cpp', '.cpp').title()} Configuration",
                        key=f"{selected_provider}-config-title",
                    )

                    # Render configuration fields based on provider type
                    if selected_provider == "lmstudio":
                        gr.Markdown(
                            "Configure your local LM Studio server connection.",
                            key="lmstudio-desc",
                        )
                        endpoint_input = gr.Textbox(
                            label="Endpoint URL",
                            value=defaults.get("endpoint_url", "http://localhost:1234/v1"),
                            placeholder="e.g., http://localhost:1234/v1",
                            info="Model list refreshes automatically on change",
                            key="lmstudio-endpoint",
                        )

                    elif selected_provider == "llamacpp":
                        gr.Markdown(
                            "Configure your local llama.cpp server connection.",
                            key="llamacpp-desc",
                        )
                        endpoint_input = gr.Textbox(
                            label="Endpoint URL",
                            value=defaults.get("endpoint_url", "http://localhost:8080"),
                            placeholder="e.g., http://localhost:8080",
                            info="Model list refreshes automatically on change",
                            key="llamacpp-endpoint",
                        )
                    else:
                        # Fallback for unknown providers
                        endpoint_input = gr.Textbox(
                            label="Endpoint URL",
                            value="",
                            placeholder="Enter endpoint URL",
                            key=f"{selected_provider}-endpoint",
                        )

                    # Model selection dropdown (dynamically populated)
                    model_dropdown = gr.Dropdown(
                        choices=[],  # Populated via fetch_models_from_endpoint
                        label="Select Model",
                        interactive=True,
                        allow_custom_value=True,  # Allow manual entry if needed
                        info="Leave empty to use default loaded model",
                        key=f"{selected_provider}-model-dropdown",
                    )

                    # Status message display
                    status_display = gr.Markdown(
                        label="Status",
                        value="ℹ️ Configure endpoint and press Refresh or modify URL",
                        key=f"{selected_provider}-status",
                    )

                    # Refresh button for manual update
                    refresh_button = gr.Button(
                        "🔄 Refresh Models", variant="secondary", key=f"{selected_provider}-refresh"
                    )

                    def update_endpoint_state(endpoint_val: str, current_state: str) -> str:
                        """Update the endpoint state with current value.

                        Args:
                            endpoint_val: Current endpoint URL from input.
                            current_state: Current state value (unused).

                        Returns:
                            Updated endpoint URL.
                        """
                        return endpoint_val if endpoint_val else "http://localhost:1234/v1"

                    def update_model_state(
                        model_val: str | None, current_state: str
                    ) -> str:
                        """Update the model state with selected value.

                        Args:
                            model_val: Currently selected model.
                            current_state: Current state value (unused).

                        Returns:
                            Updated model name.
                        """
                        return model_val if model_val else ""

                    # Wire up auto-fetch on endpoint change (within render context)
                    endpoint_input.change(
                        fn=lambda e, p, m: fetch_models_from_endpoint(e, p, m),
                        inputs=[endpoint_input, provider_selector, models_state],
                        outputs=[models_state, status_display, model_dropdown],
                    ).then(
                        fn=update_endpoint_state,
                        inputs=[endpoint_input, endpoint_state],
                        outputs=endpoint_state,
                    )

                    # Wire up manual refresh button click (within render context)
                    refresh_button.click(
                        fn=refresh_models_manually,
                        inputs=[provider_selector, endpoint_input, models_state, model_dropdown],
                        outputs=[models_state, status_display, model_dropdown],
                    ).then(
                        fn=update_endpoint_state,
                        inputs=[endpoint_input, endpoint_state],
                        outputs=endpoint_state,
                    )

                    # Also sync when model dropdown changes
                    model_dropdown.change(
                        fn=update_model_state,
                        inputs=[model_dropdown, model_state],
                        outputs=model_state,
                    )

            # Main chat interface area
            with gr.Column(scale=3):
                # Chatbot component for displaying conversation history
                # Note: In Gradio 6.9.0, Chatbot does not accept a 'type' argument
                chatbot_component = gr.Chatbot(label="Conversation", height=500)

                # Image annotation section - pre-created components updated via event handlers
                gr.Markdown("## 🖼️ Image Annotation (Optional)")
                gr.Markdown(
                    "Place images in the `input/` directory and draw rectangles to highlight regions of interest."
                )
                
                with gr.Row():
                    image_refresh_button = gr.Button("🖼️ Refresh Images", variant="secondary")
                
                # Status message for image loading
                image_status_display = gr.Markdown(
                    value="ℹ️ No images loaded. Click 'Refresh Images' or place files in `input/` directory.",
                    label="Status"
                )
                
                # Pre-create up to 2 ImageEditor components (hidden initially)
                with gr.Row(visible=False) as image_editors_row:
                    editor1_value = {
                        "background": None,
                        "layers": [],
                        "composite": None,
                    }
                    editor2_value = {
                        "background": None,
                        "layers": [],
                        "composite": None,
                    }
                    
                    image_editor_1 = gr.ImageEditor(
                        label="Image 1 (Draw rectangles)",
                        type="numpy",
                        brush=gr.Brush(
                            default_color="#FF0000",
                            colors=["#FF0000", "#00FF00", "#0000FF"]
                        ),
                        interactive=True,
                        value=editor1_value,
                    )
                    
                    image_editor_2 = gr.ImageEditor(
                        label="Image 2 (Draw rectangles)",
                        type="numpy",
                        brush=gr.Brush(
                            default_color="#FF0000",
                            colors=["#FF0000", "#00FF00", "#0000FF"]
                        ),
                        interactive=True,
                        value=editor2_value,
                    )

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

        # =====================================================================
        # EVENT HANDLERS FOR SESSION AND CHAT
        # =====================================================================

        # Event handler: Update session state when provider selection changes
        provider_selector.change(
            fn=update_session_on_provider_change,
            inputs=[provider_selector, session_state],
            outputs=session_state,
        )

        # Event handler: Clear the chat history and image editors
        def clear_all() -> tuple[str, list, gr.update, gr.update, gr.update, list]:
            """Clear chat history and reset image state."""
            return (
                "",  # prompt_input
                [],  # chatbot_component
                gr.update(visible=False),  # image_editors_row
                gr.update(value={"background": None, "layers": [], "composite": None}),  # editor1
                gr.update(value={"background": None, "layers": [], "composite": None}),  # editor2
                [],  # image_paths_state
            )
        
        clear_button.click(
            fn=clear_all,
            inputs=None,
            outputs=[prompt_input, chatbot_component, image_editors_row, image_editor_1, image_editor_2, image_paths_state],
        )

        # Event handler: Refresh images from input directory and update ImageEditor components
        def refresh_images_and_load() -> tuple[gr.update, gr.update, gr.update, list[str]]:
            """Refresh the list of images from the input directory and load into editors.
            
            Returns:
                Tuple of (row_visibility_update, editor1_value, editor2_value, paths_list).
            """
            from chatbot.image_handler import load_image_as_numpy
            
            paths = scan_input_directory(max_count=2)
            
            # Load images into numpy arrays
            image_arrays = []
            for path in paths:
                arr = load_image_as_numpy(path)
                if arr is not None:
                    image_arrays.append(arr)
            
            # Prepare editor values
            editor1_value = gr.update()
            editor2_value = gr.update()
            row_visible = False
            
            if len(image_arrays) >= 1:
                editor1_value = gr.update(
                    value={
                        "background": image_arrays[0],
                        "layers": [],
                        "composite": image_arrays[0],
                    },
                    visible=True
                )
                row_visible = True
                
            if len(image_arrays) >= 2:
                editor2_value = gr.update(
                    value={
                        "background": image_arrays[1],
                        "layers": [],
                        "composite": image_arrays[1],
                    },
                    visible=True
                )
            
            return (
                gr.update(visible=row_visible),  # Row visibility
                editor1_value,  # Editor 1 value
                editor2_value,  # Editor 2 value
                paths,  # Paths list for state
            )
        
        image_refresh_button.click(
            fn=refresh_images_and_load,
            inputs=None,
            outputs=[image_editors_row, image_editor_1, image_editor_2, image_paths_state],
        )

        # Wire up send button and prompt submit for chat processing
        # Using state components (endpoint_state, model_state) to capture config values
        # ImageEditor outputs are now captured directly from the pre-created components
        
        def process_message_with_editors(
            text: str,
            history: list,
            provider: str,
            endpoint: str,
            model: str,
            editor1: dict | None,
            editor2: dict | None,
        ):
            """Process user message with image editors collected directly.
            
            Args:
                text: User's text input.
                history: Chat history state.
                provider: Selected LLM provider.
                endpoint: API endpoint URL.
                model: Selected model name.
                editor1: First ImageEditor output or None.
                editor2: Second ImageEditor output or None.
                
            Yields:
                Tuple of (cleared_input, updated_chat_history).
            """
            # Collect non-None image editor outputs
            editors = []
            if editor1 is not None and editor1.get("background") is not None:
                editors.append(editor1)
            if editor2 is not None and editor2.get("background") is not None:
                editors.append(editor2)
            
            # Call the original process_user_message with collected editors
            # Use yield from to properly forward generator values for streaming
            yield from process_user_message(text, history, provider, endpoint, model, editors)

        send_button.click(
            fn=process_message_with_editors,
            inputs=[
                prompt_input,
                chat_history,
                provider_selector,
                endpoint_state,
                model_state,
                image_editor_1,  # Direct input from ImageEditor component
                image_editor_2,  # Direct input from ImageEditor component
            ],
            outputs=[prompt_input, chatbot_component],
        )

        prompt_input.submit(
            fn=process_message_with_editors,
            inputs=[
                prompt_input,
                chat_history,
                provider_selector,
                endpoint_state,
                model_state,
                image_editor_1,  # Direct input from ImageEditor component
                image_editor_2,  # Direct input from ImageEditor component
            ],
            outputs=[prompt_input, chatbot_component],
        )

    return demo


__all__ = ["create_chat_app"]
