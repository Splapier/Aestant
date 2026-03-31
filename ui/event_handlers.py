"""Top-level event handler wiring for the chat application.

This module contains the pure handler functions and event wiring for
cross-cutting concerns: provider change, chat clearing, and message sending.
These handlers reference components created by the sidebar and chat area modules.
"""

import gradio as gr

from chatbot.chat_handler import (
    process_user_message,
    update_session_on_provider_change,
)
from chatbot.config_manager import (
    load_models_from_config,
    load_provider_config,
)
from chatbot import get_provider
from ui.state_setup import AppState
from ui.chat_area import ChatAreaComponents


def on_provider_change(provider: str) -> tuple[list[str], str]:
    """Handle provider change — update session and load provider's config."""
    update_session_on_provider_change(provider, {})
    saved_models = load_models_from_config(provider)
    try:
        provider_instance = get_provider(provider, {})
        defaults = provider_instance.get_config_defaults()
    except ValueError:
        defaults = {"endpoint_url": ""}
    loaded_config = load_provider_config(provider, defaults)
    endpoint_url = loaded_config.get("endpoint_url", "")
    return saved_models, endpoint_url


def clear_all() -> tuple[str, list, gr.update, gr.update, gr.update, list]:
    """Clear chat history and reset image state."""
    return (
        "",  # prompt_input
        [],  # chatbot_component
        gr.update(visible=False),  # image_editors_row
        gr.update(
            value={"background": None, "layers": [], "composite": None}
        ),  # editor1
        gr.update(
            value={"background": None, "layers": [], "composite": None}
        ),  # editor2
        [],  # image_paths_state
    )


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

    Yields:
        Tuple of (cleared_input, updated_chat_history).
    """
    editors = []
    if editor1 is not None and editor1.get("background") is not None:
        editors.append(editor1)
    if editor2 is not None and editor2.get("background") is not None:
        editors.append(editor2)

    yield from process_user_message(text, history, provider, endpoint, model, editors)


def wire_events(
    provider_selector: gr.Radio,
    state: AppState,
    chat: ChatAreaComponents,
) -> None:
    """Wire up top-level event handlers for the application.

    Must be called within a gr.Blocks context after all components are created.

    Args:
        provider_selector: Provider selection radio from sidebar.
        state: Application state container.
        chat: Chat area components container.
    """
    # Provider change → update models and endpoint state
    provider_selector.change(
        fn=on_provider_change,
        inputs=[provider_selector],
        outputs=[state.models_state, state.endpoint_state],
    )

    # Clear chat → reset everything
    chat.clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            chat.prompt_input,
            chat.chatbot_component,
            chat.image_editors_row,
            chat.image_editor_1,
            chat.image_editor_2,
            state.image_paths_state,
        ],
    )

    # Send message inputs (shared by button click and enter submit)
    send_inputs = [
        chat.prompt_input,
        state.chat_history,
        provider_selector,
        state.endpoint_state,
        state.model_state,
        chat.image_editor_1,
        chat.image_editor_2,
    ]
    send_outputs = [chat.prompt_input, chat.chatbot_component]

    chat.send_button.click(
        fn=process_message_with_editors,
        inputs=send_inputs,
        outputs=send_outputs,
    )

    chat.prompt_input.submit(
        fn=process_message_with_editors,
        inputs=send_inputs,
        outputs=send_outputs,
    )
