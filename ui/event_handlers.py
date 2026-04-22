"""Top-level event handler wiring for the chat application.

This module contains the pure handler functions and event wiring for
cross-cutting concerns: provider change, chat clearing, message sending,
and preference election.
"""

import base64
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image

from chatbot.chat_handler import (
    process_user_message,
    update_session_on_provider_change,
)
from chatbot.config_manager import (
    load_models_from_config,
    load_provider_config,
)
from chatbot.preference_election import (
    create_empty_preference_json,
    load_preference_json,
    run_full_election,
)
from chatbot import get_provider
from ui.state_setup import AppState
from ui.chat_area import ChatAreaComponents


def _b64_to_numpy(b64_data_url: str | None) -> np.ndarray | None:
    """Convert a base64 data URL string back to a numpy RGB array.

    Args:
        b64_data_url: Data URL string like "data:image/png;base64,...", or None.

    Returns:
        RGB numpy array (H x W x 3), or None if input is invalid.
    """
    if not b64_data_url:
        return None
    try:
        header, data = b64_data_url.split(",", 1)
        img_bytes = base64.b64decode(data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return np.array(img)
    except Exception:
        return None


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

    def run_election(
        provider: str,
        endpoint: str,
        model: str,
    ):
        """Run the preference election tournament."""
        try:
            pref = create_empty_preference_json()
        except Exception:
            pref = load_preference_json()

        config = {
            "endpoint_url": endpoint.strip(),
            "model_name": model.strip() if model else "",
        }

        yield gr.update(visible=True, value="Starting election...")

        try:
            progress_messages = []
            generator = run_full_election(pref, provider, config)
            result = None

            while True:
                try:
                    progress_msg = next(generator)
                    progress_messages.append(progress_msg)
                    last_msg = progress_msg
                    yield gr.update(visible=True, value=last_msg)
                except StopIteration as e:
                    result = e.value
                    break

            yield gr.update(visible=True, value="Election complete!")

        except Exception as e:
            yield gr.update(visible=True, value=f"Error: {str(e)}")

    chat.run_election_button.click(
        fn=run_election,
        inputs=[
            provider_selector,
            state.endpoint_state,
            state.model_state,
        ],
        outputs=[
            chat.election_status,
        ],
    )
