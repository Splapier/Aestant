"""Sidebar layout and provider configuration panel.

This module builds the persistent sidebar for provider selection and dynamic
configuration. The @gr.render decorator creates provider-specific config fields
that update when the selected provider changes.
"""

import gradio as gr

from chatbot import get_available_providers, get_provider
from chatbot.model_fetcher import fetch_models_from_endpoint, refresh_models_manually
from chatbot.config_manager import (
    has_saved_config,
    load_models_from_config,
    load_provider_config,
    save_provider_config,
)


def _update_endpoint_state(endpoint_val: str, current_state: str) -> str:
    """Sync endpoint textbox value into persistent state."""
    return endpoint_val if endpoint_val else "http://localhost:1234/v1"


def _update_model_state(model_val: str | None, current_state: str) -> str:
    """Sync model dropdown value into persistent state."""
    return model_val if model_val else ""


def _save_config_handler(
    endpoint_val: str, model_val: str, provider: str
) -> tuple[str, str]:
    """Save provider configuration to disk."""
    config = {
        "endpoint_url": endpoint_val.strip() if endpoint_val else "",
        "model_name": model_val.strip() if model_val else "",
    }

    success = save_provider_config(provider, config)

    if success:
        gr.Info("Configuration saved successfully!")
        return f"✅ Configuration saved for {provider}", ""
    else:
        gr.Error("Failed to save configuration")
        return f"❌ Failed to save configuration for {provider}", ""


def _revert_config_handler(provider: str) -> tuple[str, str, str]:
    """Revert provider configuration to defaults."""
    try:
        provider_instance = get_provider(provider, {})
        defaults = provider_instance.get_config_defaults()
    except ValueError:
        defaults = {"endpoint_url": "", "model_name": ""}

    gr.Info("Configuration reverted to defaults")
    return (
        defaults.get("endpoint_url", ""),
        defaults.get("model_name", ""),
        f"↩️ Configuration reverted to defaults for {provider}",
    )


def _fetch_with_validation(
    endpoint_val: str | None,
    provider: str,
    current_models: list[str],
) -> tuple[list[str], str, gr.update]:
    """Fetch models with endpoint URL pre-validation."""
    if endpoint_val is None:
        endpoint_val = ""
    if endpoint_val.strip():
        return fetch_models_from_endpoint(provider, endpoint_val, current_models)
    return (
        current_models,
        "ℹ️ Enter an endpoint URL to fetch models",
        gr.update(choices=current_models),
    )


def create_sidebar(
    models_state: gr.State,
    endpoint_state: gr.State,
    model_state: gr.State,
    prompt_input: gr.Textbox,
) -> gr.Radio:
    """Create the sidebar with provider selection and dynamic config panel.

    Must be called within a gr.Blocks context. The @gr.render decorator creates
    provider-specific configuration fields that re-render on provider change.

    Args:
        models_state: Gradio state for available models list.
        endpoint_state: Gradio state for current endpoint URL.
        model_state: Gradio state for current model name.
        prompt_input: Prompt textbox (referenced by save handler output).

    Returns:
        The provider_selector Radio component for external event wiring.
    """
    with gr.Sidebar(open=True):
        gr.Markdown("## Provider Selection")

        provider_selector = gr.Radio(
            choices=get_available_providers(),
            label="Select LLM Provider",
            value="lmstudio",
            interactive=True,
        )

        @gr.render(inputs=provider_selector)
        def render_provider_config(selected_provider: str):
            """Render configuration fields for the selected provider."""
            try:
                provider_instance = get_provider(selected_provider, {})
                defaults = provider_instance.get_config_defaults()
            except ValueError:
                defaults = {"endpoint_url": "", "model_name": ""}

            loaded_config = load_provider_config(selected_provider, defaults)

            saved_models = []
            if has_saved_config(selected_provider):
                saved_models = load_models_from_config(selected_provider)

            gr.Markdown(
                f"### {selected_provider.replace('cpp', '.cpp').title()} Configuration",
                key=f"{selected_provider}-config-title",
            )

            if selected_provider == "lmstudio":
                gr.Markdown(
                    "Configure your local LM Studio server connection.",
                    key="lmstudio-desc",
                )
                endpoint_input = gr.Textbox(
                    label="Endpoint URL",
                    value=loaded_config.get("endpoint_url"),
                    placeholder="e.g., http://localhost:1234/v1",
                    info="Model list refreshes automatically on change",
                    interactive=True,
                    key="lmstudio-endpoint",
                )
            elif selected_provider == "llamacpp":
                gr.Markdown(
                    "Configure your local llama.cpp server connection.",
                    key="llamacpp-desc",
                )
                endpoint_input = gr.Textbox(
                    label="Endpoint URL",
                    value=loaded_config.get("endpoint_url"),
                    placeholder="e.g., http://localhost:8080",
                    info="Model list refreshes automatically on change",
                    interactive=True,
                    key="llamacpp-endpoint",
                )
            else:
                endpoint_input = gr.Textbox(
                    label="Endpoint URL",
                    value=loaded_config.get("endpoint_url", ""),
                    placeholder="Enter endpoint URL",
                    key=f"{selected_provider}-endpoint",
                )

            model_dropdown = gr.Dropdown(
                choices=saved_models if saved_models else [],
                label="Select Model",
                value=loaded_config.get("model_name", ""),
                interactive=True,
                allow_custom_value=True,
                info="Leave empty to use default loaded model",
                key=f"{selected_provider}-model-dropdown",
            )

            if saved_models:
                status_value = f"✅ Loaded {len(saved_models)} model(s) from config"
            else:
                status_value = "ℹ️ Configure endpoint and press Refresh or modify URL"
            status_display = gr.Markdown(
                label="Status",
                value=status_value,
                key=f"{selected_provider}-status",
            )

            with gr.Row():
                save_button = gr.Button(
                    "💾 Save Config",
                    variant="primary",
                    key=f"{selected_provider}-save",
                )
                revert_button = gr.Button(
                    "↩️ Revert to Defaults",
                    variant="secondary",
                    key=f"{selected_provider}-revert",
                )

            refresh_button = gr.Button(
                "🔄 Refresh Models",
                variant="secondary",
                key=f"{selected_provider}-refresh",
            )

            # Wire up auto-fetch on endpoint change
            endpoint_input.change(
                fn=_fetch_with_validation,
                inputs=[endpoint_input, provider_selector, models_state],
                outputs=[models_state, status_display, model_dropdown],
            ).then(
                fn=_update_endpoint_state,
                inputs=[endpoint_input, endpoint_state],
                outputs=endpoint_state,
            )

            # Wire up manual refresh button
            refresh_button.click(
                fn=refresh_models_manually,
                inputs=[
                    provider_selector,
                    endpoint_input,
                    models_state,
                    model_dropdown,
                ],
                outputs=[models_state, status_display, model_dropdown],
            ).then(
                fn=_update_endpoint_state,
                inputs=[endpoint_input, endpoint_state],
                outputs=endpoint_state,
            )

            # Wire up save button
            save_button.click(
                fn=_save_config_handler,
                inputs=[endpoint_input, model_dropdown, provider_selector],
                outputs=[status_display, prompt_input],
            )

            # Wire up revert button
            revert_button.click(
                fn=_revert_config_handler,
                inputs=[provider_selector],
                outputs=[endpoint_input, model_dropdown, status_display],
            ).then(
                fn=_update_endpoint_state,
                inputs=[endpoint_input, endpoint_state],
                outputs=endpoint_state,
            ).then(
                fn=_update_model_state,
                inputs=[model_dropdown, model_state],
                outputs=model_state,
            )

            # Sync model dropdown changes to state
            model_dropdown.change(
                fn=_update_model_state,
                inputs=[model_dropdown, model_state],
                outputs=model_state,
            )

    return provider_selector
