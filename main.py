"""Main module for the Gradio artwork comparison app."""

import os
import gradio as gr
from PIL import Image
from typing import Optional, List, Tuple, Any

from config import (
    PROVIDER_CONFIGS,
    load_config,
    save_config,
    get_default_config,
    fetch_local_models,
    get_provider_by_index,
    get_provider_labels,
    update_environment_variables
)
from image_handling import get_image_data
from chat_interface import process_chat_message


def main():
    """Main function to launch the Gradio application."""
    # Load saved configuration or use defaults
    saved_config = load_config()
    default_config = get_default_config()
    config = {**default_config, **saved_config}
    
    # Default URLs for local server presets
    default_urls = {
        "llama.cpp (port 8080)": "http://localhost:8080/v1",
        "Ollama (default)": "http://localhost:11434/v1",
        "LM Studio (default)": "http://localhost:1234/v1",
        "Custom": ""
    }
    
    # Provider labels for dropdown
    provider_labels = get_provider_labels()

    def update_provider_display(selected_label):
        """Update provider description and show/hide appropriate config sections."""
        provider_index = provider_labels.index(selected_label) if selected_label in provider_labels else 0
        provider_name = get_provider_by_index(provider_index)
        description = PROVIDER_CONFIGS[provider_name]["description"]
        
        is_local = (provider_name == "local")
        return description, gr.update(visible=is_local), gr.update(visible=not is_local)
    
    def update_url_from_preset(preset):
        """Update base URL based on selected preset."""
        if preset != "Custom":
            return default_urls.get(preset, "")
        return None  # Keep current value for Custom
    
    def refresh_models_fn(base_url):
        """Refresh available models from local server."""
        if not base_url:
            return gr.Dropdown(choices=["(enter custom model name)"], value="(enter custom model name)"), "No URL provided"
        
        models = fetch_local_models(base_url)
        if models:
            choices = ["(enter custom model name)"] + models
            status_msg = f"Found {len(models)} model(s)"
        else:
            choices = ["(enter custom model name)"]
            status_msg = "No models found. Enter a custom model name."
        
        return gr.Dropdown(choices=choices, value=choices[0]), status_msg
    
    def update_model_visibility(selected_model):
        """Show/hide custom model input based on selection."""
        return gr.update(visible=selected_model == "(enter custom model name)")
    
    def update_current_settings(provider_label, model, custom_model, base_url, openai_key, anthropic_key, google_key):
        """Update the current settings display."""
        provider_index = provider_labels.index(provider_label) if provider_label in provider_labels else 0
        provider_name = get_provider_by_index(provider_index)
        
        # Use custom model if provided
        actual_model = custom_model if custom_model else model
        
        config_display = {
            "provider": provider_name,
            "model": actual_model if actual_model else "(not set)"
        }
        
        if provider_name != "local":
            has_key = bool(openai_key or anthropic_key or google_key)
            config_display["api_key_set"] = has_key
        else:
            config_display["base_url"] = base_url
        
        return config_display
    
    def save_config_fn(provider_label, url_preset, base_url, model, custom_model, openai_key, anthropic_key, google_key):
        """Save current configuration to file."""
        provider_index = provider_labels.index(provider_label) if provider_label in provider_labels else 0
        
        config = {
            "llm_provider": provider_index,
            "local_url_preset": url_preset,
            "local_base_url": base_url,
            "llm_model": model,
            "custom_local_model": custom_model,
        }
        
        # Save API keys if provided
        if openai_key:
            config["openai_api_key"] = openai_key
        if anthropic_key:
            config["anthropic_api_key"] = anthropic_key
        if google_key:
            config["google_api_key"] = google_key
        
        success = save_config(config)
        status_msg = "Configuration saved!" if success else "Failed to save configuration."
        
        return status_msg, gr.update(visible=True)
    
    def chat_fn(message, history, image_a, image_b, provider_label, model, custom_model, base_url, openai_key, anthropic_key, google_key):
        """Process chat message and return response."""
        # Get provider name
        provider_index = provider_labels.index(provider_label) if provider_label in provider_labels else 0
        provider_name = get_provider_by_index(provider_index)
        
        # Use custom model if provided
        actual_model = custom_model if custom_model else model
        
        # Set base URL for local models
        if provider_name == "local" and base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        
        # Update API keys in environment
        if provider_name == "openai" and openai_key:
            update_environment_variables("openai", openai_key)
        elif provider_name == "anthropic" and anthropic_key:
            update_environment_variables("anthropic", anthropic_key)
        elif provider_name == "google" and google_key:
            update_environment_variables("google", google_key)
        
        # Get base64 encoded images
        base64_a, base64_b = get_image_data(image_a, image_b)
        
        # Process the message
        response = process_chat_message(message, [], base64_a, base64_b, provider_name, actual_model)
        
        # Update history (Gradio Chatbot default format is list of tuples)
        if history is None:
            history = []
        
        history.append((message, response))
        
        return history, ""
    
    def clear_chat_fn():
        """Clear the chat history."""
        return []

    with gr.Blocks(title="Artwork Comparison", fill_height=True) as demo:
        gr.Markdown("# 🎨 Artwork Comparison")
        
        # Sidebar for configuration
        with gr.Sidebar():
            gr.Markdown("## ⚙️ LLM Configuration")
            
            # Save configuration button
            save_btn = gr.Button("💾 Save Configuration", variant="primary")
            save_status = gr.Textbox(label="Status", visible=False, interactive=False)
            
            # Provider selection
            provider_dropdown = gr.Dropdown(
                choices=provider_labels,
                value=provider_labels[config["llm_provider"]],
                label="Provider"
            )
            
            # Provider description
            provider_description = gr.Markdown(PROVIDER_CONFIGS["local"]["description"])
            
            # Local server configuration section
            with gr.Group(visible=True) as local_config_section:
                gr.Markdown("### Local Server Configuration")
                
                # URL preset selection
                url_preset_dropdown = gr.Dropdown(
                    choices=list(default_urls.keys()),
                    value=config["local_url_preset"],
                    label="Server URL Preset"
                )
                
                # Base URL input
                base_url_input = gr.Textbox(
                    label="Base URL",
                    value=default_urls.get(config["local_url_preset"], "http://localhost:1234/v1"),
                    placeholder="e.g., http://localhost:11434/v1"
                )
                
                # Refresh models button and available models dropdown
                refresh_btn = gr.Button("🔄 Refresh Models", variant="secondary")
                local_models_dropdown = gr.Dropdown(
                    choices=["(enter custom model name)"],
                    value="(enter custom model name)",
                    label="Available Models",
                    interactive=True
                )
                
                # Custom model input
                custom_model_input = gr.Textbox(
                    label="Custom Model Name",
                    placeholder="e.g., llava, llama3.2-vision, moondream2",
                    visible=False
                )
                
                # Setup instructions expander
                with gr.Accordion("Local Server Setup Instructions", open=False):
                    gr.Markdown("""### llama.cpp
```bash
# Basic server on port 8080
./build/bin/llama-server -m ./models/model.gguf --port 8080

# With vision support (multimodal)
./build/bin/llama-server \\
    -m ./models/llava-model.gguf \\
    -mm ./models/mmproj.gguf \\
    --port 8080

# Using Docker
docker run -v /path/to/models:/models -p 8080:8080 \\
    ghcr.io/ggml-org/llama.cpp:server \\
    -m /models/model.gguf --port 8080
```
**Base URL:** `http://localhost:8080/v1`

---

### Ollama
```bash
# Start Ollama server (default port 11434)
ollama serve

# Run a vision model
ollama run llava
```
**Base URL:** `http://localhost:11434/v1`

---

### LM Studio
```bash
# Start LM Studio and enable local server in settings
# Default port is 1234
```
**Base URL:** `http://localhost:1234/v1`""")
            
            # Cloud provider configuration section
            with gr.Group(visible=False) as cloud_config_section:
                gr.Markdown("### API Configuration")
                
                # Model selection for cloud providers
                cloud_model_dropdown = gr.Dropdown(
                    choices=PROVIDER_CONFIGS["openai"]["models"],
                    value=PROVIDER_CONFIGS["openai"]["default_model"],
                    label="Model"
                )
                
                # API key input
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Enter your API key"
                )
            
            # Current settings display
            gr.Markdown("### Current Settings")
            current_settings_json = gr.JSON(
                value={"provider": "local", "model": "(not set)"},
                label="Configuration"
            )
        
        # Main content area
        with gr.Row():
            # Image A upload and display
            with gr.Column(scale=1):
                gr.Markdown("## 🖼️ Image A")
                image_a_input = gr.Image(
                    label="Upload Image A",
                    type="pil",
                    sources=["upload"],
                    height=300
                )
            
            # Image B upload and display
            with gr.Column(scale=1):
                gr.Markdown("## 🖼️ Image B")
                image_b_input = gr.Image(
                    label="Upload Image B",
                    type="pil",
                    sources=["upload"],
                    height=300)
        
        # Chat interface section
        gr.Markdown("---")
        gr.Markdown("## 💬 Chat")
        
        # Chatbot component
        chatbot = gr.Chatbot(
            label="Conversation",
            height=400
        )
        
        # Text input for user messages
        chat_input = gr.Textbox(
            label="Message",
            placeholder="Type a message about the images...",
            lines=2
        )
        
        # Send button
        send_btn = gr.Button("Send", variant="primary")
        
        # Clear chat button
        clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # Event handlers
        
        # Provider selection updates
        provider_dropdown.change(
            fn=update_provider_display,
            inputs=[provider_dropdown],
            outputs=[provider_description, local_config_section, cloud_config_section]
        )
        
        # URL preset changes base URL
        url_preset_dropdown.change(
            fn=update_url_from_preset,
            inputs=[url_preset_dropdown],
            outputs=[base_url_input]
        )
        
        # Refresh models button
        refresh_btn.click(
            fn=refresh_models_fn,
            inputs=[base_url_input],
            outputs=[local_models_dropdown, save_status]
        )
        
        # Model selection visibility
        local_models_dropdown.change(
            fn=update_model_visibility,
            inputs=[local_models_dropdown],
            outputs=[custom_model_input]
        )
        
        # Save configuration button
        save_btn.click(
            fn=save_config_fn,
            inputs=[
                provider_dropdown,
                url_preset_dropdown,
                base_url_input,
                local_models_dropdown,
                custom_model_input,
                api_key_input,
                api_key_input,
                api_key_input
            ],
            outputs=[save_status, save_status]
        )
        
        # Update current settings display on various inputs
        for component in [provider_dropdown, local_models_dropdown, custom_model_input, 
                          base_url_input, cloud_model_dropdown]:
            component.change(
                fn=update_current_settings,
                inputs=[provider_dropdown, local_models_dropdown, custom_model_input, 
                       base_url_input, api_key_input, api_key_input, api_key_input],
                outputs=[current_settings_json]
            )
        
        # Chat functionality
        send_btn.click(
            fn=chat_fn,
            inputs=[
                chat_input,
                chatbot,
                image_a_input,
                image_b_input,
                provider_dropdown,
                local_models_dropdown,
                custom_model_input,
                base_url_input,
                api_key_input,
                api_key_input,
                api_key_input
            ],
            outputs=[chatbot, chat_input]
        )
        
        # Allow Enter key to send message
        chat_input.submit(
            fn=chat_fn,
            inputs=[
                chat_input,
                chatbot,
                image_a_input,
                image_b_input,
                provider_dropdown,
                local_models_dropdown,
                custom_model_input,
                base_url_input,
                api_key_input,
                api_key_input,
                api_key_input
            ],
            outputs=[chatbot, chat_input]
        )
        
        # Clear chat button
        clear_chat_btn.click(
            fn=clear_chat_fn,
            inputs=[],
            outputs=[chatbot]
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
