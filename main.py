import os
import streamlit as st

from chat_interface import render_chat_interface
from image_handling import render_image_uploaders
from config import render_config_sidebar, PROVIDER_CONFIGS


def main():
    """Main function for the Streamlit artwork comparison app."""
    st.title("Artwork Comparison")
    
    # Render configuration sidebar first to set up session state
    render_config_sidebar()
    
    # Get selected provider and model from session state
    provider = st.session_state.get("llm_provider", 0)
    provider_options = list(PROVIDER_CONFIGS.keys())
    selected_provider = provider_options[provider] if provider < len(provider_options) else "openai"
    model = st.session_state.get("llm_model", PROVIDER_CONFIGS[selected_provider]["default_model"])
    
    # Set base URL for local models if configured
    if selected_provider == "local":
        base_url = st.session_state.get("local_base_url", "http://localhost:1234/v1")
        os.environ["OPENAI_BASE_URL"] = base_url
    
    # Render image uploaders side by side and get base64 encoded images
    base64_a, base64_b = render_image_uploaders()
    
    # Render chat interface at the bottom with the uploaded images and provider settings
    render_chat_interface(
        base64_image_a=base64_a,
        base64_image_b=base64_b,
        provider=selected_provider,
        model=model
    )


if __name__ == "__main__":
    main()
