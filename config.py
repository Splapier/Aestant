"""Configuration component for LLM provider selection."""

import os
import json
import requests
import streamlit as st

CONFIG_FILE = ".aestant_config.json"


def load_config() -> dict:
    """Load saved configuration from file.
    
    Returns:
        dict: Configuration dictionary with saved settings
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {}


def save_config(config: dict):
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
    """
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        st.error(f"Failed to save configuration: {e}")


def initialize_session_state():
    """Initialize session state from saved config or defaults."""
    saved_config = load_config()
    
    # Provider selection (index)
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = saved_config.get("llm_provider", 0)
    
    # Local URL preset
    if "local_url_preset" not in st.session_state:
        st.session_state.local_url_preset = saved_config.get("local_url_preset", "llama.cpp (port 8080)")
    
    # Local base URL
    if "local_base_url" not in st.session_state:
        st.session_state.local_base_url = saved_config.get("local_base_url", "http://localhost:1234/v1")
    
    # Model selection
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = saved_config.get("llm_model", "")
    
    # Custom local model
    if "custom_local_model" not in st.session_state:
        st.session_state.custom_local_model = saved_config.get("custom_local_model", "")
    
    # API keys for cloud providers
    for provider in ["openai", "anthropic", "google"]:
        key_name = f"{provider}_api_key"
        if key_name not in st.session_state:
            st.session_state[key_name] = saved_config.get(key_name, "")


def save_current_config():
    """Save current session state to config file."""
    config = {
        "llm_provider": st.session_state.get("llm_provider", 0),
        "local_url_preset": st.session_state.get("local_url_preset", "llama.cpp (port 8080)"),
        "local_base_url": st.session_state.get("local_base_url", ""),
        "llm_model": st.session_state.get("llm_model", ""),
        "custom_local_model": st.session_state.get("custom_local_model", ""),
    }
    
    # Save API keys for cloud providers
    for provider in ["openai", "anthropic", "google"]:
        key_name = f"{provider}_api_key"
        if st.session_state.get(key_name, ""):
            config[key_name] = st.session_state[key_name]
    
    save_config(config)


# Provider configurations with available models
PROVIDER_CONFIGS = {
    "local": {
        "name": "Local (llama.cpp/Ollama/LM Studio)",
        "default_model": "",  # No default - user must specify
        "models": [],  # Will be populated from server or manual input
        "env_var": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "description": "Local models via OpenAI-compatible API (llama.cpp, Ollama, LM Studio)"
    },
    "openai": {
        "name": "OpenAI",
        "default_model": "gpt-4o",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        "env_var": "OPENAI_API_KEY",
        "description": "OpenAI models (ChatGPT)"
    },
    "anthropic": {
        "name": "Anthropic",
        "default_model": "claude-sonnet-4-5-20250929",
        "models": [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-6",
            "claude-haiku-4-5"
        ],
        "env_var": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude models"
    },
    "google": {
        "name": "Google",
        "default_model": "gemini-2.5-flash",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.0-flash-thinking-exp",
            "gemini-1.5-pro"
        ],
        "env_var": "GOOGLE_API_KEY",
        "description": "Google Gemini models"
    }
}


def fetch_local_models(base_url: str) -> list:
    """Fetch available models from a local server.
    
    Tries multiple endpoints to support different servers:
    - OpenAI-compatible /v1/models (llama.cpp, LM Studio)
    - Ollama-specific /api/tags endpoint
    
    Args:
        base_url: Base URL of the local server (e.g., http://localhost:11434/v1)
        
    Returns:
        List of model names available on the server
    """
    models = []
    
    try:
        # Try OpenAI-compatible /v1/models endpoint first
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                for model in data["data"]:
                    # Handle different field names (id or name)
                    model_id = model.get("id") or model.get("name", "")
                    if model_id:
                        models.append(model_id)
        
        # If no models found, try Ollama-specific /api/tags endpoint (without /v1 suffix)
        if not models:
            ollama_base = base_url.replace("/v1", "", 1).rstrip("/")
            response = requests.get(f"{ollama_base}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "models" in data and isinstance(data["models"], list):
                    for model in data["models"]:
                        model_name = model.get("name") or model.get("model", "")
                        if model_name:
                            models.append(model_name)
        
        return models
    except requests.RequestException:
        return []
    except Exception:
        return []


def render_config_sidebar():
    """Render the configuration sidebar for LLM provider selection.
    
    Returns:
        tuple: (provider, model) - Selected provider and model names
    """
    # Initialize session state from saved config
    initialize_session_state()
    
    with st.sidebar:
        st.header("LLM Configuration")
        
        # Save configuration button at the top
        if st.button("💾 Save Configuration", key="save_config_button"):
            save_current_config()
            st.success("Configuration saved!")
        
        # Provider selection
        provider_options = list(PROVIDER_CONFIGS.keys())
        provider_labels = [PROVIDER_CONFIGS[p]["name"] for p in provider_options]
        
        selected_index = st.selectbox(
            "Provider",
            options=range(len(provider_labels)),
            format_func=lambda x: provider_labels[x],
            index=0,  # Default to Local
            key="llm_provider"
        )
        provider = provider_options[selected_index]
        
        # Display provider description
        st.info(PROVIDER_CONFIGS[provider]["description"])
        
        if provider == "local":
            # For local models, show base URL input and model fetching
            st.subheader("Local Server Configuration")
            
            # Base URL preset selection
            default_urls = {
                "llama.cpp (port 8080)": "http://localhost:8080/v1",
                "Ollama (default)": "http://localhost:11434/v1",
                "LM Studio (default)": "http://localhost:1234/v1",
                "Custom": ""
            }
            
            selected_url_key = st.selectbox(
                "Server URL Preset",
                options=list(default_urls.keys()),
                index=0,
                key="local_url_preset"
            )
            
            base_url = st.text_input(
                "Base URL",
                value=default_urls[selected_url_key] if selected_url_key != "Custom" else "",
                help="URL of your local LLM server with /v1 suffix (e.g., http://localhost:11434/v1)",
                key="local_base_url"
            )
            
            # Button to fetch available models
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Available Models**")
            with col2:
                if st.button("🔄 Refresh", key="refresh_models"):
                    st.session_state.local_models_fetched = False
            
            # Fetch models if not already fetched or refresh requested
            if "local_models_fetched" not in st.session_state:
                st.session_state.local_models_fetched = False
            
            if not st.session_state.get("local_models_fetched", True):
                with st.spinner(f"Fetching models from {base_url}..."):
                    available_models = fetch_local_models(base_url)
                    st.session_state.local_available_models = available_models
                    st.session_state.local_models_fetched = True
                
                if available_models:
                    st.success(f"Found {len(available_models)} model(s)")
                else:
                    st.warning("No models found. Enter a custom model name below.")
            
            # Get available models from session state or empty list
            available_models = st.session_state.get("local_available_models", [])
            
            # Model selection with option to add custom
            model_options = ["(enter custom model name)"] + available_models if available_models else []
            model = st.selectbox(
                "Select or enter model",
                options=model_options,
                index=0,
                key="llm_model"
            )
            
            # If user selected to enter custom, show text input
            if model == "(enter custom model name)":
                model = st.text_input(
                    "Custom Model Name",
                    placeholder="e.g., llava, llama3.2-vision, moondream2",
                    key="custom_local_model"
                )
            
            # Show detailed instructions for different local servers
            with st.expander("Local Server Setup Instructions", expanded=False):
                st.markdown("### llama.cpp")
                st.code(
                    """# Basic server on port 8080
./build/bin/llama-server -m ./models/model.gguf --port 8080

# With vision support (multimodal)
./build/bin/llama-server \\
    -m ./models/llava-model.gguf \\
    -mm ./models/mmproj.gguf \\
    --port 8080

# Using Docker
docker run -v /path/to/models:/models -p 8080:8080 \\
    ghcr.io/ggml-org/llama.cpp:server \\
    -m /models/model.gguf --port 8080""",
                    language="bash"
                )
                st.markdown("**Base URL:** `http://localhost:8080/v1`")
                
                st.divider()
                
                st.markdown("### Ollama")
                st.code(
                    """# Start Ollama server (default port 11434)
ollama serve

# Run a vision model
ollama run llava""",
                    language="bash"
                )
                st.markdown("**Base URL:** `http://localhost:11434/v1`")
                
                st.divider()
                
                st.markdown("### LM Studio")
                st.code(
                    """# Start LM Studio and enable local server in settings
# Default port is 1234""",
                    language="bash"
                )
                st.markdown("**Base URL:** `http://localhost:1234/v1`")
        else:
            # For cloud providers, show model selection and API key input
            config = PROVIDER_CONFIGS[provider]
            model = st.selectbox(
                "Model",
                options=config["models"],
                index=0,
                key="llm_model"
            )
            
            api_key_help = f"API key for {config['name']} (or set as environment variable)"
            api_key = st.text_input(
                "API Key",
                type="password",
                help=api_key_help,
                key=f"{provider}_api_key"
            )
            if api_key:
                os.environ[config["env_var"]] = api_key
        
        # Display current configuration
        st.divider()
        st.subheader("Current Settings")
        
        config_display = {
            "provider": provider,
            "model": model if model else "(not set)"
        }
        
        if provider != "local":
            config_display["api_key_set"] = bool(st.session_state.get(f"{provider}_api_key", ""))
        else:
            config_display["base_url"] = st.session_state.get("local_base_url", "")
        
        st.json(config_display)
