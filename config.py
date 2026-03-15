"""Configuration module for LLM provider selection."""

import os
import json
import requests
from typing import Optional, Dict, Any

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


def save_config(config: dict) -> bool:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except IOError:
        return False


def get_default_config() -> dict:
    """Get default configuration values.
    
    Returns:
        dict: Default configuration dictionary
    """
    return {
        "llm_provider": 0,
        "local_url_preset": "llama.cpp (port 8080)",
        "local_base_url": "http://localhost:1234/v1",
        "llm_model": "",
        "custom_local_model": "",
    }


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


def get_provider_by_index(index: int) -> str:
    """Get provider name by index.
    
    Args:
        index: Index of the provider
        
    Returns:
        Provider name string
    """
    provider_options = list(PROVIDER_CONFIGS.keys())
    if 0 <= index < len(provider_options):
        return provider_options[index]
    return "local"


def get_provider_labels() -> list:
    """Get display labels for all providers.
    
    Returns:
        List of provider display names
    """
    return [PROVIDER_CONFIGS[p]["name"] for p in PROVIDER_CONFIGS.keys()]


def update_environment_variables(provider: str, api_key: Optional[str] = None) -> None:
    """Update environment variables based on selected provider and API key.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        api_key: API key to set in environment variable
    """
    if provider in PROVIDER_CONFIGS and api_key:
        env_var = PROVIDER_CONFIGS[provider]["env_var"]
        os.environ[env_var] = api_key
