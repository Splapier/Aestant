"""Providers package for LLM chat interfaces.

This package contains implementations of various LLM providers that can be
used with the Gradio chat application. Each provider implements the common
interface defined in BaseProvider, ensuring consistent integration across
different backend services.

Available Providers:
    - LMStudioProvider: For local LM Studio server connections
    - LlamaCppProvider: For local llama.cpp server connections

Usage Example:
    >>> from chatbot.providers import get_provider
    >>> provider = get_provider("lmstudio", {"endpoint_url": "http://localhost:1234/v1"})
    >>> for chunk in provider.stream_chat([{"role": "user", "content": "Hello"}]):
    ...     print(chunk, end="", flush=True)
"""

from chatbot.providers.base import BaseProvider
from chatbot.providers.lmstudio_provider import LMStudioProvider
from chatbot.providers.llamacpp_provider import LlamaCppProvider

# Provider registry mapping provider type strings to their classes
PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "lmstudio": LMStudioProvider,
    "llamacpp": LlamaCppProvider,
}


def get_provider(provider_type: str, config: dict) -> BaseProvider:
    """Factory function to create a provider instance by type.

    Args:
        provider_type: String identifier for the provider ("lmstudio" or "llamacpp").
        config: Configuration dictionary specific to the provider type.

    Returns:
        An instance of the requested provider class implementing BaseProvider.

    Raises:
        ValueError: If provider_type is not recognized.

    Example:
        >>> provider = get_provider("lmstudio", {"endpoint_url": "http://localhost:1234/v1"})
        >>> for chunk in provider.stream_chat([{"role": "user", "content": "Hi"}]):
        ...     print(chunk, end="", flush=True)
    """
    if provider_type not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider type '{provider_type}'. Available providers: {available}"
        )

    return PROVIDER_REGISTRY[provider_type](config)


def get_available_providers() -> list[str]:
    """Return a list of available provider type identifiers.

    Returns:
        List of strings representing available provider types.

    Example:
        >>> get_available_providers()
        ['lmstudio', 'llamacpp']
    """
    return list(PROVIDER_REGISTRY.keys())


__all__ = [
    "BaseProvider",
    "LMStudioProvider",
    "LlamaCppProvider",
    "get_provider",
    "get_available_providers",
    "PROVIDER_REGISTRY",
]
