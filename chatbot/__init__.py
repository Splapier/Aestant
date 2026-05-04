"""Chatbot package for LLM chat interface.

This package contains the core logic for the modular LLM chat application,
including state management, model fetching, chat handling functionality,
embedding generation, average embedding calculation, and similarity search.

Subpackages:
    providers - Provider implementations for different LLM backends

Modules:
    state - State management dataclasses
    model_fetcher - Model fetching logic from provider endpoints
    chat_handler - Chat message processing and response generation
    embedding_engine - CLIP embedding generation for images and tags
    average_embeddings - Running average calculation for embeddings
    similarity_search - Cosine similarity and winner selection

Usage Example:
    >>> from chatbot import get_provider, get_available_providers
    >>> providers = get_available_providers()
    >>> provider = get_provider("lmstudio", {"endpoint_url": "http://localhost:1234/v1"})
"""

from chatbot.providers import get_provider, get_available_providers
from chatbot.state import ChatSessionState, ModelState

__all__ = [
    "get_provider",
    "get_available_providers",
    "ChatSessionState",
    "ModelState",
]
