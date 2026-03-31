"""UI package for the LLM chat application.

This package contains all UI-related code for building the Gradio interface,
including the main app builder that composes all components and event handlers.

Usage Example:
    >>> from ui import create_chat_app
    >>> demo = create_chat_app()
    >>> demo.launch()
"""

from ui.app_builder import create_chat_app

__all__ = ["create_chat_app"]
