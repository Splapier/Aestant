"""UI package for the LLM chat application.

This package contains all UI-related code for building the Gradio interface,
including component builders, layout composition, and the main app builder.

Modules:
    components - Reusable Gradio component creation functions
    layouts - Layout composition using Gradio Blocks context managers
    app_builder - Main factory function for creating the complete application

Usage Example:
    >>> from ui import create_chat_app
    >>> demo = create_chat_app()
    >>> demo.launch()
"""

from ui.app_builder import create_chat_app

__all__ = ["create_chat_app"]
