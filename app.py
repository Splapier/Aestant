"""Main entry point for the LLM chat application.

This module serves as the minimal entry point for launching the Gradio-based
LLM chat interface. All core functionality has been extracted into dedicated
packages:

- `chatbot`: Core logic including state management, model fetching, and chat handling
- `ui`: UI components and application builder

Usage:
    >>> python app.py

Or programmatically:
    >>> from ui import create_chat_app
    >>> demo = create_chat_app()
    >>> demo.launch(server_name="127.0.0.1", server_port=7860)
"""

from ui import create_chat_app


# Create the application instance
demo = create_chat_app()


if __name__ == "__main__":
    # Launch the Gradio application
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
