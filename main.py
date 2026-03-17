"""Entry point for the Modular LLM Chat Application.

This script launches the Gradio-based chat interface that supports multiple
local LLM providers (LM Studio and llama.cpp) with streaming responses.

Usage:
    Run this file to start the application:
        uv run python main.py
    
    Or directly with Python:
        python main.py
    
    The application will be available at http://127.0.0.1:7860
"""

from app import demo


if __name__ == "__main__":
    # Launch the Gradio LLM chat application
    # - server_name="127.0.0.1": Bind to localhost only (security)
    # - server_port=7860: Use standard Gradio port
    # - show_error=True: Display detailed error messages in browser
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        quiet=False,
    )
