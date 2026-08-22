"""Entry point for the DINOv2 + LinUCB image preference app.

This script launches the Gradio-based image preference bandit: it extracts
dense DINOv2 features for all images in the images directory (cached on
disk), scores random batches with a LinUCB bandit, shows the top-2
candidates, and learns from each user choice.

Usage:
    Run this file to start the application:
        uv run python bandit_app.py

    The application will be available at http://127.0.0.1:7861
"""

from image_bandit.app import create_bandit_app

demo = create_bandit_app()


if __name__ == "__main__":
    # Launch the image preference bandit application
    # - server_port=7861: keep clear of the chat app on port 7860
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
        quiet=False,
    )
