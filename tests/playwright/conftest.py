"""Shared fixtures for Playwright end-to-end tests.

Manages the Gradio app lifecycle and configures Playwright fixtures
provided by the pytest-playwright plugin.
"""

import pytest
import time
from app import create_chat_app


@pytest.fixture(scope="session")
def app_url():
    """Launch the Gradio app once per test session and yield the URL.

    Starts the app on a local port, waits for it to be ready, then yields
    the URL for tests to navigate to. The app is shut down when the session
    ends.
    """
    app = create_chat_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        prevent_thread_lock=True,
        show_error=True,
    )
    url = "http://127.0.0.1:7860"

    # Give the server a moment to start
    time.sleep(3)

    yield url

    app.close()


@pytest.fixture(scope="session")
def browser_type_launch_args():
    """Configure Playwright to launch browsers headless."""
    return {"headless": True}
