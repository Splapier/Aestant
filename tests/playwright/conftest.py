"""Shared fixtures for Playwright end-to-end tests.

Manages the Gradio app lifecycle and configures Playwright fixtures
provided by the pytest-playwright plugin.
"""

import os

import pytest
import time
from pathlib import Path
from PIL import Image
from app import create_chat_app

INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "input"


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    """Point CONFIG_DIR to a temporary directory for test isolation."""
    monkeypatch.setattr("chatbot.config_manager.CONFIG_DIR", tmp_path)
    import chatbot.config_manager

    chatbot.config_manager.CONFIG_DIR = tmp_path
    return tmp_path


@pytest.fixture(scope="session")
def _test_images_session():
    """Ensure test images exist in input/ for the duration of the session.

    Called before app_url so that create_chat_app() can auto-load them.
    """
    paths = []
    for i, color in enumerate(["red", "blue"]):
        p = INPUT_DIR / f"test_image_{i}.png"
        if not p.exists():
            Image.new("RGB", (50, 50), color).save(p)
        paths.append(p)
    yield [str(p) for p in paths]
    for p in paths:
        if p.exists():
            os.remove(p)


@pytest.fixture(scope="session")
def app_url(_test_images_session):
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
