"""Tests for chat message handling functionality.

This module provides unit tests for the chat_handler module,
covering user message processing, session state updates, chat clearing,
and error handling during provider streaming.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
import numpy as np

from chatbot.chat_handler import (
    update_session_on_provider_change,
    process_user_message,
    clear_chat,
)


class TestUpdateSessionOnProviderChange:
    """Tests for the update_session_on_provider_change function."""

    def test_updates_provider_type(self):
        """Test that provider type is updated in existing session."""
        session = {
            "provider_type": "lmstudio",
            "endpoint_url": "http://localhost:1234/v1",
        }
        result = update_session_on_provider_change("llamacpp", session)
        assert result["provider_type"] == "llamacpp"

    def test_creates_new_session_if_none(self):
        """Test that a new session dict is created when None is passed."""
        result = update_session_on_provider_change("llamacpp", None)
        assert result == {"provider_type": "llamacpp"}

    def test_preserves_existing_session_data(self):
        """Test that existing session fields are preserved on provider change."""
        session = {
            "provider_type": "lmstudio",
            "endpoint_url": "http://localhost:1234/v1",
            "model_name": "test-model",
        }
        result = update_session_on_provider_change("llamacpp", session)
        assert result["endpoint_url"] == "http://localhost:1234/v1"
        assert result["model_name"] == "test-model"
        assert result["provider_type"] == "llamacpp"

    def test_returns_same_session_object(self):
        """Test that the same dict object is returned (mutated in place)."""
        session = {}
        result = update_session_on_provider_change("lmstudio", session)
        assert result is session


class TestProcessUserMessage:
    """Tests for the process_user_message generator function."""

    def _collect_all(
        self,
        prompt,
        history,
        provider_type="lmstudio",
        endpoint_url="http://localhost:1234/v1",
        model_name="",
        image_editors_value=None,
    ):
        """Helper to collect all yields from process_user_message."""
        return list(
            process_user_message(
                prompt,
                history,
                provider_type,
                endpoint_url,
                model_name,
                image_editors_value,
            )
        )

    def test_clears_input_first_yield(self):
        """Test that the first yield clears the input textbox."""
        with patch("chatbot.chat_handler.get_provider"):
            mock_provider = Mock()
            mock_provider.stream_chat.return_value = iter(["Hello!"])
            with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
                results = self._collect_all("Hello", [])
                # First yield should clear the input
                assert results[0][0] == ""

    def test_skips_empty_input(self):
        """Test that empty input is skipped (no message added to history)."""
        with patch("chatbot.chat_handler.get_provider"):
            results = self._collect_all("", [])
            # Only one yield - the clear
            assert len(results) == 1
            assert results[0][0] == ""
            assert results[0][1] == []

    def test_skips_whitespace_input(self):
        """Test that whitespace-only input is skipped."""
        with patch("chatbot.chat_handler.get_provider"):
            results = self._collect_all("   \n  ", [])
            assert len(results) == 1
            assert results[0][1] == []

    def test_adds_user_message_to_history(self):
        """Test that user message is added to conversation history."""
        mock_provider = Mock()
        mock_provider.stream_chat.return_value = iter(["Response"])
        with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
            results = self._collect_all("Hello", [])
            # Second yield has user message (before assistant placeholder)
            assert results[1][1][0]["role"] == "user"
            assert results[1][1][0]["content"] == "Hello"

    def test_streams_response_from_provider(self):
        """Test that provider response chunks are streamed into history."""
        mock_provider = Mock()
        mock_provider.stream_chat.return_value = iter(["Hel", "lo ", "world!"])
        with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
            results = self._collect_all("Say hello", [])
            # Should have: clear + user msg + 3 stream chunks
            assert len(results) == 5
            # Last yield should have complete response
            last_history = results[-1][1]
            assert last_history[-1]["role"] == "assistant"
            assert last_history[-1]["content"] == "Hello world!"

    def test_none_history_initializes_empty(self):
        """Test that None history is handled gracefully."""
        mock_provider = Mock()
        mock_provider.stream_chat.return_value = iter(["Hi"])
        with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
            results = self._collect_all("Test", None)
            # Should not raise, and history should be built
            assert len(results) >= 2

    def test_connection_error_appended_to_history(self):
        """Test that ConnectionError is caught and added to history."""
        mock_provider = Mock()
        mock_provider.stream_chat.side_effect = ConnectionError("Connection refused")
        with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
            results = self._collect_all("Hello", [])
            last_history = results[-1][1]
            last_msg = last_history[-1]
            assert last_msg["role"] == "assistant"
            assert "Connection Error" in last_msg["content"]
            assert "Connection refused" in last_msg["content"]

    def test_value_error_appended_to_history(self):
        """Test that ValueError is caught and added to history."""
        mock_provider = Mock()
        mock_provider.stream_chat.side_effect = ValueError("Invalid config")
        with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
            results = self._collect_all("Hello", [])
            last_history = results[-1][1]
            last_msg = last_history[-1]
            assert last_msg["role"] == "assistant"
            assert "Configuration Error" in last_msg["content"]
            assert "Invalid config" in last_msg["content"]

    def test_runtime_error_appended_to_history(self):
        """Test that RuntimeError is caught and added to history."""
        mock_provider = Mock()
        mock_provider.stream_chat.side_effect = RuntimeError("Model crashed")
        with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
            results = self._collect_all("Hello", [])
            last_history = results[-1][1]
            last_msg = last_history[-1]
            assert last_msg["role"] == "assistant"
            assert "Runtime Error" in last_msg["content"]
            assert "Model crashed" in last_msg["content"]

    def test_generic_exception_appended_to_history(self):
        """Test that unexpected exceptions are caught and added to history."""
        mock_provider = Mock()
        mock_provider.stream_chat.side_effect = TypeError("Bad type")
        with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
            results = self._collect_all("Hello", [])
            last_history = results[-1][1]
            last_msg = last_history[-1]
            assert last_msg["role"] == "assistant"
            assert "Unexpected Error" in last_msg["content"]
            assert "TypeError" in last_msg["content"]

    def test_stream_empty_chunks_skipped(self):
        """Test that empty/falsy stream chunks are not appended."""
        mock_provider = Mock()
        mock_provider.stream_chat.return_value = iter(["Hi", "", None, " there"])
        with patch("chatbot.chat_handler.get_provider", return_value=mock_provider):
            results = self._collect_all("Test", [])
            last_history = results[-1][1]
            assert last_history[-1]["content"] == "Hi there"

    def test_image_editors_processed(self):
        """Test that image editors are passed to prepare_multimodal_payload."""
        mock_provider = Mock()
        mock_provider.stream_chat.return_value = iter(["Nice image!"])
        mock_editor = {
            "background": np.zeros((10, 10, 3), dtype=np.uint8),
            "layers": [],
        }
        with (
            patch("chatbot.chat_handler.get_provider", return_value=mock_provider),
            patch(
                "chatbot.chat_handler.prepare_multimodal_payload",
                return_value=("Test", []),
            ) as mock_prepare,
        ):
            self._collect_all("Test", [], image_editors_value=[mock_editor])
            mock_prepare.assert_called_once()
            assert mock_prepare.call_args[0][0] == "Test"
            assert mock_prepare.call_args[0][1] == [mock_editor]

    def test_provider_created_with_correct_config(self):
        """Test that get_provider is called with correct provider type and config."""
        mock_provider = Mock()
        mock_provider.stream_chat.return_value = iter(["Hi"])
        with patch(
            "chatbot.chat_handler.get_provider", return_value=mock_provider
        ) as mock_get:
            self._collect_all(
                "Hello",
                [],
                provider_type="llamacpp",
                endpoint_url="http://localhost:8080",
                model_name="my-model",
            )
            mock_get.assert_called_once()
            assert mock_get.call_args[0][0] == "llamacpp"
            config = mock_get.call_args[0][1]
            assert config["endpoint_url"] == "http://localhost:8080"
            assert config["model_name"] == "my-model"


class TestClearChat:
    """Tests for the clear_chat function."""

    def test_returns_empty_string_and_list(self):
        """Test that clear_chat returns empty string and empty list."""
        prompt, history = clear_chat()
        assert prompt == ""
        assert history == []

    def test_returns_tuple(self):
        """Test that clear_chat returns a tuple."""
        result = clear_chat()
        assert isinstance(result, tuple)
        assert len(result) == 2
