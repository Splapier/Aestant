"""Tests for state management classes.

This module provides unit tests for the state dataclasses used
throughout the chatbot application, covering initialization,
default values, and helper methods.
"""

import pytest

from chatbot.state import (
    ChatSessionState,
    ModelState,
    ChatHistoryState,
)


class TestChatSessionState:
    """Tests for the ChatSessionState dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        state = ChatSessionState()
        assert state.provider_type == "lmstudio"
        assert state.endpoint_url == "http://localhost:1234/v1"
        assert state.model_name == ""

    def test_can_set_provider_type(self):
        """Test setting a custom provider type."""
        state = ChatSessionState()
        state.provider_type = "llamacpp"
        assert state.provider_type == "llamacpp"

    def test_can_set_endpoint_url(self):
        """Test setting a custom endpoint URL."""
        state = ChatSessionState()
        state.endpoint_url = "http://localhost:8080"
        assert state.endpoint_url == "http://localhost:8080"

    def test_can_set_model_name(self):
        """Test setting a model name."""
        state = ChatSessionState()
        state.model_name = "my-model"
        assert state.model_name == "my-model"

    def test_initialization_with_kwargs(self):
        """Test creating state with custom values via kwargs."""
        state = ChatSessionState(
            provider_type="llamacpp",
            endpoint_url="http://localhost:8080",
            model_name="test-model",
        )
        assert state.provider_type == "llamacpp"
        assert state.endpoint_url == "http://localhost:8080"
        assert state.model_name == "test-model"


class TestModelState:
    """Tests for the ModelState dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        state = ModelState()
        assert state.available_models == []
        assert state.selected_model is None

    def test_can_set_available_models(self):
        """Test setting available models."""
        state = ModelState()
        state.available_models = ["model-a", "model-b"]
        assert state.available_models == ["model-a", "model-b"]

    def test_can_set_selected_model(self):
        """Test setting selected model."""
        state = ModelState()
        state.selected_model = "model-a"
        assert state.selected_model == "model-a"

    def test_empty_models_default(self):
        """Test that default models list is empty."""
        state = ModelState()
        assert len(state.available_models) == 0

    def test_initialization_with_kwargs(self):
        """Test creating state with custom values via kwargs."""
        state = ModelState(
            available_models=["m1", "m2"],
            selected_model="m2",
        )
        assert state.available_models == ["m1", "m2"]
        assert state.selected_model == "m2"

    def test_models_list_is_new_instance(self):
        """Test that each instance gets its own models list."""
        state_a = ModelState()
        state_b = ModelState()
        state_a.available_models.append("model-1")
        assert state_b.available_models == []


class TestChatHistoryState:
    """Tests for the ChatHistoryState dataclass."""

    def test_default_empty_messages(self):
        """Test that default messages list is empty."""
        state = ChatHistoryState()
        assert state.messages == []

    def test_add_user_message(self):
        """Test adding a user message."""
        state = ChatHistoryState()
        state.add_user_message("Hello!")
        assert len(state.messages) == 1
        assert state.messages[0] == {"role": "user", "content": "Hello!"}

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        state = ChatHistoryState()
        state.add_assistant_message("Hi there!")
        assert len(state.messages) == 1
        assert state.messages[0] == {"role": "assistant", "content": "Hi there!"}

    def test_add_multiple_messages(self):
        """Test adding alternating user and assistant messages."""
        state = ChatHistoryState()
        state.add_user_message("Question")
        state.add_assistant_message("Answer")
        state.add_user_message("Follow-up")
        assert len(state.messages) == 3
        assert state.messages[0]["role"] == "user"
        assert state.messages[1]["role"] == "assistant"
        assert state.messages[2]["role"] == "user"

    def test_clear_empties_messages(self):
        """Test that clear removes all messages."""
        state = ChatHistoryState()
        state.add_user_message("msg1")
        state.add_assistant_message("msg2")
        state.clear()
        assert state.messages == []

    def test_clear_on_empty_history(self):
        """Test that clear on empty history does not raise."""
        state = ChatHistoryState()
        state.clear()
        assert state.messages == []

    def test_get_last_assistant_message_index_found(self):
        """Test finding the last assistant message index."""
        state = ChatHistoryState()
        state.add_user_message("Hello")
        state.add_assistant_message("Response 1")
        state.add_user_message("Follow-up")
        state.add_assistant_message("Response 2")

        index = state.get_last_assistant_message_index()
        assert index == 3

    def test_get_last_assistant_message_index_not_found(self):
        """Test returns None when no assistant messages exist."""
        state = ChatHistoryState()
        state.add_user_message("Hello")
        state.add_user_message("Anyone there?")

        index = state.get_last_assistant_message_index()
        assert index is None

    def test_get_last_assistant_message_index_empty(self):
        """Test returns None when history is empty."""
        state = ChatHistoryState()
        index = state.get_last_assistant_message_index()
        assert index is None

    def test_update_last_assistant_message_success(self):
        """Test updating the last assistant message content."""
        state = ChatHistoryState()
        state.add_user_message("Hello")
        state.add_assistant_message("Old response")

        result = state.update_last_assistant_message("New response")
        assert result is True
        assert state.messages[1]["content"] == "New response"

    def test_update_last_assistant_message_failure(self):
        """Test returns False when no assistant message exists."""
        state = ChatHistoryState()
        state.add_user_message("Hello")

        result = state.update_last_assistant_message("Update")
        assert result is False

    def test_update_last_assistant_message_empty(self):
        """Test returns False when history is empty."""
        state = ChatHistoryState()
        result = state.update_last_assistant_message("Update")
        assert result is False

    def test_update_last_updates_only_last(self):
        """Test that update_last only modifies the most recent assistant message."""
        state = ChatHistoryState()
        state.add_assistant_message("First")
        state.add_user_message("Question")
        state.add_assistant_message("Second")

        state.update_last_assistant_message("Updated Second")

        # First assistant message should be unchanged
        assert state.messages[0]["content"] == "First"
        # Last assistant message should be updated
        assert state.messages[2]["content"] == "Updated Second"
