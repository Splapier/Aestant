"""Tests for state management classes.

This module provides unit tests for the state dataclasses used
throughout the chatbot application, covering initialization,
default values, and helper methods.
"""

import pytest

from chatbot.state import (
    ChatSessionState,
    ModelState,
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
