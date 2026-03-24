"""Tests for model fetcher and endpoint URL validation.

This module provides unit tests for the model fetching functionality,
URL validation, and provider switching behavior to ensure the fixes
for the provider switching and URL validation bugs remain working.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import gradio as gr

from chatbot.model_fetcher import (
    validate_endpoint_url,
    fetch_models_from_endpoint,
    refresh_models_manually,
)


class TestValidateEndpointUrl:
    """Tests for endpoint URL validation."""

    def test_valid_http_url(self):
        """Test that valid http:// URL is accepted."""
        is_valid, error = validate_endpoint_url("http://localhost:1234/v1")
        assert is_valid is True
        assert error == ""

    def test_valid_https_url(self):
        """Test that valid https:// URL is accepted."""
        is_valid, error = validate_endpoint_url("https://api.example.com/v1")
        assert is_valid is True
        assert error == ""

    def test_valid_url_with_path(self):
        """Test that URL with path is accepted."""
        is_valid, error = validate_endpoint_url("http://localhost:8080/models")
        assert is_valid is True
        assert error == ""

    def test_url_without_protocol_rejected(self):
        """Test that URL without http/https is rejected."""
        is_valid, error = validate_endpoint_url("localhost:1234")
        assert is_valid is False
        assert "Invalid URL format" in error
        assert "http:// or https://" in error

    def test_empty_string_rejected(self):
        """Test that empty string is rejected."""
        is_valid, error = validate_endpoint_url("")
        assert is_valid is False
        assert "Please enter a valid endpoint URL" in error

    def test_none_url_rejected(self):
        """Test that None is rejected."""
        is_valid, error = validate_endpoint_url(None)
        assert is_valid is False
        assert "Please enter a valid endpoint URL" in error

    def test_whitespace_only_rejected(self):
        """Test that whitespace-only string is rejected."""
        is_valid, error = validate_endpoint_url("   ")
        assert is_valid is False


class TestFetchModelsFromEndpoint:
    """Tests for model fetching functionality."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        mock = Mock()
        mock.fetch_models.return_value = ["model-1", "model-2"]
        return mock

    def test_empty_endpoint_returns_current_models(self):
        """Test that empty endpoint returns current models without error."""
        models, status, update = fetch_models_from_endpoint("lmstudio", "", [])
        assert models == []
        assert "endpoint url" in status.lower()
        assert update["choices"] == []

    def test_none_endpoint_returns_current_models(self):
        """Test that None endpoint returns current models without error."""
        models, status, update = fetch_models_from_endpoint("lmstudio", None, [])
        assert models == []
        assert "endpoint url" in status.lower()
        assert update["choices"] == []

    def test_whitespace_endpoint_returns_current_models(self):
        """Test that whitespace endpoint returns current models without error."""
        models, status, update = fetch_models_from_endpoint(
            "lmstudio", "   ", ["existing-model"]
        )
        assert models == ["existing-model"]

    @patch("chatbot.model_fetcher.get_provider")
    def test_valid_endpoint_fetches_models(self, mock_get_provider):
        """Test that valid endpoint fetches models successfully."""
        mock_provider = Mock()
        mock_provider.fetch_models.return_value = ["model-a", "model-b"]
        mock_get_provider.return_value = mock_provider

        models, status, update = fetch_models_from_endpoint(
            "lmstudio", "http://localhost:1234/v1", []
        )

        assert models == ["model-a", "model-b"]
        assert "Found" in status
        assert "2" in status

    @patch("chatbot.model_fetcher.get_provider")
    def test_argument_order_correct(self, mock_get_provider):
        """Test that provider and endpoint arguments are passed in correct order.

        This test specifically verifies the fix for Bug 2 where arguments
        were being passed in wrong order causing URL validation to fail.
        """
        mock_provider = Mock()
        mock_provider.fetch_models.return_value = ["test-model"]
        mock_get_provider.return_value = mock_provider

        fetch_models_from_endpoint("lmstudio", "http://localhost:1234/v1", [])

        mock_get_provider.assert_called_once()
        call_args = mock_get_provider.call_args
        config = call_args[0][1]
        assert config["endpoint_url"] == "http://localhost:1234/v1"

    @patch("chatbot.model_fetcher.get_provider")
    def test_invalid_url_returns_error_status(self, mock_get_provider):
        """Test that invalid URL format returns error status."""
        mock_provider = Mock()
        mock_provider.fetch_models.return_value = []
        mock_get_provider.return_value = mock_provider

        models, status, update = fetch_models_from_endpoint(
            "lmstudio", "invalid-url", []
        )

        assert "Invalid URL format" in status
        assert models == []

    @patch("chatbot.model_fetcher.get_provider")
    def test_no_models_found_returns_info_status(self, mock_get_provider):
        """Test that empty model list returns informational status."""
        mock_provider = Mock()
        mock_provider.fetch_models.return_value = []
        mock_get_provider.return_value = mock_provider

        models, status, update = fetch_models_from_endpoint(
            "lmstudio", "http://localhost:1234/v1", []
        )

        assert "No models found" in status

    @patch("chatbot.model_fetcher.get_provider")
    def test_connection_error_returns_error_status(self, mock_get_provider):
        """Test that connection errors are handled gracefully."""
        from requests.exceptions import ConnectionError as RequestsConnectionError

        mock_get_provider.side_effect = RequestsConnectionError("Connection refused")

        models, status, update = fetch_models_from_endpoint(
            "lmstudio", "http://localhost:1234/v1", ["existing-model"]
        )

        assert "ConnectionError" in status or "Connection Error" in status
        assert models == ["existing-model"]


class TestRefreshModelsManually:
    """Tests for manual model refresh functionality."""

    @patch("chatbot.model_fetcher.fetch_models_from_endpoint")
    @patch("chatbot.model_fetcher.save_models_to_config")
    def test_refresh_saves_models_to_config(self, mock_save, mock_fetch):
        """Test that refresh saves models to config file."""
        mock_fetch.return_value = (
            ["new-model"],
            "✅ Found 1 model(s)",
            {"choices": ["new-model"], "value": "new-model"},
        )

        models, status, update = refresh_models_manually(
            "lmstudio", "http://localhost:1234/v1", [], None
        )

        mock_save.assert_called_once_with("lmstudio", ["new-model"])

    @patch("chatbot.model_fetcher.fetch_models_from_endpoint")
    @patch("chatbot.model_fetcher.save_models_to_config")
    def test_refresh_preserves_selected_model(self, mock_save, mock_fetch):
        """Test that refresh preserves currently selected model."""
        mock_fetch.return_value = (
            ["model-a", "model-b"],
            "✅ Found 2 model(s)",
            {"choices": ["model-a", "model-b"], "value": "model-a"},
        )

        models, status, update = refresh_models_manually(
            "lmstudio", "http://localhost:1234/v1", [], "model-a"
        )

        assert update["value"] == "model-a"

    @patch("chatbot.model_fetcher.fetch_models_from_endpoint")
    @patch("chatbot.model_fetcher.save_models_to_config")
    def test_refresh_does_not_save_if_no_models(self, mock_save, mock_fetch):
        """Test that refresh doesn't save if no models found."""
        mock_fetch.return_value = (
            [],
            "ℹ️ No models found at this endpoint",
            {"choices": []},
        )

        refresh_models_manually("lmstudio", "http://localhost:1234/v1", [], None)

        mock_save.assert_not_called()
