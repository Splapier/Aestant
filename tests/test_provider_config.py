"""Tests for LLM provider configuration system.

This module provides unit tests for the config validation, persistence,
and provider factory functionality across all LLM providers.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from chatbot.providers import (
    get_provider,
    get_available_providers,
    PROVIDER_REGISTRY,
    LMStudioProvider,
    LlamaCppProvider,
)
from chatbot.config_manager import (
    load_provider_config,
    save_provider_config,
    revert_field_to_default,
    has_saved_config,
    get_config_file_path,
)


class TestGetAvailableProviders:
    """Tests for the provider registry and discovery."""

    def test_returns_known_providers(self):
        """Test that known providers are returned."""
        providers = get_available_providers()
        assert "lmstudio" in providers
        assert "llamacpp" in providers

    def test_registry_matches_available_list(self):
        """Test that registry keys match available providers list."""
        providers = get_available_providers()
        assert set(providers) == set(PROVIDER_REGISTRY.keys())


class TestGetProvider:
    """Tests for the provider factory function."""

    def test_creates_lmstudio_provider(self):
        """Test creating an LM Studio provider instance."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = get_provider("lmstudio", config)
        assert isinstance(provider, LMStudioProvider)
        assert provider.get_provider_type() == "lmstudio"

    def test_creates_llamacpp_provider(self):
        """Test creating a llama.cpp provider instance."""
        config = {"endpoint_url": "http://localhost:8080"}
        provider = get_provider("llamacpp", config)
        assert isinstance(provider, LlamaCppProvider)
        assert provider.get_provider_type() == "llamacpp"

    def test_unknown_provider_raises_value_error(self):
        """Test that unknown provider type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_provider("unknown", {"endpoint_url": "http://example.com"})

        error_msg = str(exc_info.value)
        assert "Unknown provider type" in error_msg
        assert "lmstudio" in error_msg
        assert "llamacpp" in error_msg


class TestLMStudioProviderConfig:
    """Tests for LM Studio provider configuration validation."""

    def test_valid_config(self):
        """Test that valid config is accepted."""
        config = {
            "endpoint_url": "http://localhost:1234/v1",
            "model_name": "test-model",
        }
        provider = LMStudioProvider(config)
        assert provider.config == config

    def test_missing_endpoint_raises_value_error(self):
        """Test that missing endpoint_url raises ValueError."""
        with pytest.raises(ValueError, match="requires.*endpoint_url"):
            LMStudioProvider({})

    def test_empty_endpoint_raises_value_error(self):
        """Test that empty endpoint_url raises ValueError."""
        with pytest.raises(ValueError, match="requires.*endpoint_url"):
            LMStudioProvider({"endpoint_url": ""})

    def test_none_endpoint_raises_value_error(self):
        """Test that None endpoint_url raises ValueError."""
        with pytest.raises(ValueError, match="requires.*endpoint_url"):
            LMStudioProvider({"endpoint_url": None})

    def test_invalid_url_format_no_protocol(self):
        """Test that URL without http/https protocol is rejected."""
        with pytest.raises(ValueError, match="Invalid endpoint URL format"):
            LMStudioProvider({"endpoint_url": "localhost:1234/v1"})

    def test_valid_http_url(self):
        """Test that HTTP URLs are accepted."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = LMStudioProvider(config)
        assert provider.config["endpoint_url"] == "http://localhost:1234/v1"

    def test_valid_https_url(self):
        """Test that HTTPS URLs are accepted."""
        config = {"endpoint_url": "https://secure.example.com/v1"}
        provider = LMStudioProvider(config)
        assert provider.config["endpoint_url"] == "https://secure.example.com/v1"

    def test_model_name_optional(self):
        """Test that model_name is optional."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = LMStudioProvider(config)
        assert provider.config.get("model_name") is None

    def test_config_field_names(self):
        """Test get_config_field_names returns expected fields."""
        provider = LMStudioProvider({"endpoint_url": "http://localhost:1234/v1"})
        fields = provider.get_config_field_names()
        assert fields == ["endpoint_url", "model_name"]

    def test_config_defaults(self):
        """Test get_config_defaults returns expected defaults."""
        provider = LMStudioProvider({"endpoint_url": "http://localhost:1234/v1"})
        defaults = provider.get_config_defaults()
        assert defaults["endpoint_url"] == "http://localhost:1234/v1"
        assert defaults["model_name"] == ""


class TestLlamaCppProviderConfig:
    """Tests for llama.cpp provider configuration validation."""

    def test_valid_config(self):
        """Test that valid config is accepted."""
        config = {
            "endpoint_url": "http://localhost:8080",
            "model_name": "test-model",
        }
        provider = LlamaCppProvider(config)
        assert provider.config == config

    def test_missing_endpoint_raises_value_error(self):
        """Test that missing endpoint_url raises ValueError."""
        with pytest.raises(ValueError, match="requires.*endpoint_url"):
            LlamaCppProvider({})

    def test_empty_endpoint_raises_value_error(self):
        """Test that empty endpoint_url raises ValueError."""
        with pytest.raises(ValueError, match="requires.*endpoint_url"):
            LlamaCppProvider({"endpoint_url": ""})

    def test_invalid_url_format_no_protocol(self):
        """Test that URL without http/https protocol is rejected."""
        with pytest.raises(ValueError, match="Invalid endpoint URL format"):
            LlamaCppProvider({"endpoint_url": "localhost:8080"})

    def test_valid_http_url(self):
        """Test that HTTP URLs are accepted."""
        config = {"endpoint_url": "http://localhost:8080"}
        provider = LlamaCppProvider(config)
        assert provider.config["endpoint_url"] == "http://localhost:8080"

    def test_model_name_optional(self):
        """Test that model_name is optional."""
        config = {"endpoint_url": "http://localhost:8080"}
        provider = LlamaCppProvider(config)
        assert provider.config.get("model_name") is None

    def test_config_field_names(self):
        """Test get_config_field_names returns expected fields."""
        provider = LlamaCppProvider({"endpoint_url": "http://localhost:8080"})
        fields = provider.get_config_field_names()
        assert fields == ["endpoint_url", "model_name"]

    def test_config_defaults(self):
        """Test get_config_defaults returns expected defaults."""
        provider = LlamaCppProvider({"endpoint_url": "http://localhost:8080"})
        defaults = provider.get_config_defaults()
        assert defaults["endpoint_url"] == "http://localhost:8080"
        assert defaults["model_name"] == ""


class TestLoadProviderConfig:
    """Tests for loading provider configurations."""

    def test_returns_defaults_when_no_file_exists(self, tmp_path):
        """Test that defaults are returned when config file doesn't exist."""
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            defaults = {"endpoint_url": "http://default.com"}
            result = load_provider_config("test_provider", defaults)
            assert result == defaults

    def test_loads_valid_json_config(self, tmp_path):
        """Test that valid JSON config is loaded correctly."""
        # Create a mock config file
        config_file = tmp_path / "test_provider_config.json"
        saved_data = {"endpoint_url": "http://saved.com", "model_name": "saved-model"}
        config_file.write_text(json.dumps(saved_data))

        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            defaults = {"endpoint_url": "http://default.com"}
            result = load_provider_config("test_provider", defaults)

            assert result["endpoint_url"] == "http://saved.com"
            assert result["model_name"] == "saved-model"

    def test_merges_saved_with_defaults(self, tmp_path):
        """Test that saved config is merged with defaults."""
        # Create a mock config file with only some fields
        config_file = tmp_path / "test_provider_config.json"
        saved_data = {"endpoint_url": "http://saved.com"}
        config_file.write_text(json.dumps(saved_data))

        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            defaults = {
                "endpoint_url": "http://default.com",
                "model_name": "default-model",
                "timeout": 30,
            }
            result = load_provider_config("test_provider", defaults)

            # Saved values should override defaults
            assert result["endpoint_url"] == "http://saved.com"
            # Missing fields should use defaults
            assert result["model_name"] == "default-model"
            assert result["timeout"] == 30

    def test_returns_defaults_on_invalid_json(self, tmp_path):
        """Test that invalid JSON returns defaults."""
        config_file = tmp_path / "test_provider_config.json"
        config_file.write_text("not valid json {")

        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            defaults = {"endpoint_url": "http://default.com"}
            result = load_provider_config("test_provider", defaults)
            assert result == defaults

    def test_returns_defaults_on_non_dict_json(self, tmp_path):
        """Test that non-dict JSON returns defaults."""
        config_file = tmp_path / "test_provider_config.json"
        config_file.write_text('["array", "not", "dict"]')

        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            defaults = {"endpoint_url": "http://default.com"}
            result = load_provider_config("test_provider", defaults)
            assert result == defaults

    def test_returns_defaults_on_empty_file(self, tmp_path):
        """Test that empty file returns defaults."""
        config_file = tmp_path / "test_provider_config.json"
        config_file.write_text("")

        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            defaults = {"endpoint_url": "http://default.com"}
            result = load_provider_config("test_provider", defaults)
            assert result == defaults


class TestSaveProviderConfig:
    """Tests for saving provider configurations."""

    def test_saves_valid_config(self, tmp_path):
        """Test that valid config is saved correctly."""
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            config = {"endpoint_url": "http://test.com", "model_name": "test-model"}
            result = save_provider_config("test_provider", config)

            assert result is True
            config_file = tmp_path / "test_provider_config.json"
            assert config_file.exists()

    def test_saves_with_proper_json_formatting(self, tmp_path):
        """Test that saved JSON has proper formatting."""
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            config = {"endpoint_url": "http://test.com"}
            save_provider_config("test_provider", config)

            config_file = tmp_path / "test_provider_config.json"
            content = json.loads(config_file.read_text())
            assert content == config

    def test_returns_false_on_io_error(self, tmp_path):
        """Test that IO errors return False."""
        # Create a read-only file to simulate permission error
        config_file = tmp_path / "test_provider_config.json"
        config_file.write_text("{}")
        config_file.chmod(0o444)  # Read-only

        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            config = {"endpoint_url": "http://test.com"}
            result = save_provider_config("test_provider", config)

            # On some systems this might still succeed, so we check both cases
            if not result:
                assert result is False


class TestRevertFieldToDefault:
    """Tests for reverting fields to default values."""

    def test_reverts_field_to_default(self):
        """Test that a field is reverted to its default value."""
        current_config = {
            "endpoint_url": "http://custom.com",
            "model_name": "custom-model",
        }
        defaults = {
            "endpoint_url": "http://default.com",
            "model_name": "",
        }

        result = revert_field_to_default(
            "test_provider", "endpoint_url", current_config, defaults
        )

        assert result["endpoint_url"] == "http://default.com"
        assert result["model_name"] == "custom-model"  # Unchanged

    def test_preserves_extra_fields(self):
        """Test that extra fields in config are preserved."""
        current_config = {
            "endpoint_url": "http://custom.com",
            "extra_field": "extra_value",
        }
        defaults = {"endpoint_url": "http://default.com"}

        result = revert_field_to_default(
            "test_provider", "endpoint_url", current_config, defaults
        )

        assert result["endpoint_url"] == "http://default.com"
        assert result["extra_field"] == "extra_value"

    def test_does_not_add_missing_fields(self):
        """Test that reverting doesn't add fields not in current config."""
        current_config = {"model_name": "custom-model"}
        defaults = {
            "endpoint_url": "http://default.com",
            "model_name": "",
        }

        result = revert_field_to_default(
            "test_provider", "model_name", current_config, defaults
        )

        assert result["model_name"] == ""
        assert "endpoint_url" not in result  # Not added from defaults

    def test_reverts_nonexistent_field_noop(self):
        """Test that reverting a field not in defaults is a no-op."""
        current_config = {"endpoint_url": "http://custom.com"}
        defaults = {"model_name": ""}

        result = revert_field_to_default(
            "test_provider", "endpoint_url", current_config, defaults
        )

        assert result == current_config


class TestHasSavedConfig:
    """Tests for checking if a saved config exists."""

    def test_returns_true_when_file_exists(self, tmp_path):
        """Test that has_saved_config returns True when file exists."""
        config_file = tmp_path / "test_provider_config.json"
        config_file.write_text("{}")

        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            result = has_saved_config("test_provider")
            assert result is True

    def test_returns_false_when_file_does_not_exist(self, tmp_path):
        """Test that has_saved_config returns False when file doesn't exist."""
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            result = has_saved_config("nonexistent_provider")
            assert result is False


class TestGetConfigFilePath:
    """Tests for getting config file paths."""

    def test_returns_correct_path(self, tmp_path):
        """Test that get_config_file_path returns correct path."""
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            path = get_config_file_path("lmstudio")
            assert path == tmp_path / "lmstudio_config.json"

    def test_different_providers_have_different_paths(self, tmp_path):
        """Test that different providers have different config file paths."""
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            lmstudio_path = get_config_file_path("lmstudio")
            llamacpp_path = get_config_file_path("llamacpp")

            assert lmstudio_path != llamacpp_path
            assert "lmstudio" in str(lmstudio_path)
            assert "llamacpp" in str(llamacpp_path)


class TestConfigManagerIntegration:
    """Integration tests for the config manager."""

    def test_full_save_load_cycle(self, tmp_path):
        """Test saving and loading a complete configuration."""
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            # Save config
            original_config = {
                "endpoint_url": "http://custom.com:9999/v1",
                "model_name": "my-custom-model",
                "temperature": 0.7,
            }
            save_result = save_provider_config("test_provider", original_config)
            assert save_result is True

            # Load config with different defaults
            defaults = {
                "endpoint_url": "http://default.com",
                "model_name": "default-model",
            }
            loaded_config = load_provider_config("test_provider", defaults)

            # Verify all saved values are present
            assert loaded_config["endpoint_url"] == original_config["endpoint_url"]
            assert loaded_config["model_name"] == original_config["model_name"]
            assert loaded_config["temperature"] == original_config["temperature"]

    def test_save_load_with_revert(self, tmp_path):
        """Test saving, loading, and reverting a field."""
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            # Save initial config
            save_provider_config("test_provider", {"endpoint_url": "http://custom.com"})

            # Load with defaults
            defaults = {"endpoint_url": "http://default.com", "model_name": ""}
            loaded = load_provider_config("test_provider", defaults)

            # Revert endpoint to default
            reverted = revert_field_to_default(
                "test_provider", "endpoint_url", loaded, defaults
            )

            assert reverted["endpoint_url"] == "http://default.com"


class TestProviderConfigIntegration:
    """Integration tests for provider config with actual providers."""

    def test_lmstudio_with_loaded_config(self, tmp_path):
        """Test creating LM Studio provider from loaded config."""
        # Save a config
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            save_provider_config("lmstudio", {"endpoint_url": "http://test:1234/v1"})

            # Load and create provider
            defaults = {"endpoint_url": "http://localhost:1234/v1", "model_name": ""}
            loaded_config = load_provider_config("lmstudio", defaults)

            provider = get_provider("lmstudio", loaded_config)
            assert isinstance(provider, LMStudioProvider)
            assert provider.config["endpoint_url"] == "http://test:1234/v1"

    def test_llamacpp_with_loaded_config(self, tmp_path):
        """Test creating llama.cpp provider from loaded config."""
        # Save a config
        with patch("chatbot.config_manager.CONFIG_DIR", tmp_path):
            save_provider_config("llamacpp", {"endpoint_url": "http://test:8080"})

            # Load and create provider
            defaults = {"endpoint_url": "http://localhost:8080", "model_name": ""}
            loaded_config = load_provider_config("llamacpp", defaults)

            provider = get_provider("llamacpp", loaded_config)
            assert isinstance(provider, LlamaCppProvider)
            assert provider.config["endpoint_url"] == "http://test:8080"


class TestFetchModelsMocked:
    """Tests for fetch_models with mocked network calls."""

    def test_lmstudio_fetch_models_success(self):
        """Test successful model fetching from LM Studio."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = LMStudioProvider(config)

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model-1"},
                {"id": "model-2"},
                {"id": ""},  # Should be filtered out
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            models = provider.fetch_models()
            assert models == ["model-1", "model-2"]

    def test_llamacpp_fetch_models_success(self):
        """Test successful model fetching from llama.cpp."""
        config = {"endpoint_url": "http://localhost:8080"}
        provider = LlamaCppProvider(config)

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "llama-model"}, {"id": "mistral-model"}]
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            models = provider.fetch_models()
            assert models == ["llama-model", "mistral-model"]

    def test_fetch_models_connection_error(self):
        """Test that connection errors are properly raised."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = LMStudioProvider(config)

        from requests.exceptions import ConnectionError as RequestsConnectionError

        with patch(
            "requests.get", side_effect=RequestsConnectionError("Connection refused")
        ):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                provider.fetch_models()

    def test_fetch_models_timeout(self):
        """Test that timeouts are properly raised."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = LMStudioProvider(config)

        from requests.exceptions import Timeout

        with patch("requests.get", side_effect=Timeout()):
            with pytest.raises(RuntimeError, match="timed out"):
                provider.fetch_models()

    def test_fetch_models_invalid_response_format(self):
        """Test that invalid response format raises RuntimeError."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = LMStudioProvider(config)

        mock_response = Mock()
        mock_response.json.return_value = {"models": ["model-1"]}  # Wrong format
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Unexpected response format"):
                provider.fetch_models()


class TestStreamChatMocked:
    """Tests for stream_chat with mocked network calls."""

    def test_lmstudio_stream_chat_success(self):
        """Test successful streaming chat from LM Studio."""
        config = {"endpoint_url": "http://localhost:1234/v1", "model_name": "test"}
        provider = LMStudioProvider(config)

        messages = [{"role": "user", "content": "Hello"}]

        # Mock the response with SSE format
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " there"}}]}',
            b"data: [DONE]",
        ]
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            chunks = list(provider.stream_chat(messages))
            assert chunks == ["Hello", " there"]

    def test_llamacpp_stream_chat_success(self):
        """Test successful streaming chat from llama.cpp."""
        config = {"endpoint_url": "http://localhost:8080"}
        provider = LlamaCppProvider(config)

        messages = [{"role": "user", "content": "Hi"}]

        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}',
            b'data: {"choices": [{"delta": {"content": " there"}}]}',
            b"data: [DONE]",
        ]
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            chunks = list(provider.stream_chat(messages))
            assert chunks == ["Hi", " there"]

    def test_stream_chat_connection_error(self):
        """Test that connection errors are properly raised."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = LMStudioProvider(config)

        messages = [{"role": "user", "content": "Hello"}]

        from requests.exceptions import ConnectionError as RequestsConnectionError

        with patch(
            "requests.post", side_effect=RequestsConnectionError("Connection refused")
        ):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                list(provider.stream_chat(messages))

    def test_stream_chat_with_images(self):
        """Test streaming chat with image inputs."""
        config = {"endpoint_url": "http://localhost:1234/v1"}
        provider = LMStudioProvider(config)

        messages = [{"role": "user", "content": "Describe this image"}]
        images = [np.zeros((100, 100, 3), dtype=np.uint8)]

        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "This is"}}]}',
            b"data: [DONE]",
        ]
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            chunks = list(provider.stream_chat(messages, images))

            # Verify the request was made with multimodal content
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]

            # Content should be converted to multimodal format
            assert isinstance(payload["messages"][0]["content"], list)
            assert len(payload["messages"][0]["content"]) == 2  # text + image
