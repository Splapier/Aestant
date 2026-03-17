"""LM Studio local LLM provider implementation.

This module provides integration with a locally running LM Studio server,
supporting streaming chat completions via the OpenAI-compatible API endpoint.
"""

import json as json_module
import requests
from typing import Any

from chatbot.providers.base import BaseProvider


class LMStudioProvider(BaseProvider):
    """Provider for connecting to local LM Studio server.

    LM Studio exposes an OpenAI-compatible API when running its local server,
    allowing seamless integration with standard chat completion interfaces.

    Configuration:
        endpoint_url (str): The URL of the LM Studio server (default: http://localhost:1234/v1)
        model_name (str): The name/identifier of the loaded model in LM Studio

    Attributes:
        config (dict): Configuration dictionary with endpoint_url and model_name.
        provider_type (str): "lmstudio" identifier string.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the LM Studio provider with configuration.

        Args:
            config: Dictionary containing 'endpoint_url' and optionally 'model_name'.

        Raises:
            ValueError: If endpoint_url is missing or invalid.
        """
        super().__init__(config)
        self.provider_type = "lmstudio"

    def _validate_config(self) -> None:
        """Validate the LM Studio configuration parameters.

        Checks that endpoint_url is present and has a valid URL format.
        Model name is optional as LM Studio may have a default model loaded.

        Raises:
            ValueError: If endpoint_url is missing or not a valid URL format.
        """
        if "endpoint_url" not in self.config or not self.config["endpoint_url"]:
            raise ValueError("LM Studio provider requires 'endpoint_url' configuration")

        # Basic URL validation - check for http:// or https:// prefix
        endpoint = self.config["endpoint_url"]
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid endpoint URL format: '{endpoint}'. "
                "URL must start with 'http://' or 'https://'."
            )

    def stream_chat(self, messages: list[dict[str, str]]) -> Any:
        """Generate a streaming response from LM Studio.

        Sends the chat history to the LM Studio server and yields partial
        responses as they are generated using Server-Sent Events (SSE).

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Format: [{"role": "user", "content": "..."}, ...]

        Yields:
            str: Partial response content chunks as they are generated.

        Raises:
            ConnectionError: If unable to connect to the LM Studio endpoint.
            requests.RequestException: For other HTTP-related errors.
            RuntimeError: If an unexpected error occurs during inference.
        """
        # Build the API request URL
        endpoint_url = self.config["endpoint_url"]
        api_url = f"{endpoint_url}/chat/completions"

        # Prepare the request payload
        payload = {
            "model": self.config.get("model_name", ""),
            "messages": messages,
            "stream": True,
        }

        # Set up headers for OpenAI-compatible API
        headers = {"Content-Type": "application/json"}

        try:
            # Make the streaming request to LM Studio
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=120,  # 2-minute timeout for long generations
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Process the streaming response (SSE format)
            full_content = ""
            for line in response.iter_lines():
                if line:
                    # Decode the line and check for SSE data prefix
                    decoded_line = line.decode("utf-8")

                    # Skip non-data lines and empty messages
                    if not decoded_line.startswith("data:"):
                        continue

                    # Extract the JSON payload from SSE format
                    data_content = decoded_line[5:].strip()
                    if data_content == "[DONE]":
                        break

                    try:
                        data = json_module.loads(data_content)

                        # Extract content from the response structure
                        # OpenAI-compatible format: choices[0].delta.content
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                full_content += content
                                yield content

                    except json_module.JSONDecodeError as e:
                        # Skip malformed JSON lines
                        continue

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to LM Studio at {endpoint_url}. "
                "Please ensure LM Studio is running and the endpoint URL is correct."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "Request timed out. The model may be taking too long to generate a response."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with LM Studio: {str(e)}") from e

    def get_provider_type(self) -> str:
        """Return the provider type identifier.

        Returns:
            str: "lmstudio" - the identifier for this provider.
        """
        return self.provider_type

    def get_config_field_names(self) -> list[str]:
        """Return configuration field names for LM Studio UI rendering.

        Returns:
            List of configuration field names to display in the sidebar.
        """
        return ["endpoint_url", "model_name"]

    def get_config_defaults(self) -> dict[str, str]:
        """Return default configuration values for LM Studio.

        Returns:
            Dictionary with default endpoint URL and empty model name.
        """
        return {"endpoint_url": "http://localhost:1234/v1", "model_name": ""}

    def fetch_models(self) -> list[str]:
        """Fetch available models from LM Studio /models endpoint.

        Queries the OpenAI-compatible /models endpoint to retrieve a list of
        available model identifiers that can be used for chat completions.

        Returns:
            List of model identifiers (e.g., ['local-model-1', 'model-2']).
            Empty list if no models are found or endpoint doesn't support listing.

        Raises:
            ConnectionError: If LM Studio server is unreachable.
            RuntimeError: If response format is invalid or unexpected.
            requests.Timeout: If the request times out after 10 seconds.
        """
        endpoint_url = self.config["endpoint_url"]
        models_url = f"{endpoint_url}/models"

        try:
            response = requests.get(
                models_url,
                timeout=10,  # 10-second timeout for model listing
            )
            response.raise_for_status()

            data = response.json()
            # OpenAI-compatible format: {"data": [{"id": "model-name"}, ...]}
            if "data" in data and isinstance(data["data"], list):
                return [
                    model.get("id", "")
                    for model in data["data"]
                    if model.get("id")
                ]

            raise RuntimeError(
                "Unexpected response format from /models endpoint. "
                "Expected OpenAI-compatible format with 'data' array."
            )

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to LM Studio at {endpoint_url}. "
                "Please ensure LM Studio server is running and the endpoint URL is correct."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Request timed out after 10 seconds while fetching models from {endpoint_url}"
            ) from e
        except ValueError as e:
            # JSON decode error
            raise RuntimeError(
                "Invalid JSON response from LM Studio /models endpoint. "
                "The server may not be responding correctly."
            ) from e
