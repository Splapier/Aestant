"""llama.cpp local LLM provider implementation.

This module provides integration with a locally running llama.cpp server,
supporting streaming chat completions via the built-in HTTP server API.
"""

import json as json_module
import requests
from typing import Any
import numpy as np

from chatbot.providers.base import BaseProvider, convert_messages_to_multimodal


class LlamaCppProvider(BaseProvider):
    """Provider for connecting to local llama.cpp server.

    The llama.cpp project includes a server mode that exposes an OpenAI-compatible
    API endpoint, allowing seamless integration with standard chat completion interfaces.

    Configuration:
        endpoint_url (str): The URL of the llama.cpp server (default: http://localhost:8080)
        model_name (str, optional): The name/identifier of the loaded model

    Attributes:
        config (dict): Configuration dictionary with endpoint_url and optionally model_name.
        provider_type (str): "llamacpp" identifier string.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the llama.cpp provider with configuration.

        Args:
            config: Dictionary containing 'endpoint_url' and optionally 'model_name'.

        Raises:
            ValueError: If endpoint_url is missing or invalid.
        """
        super().__init__(config)
        self.provider_type = "llamacpp"

    def _validate_config(self) -> None:
        """Validate the llama.cpp configuration parameters.

        Checks that endpoint_url is present and has a valid URL format.
        Model name is optional as llama.cpp may have a default model loaded.

        Raises:
            ValueError: If endpoint_url is missing or not a valid URL format.
        """
        if "endpoint_url" not in self.config or not self.config["endpoint_url"]:
            raise ValueError("llama.cpp provider requires 'endpoint_url' configuration")

        # Basic URL validation - check for http:// or https:// prefix
        endpoint = self.config["endpoint_url"]
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid endpoint URL format: '{endpoint}'. "
                "URL must start with 'http://' or 'https://'."
            )

    def stream_chat(
        self, 
        messages: list[dict[str, str]], 
        images: list[np.ndarray] | None = None
    ) -> Any:
        """Generate a streaming response from llama.cpp server.

        Sends the chat history to the llama.cpp server and yields partial
        responses as they are generated using Server-Sent Events (SSE).
        
        Supports multimodal inputs by converting images to base64 data URLs
        and embedding them in the messages payload.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Format: [{"role": "user", "content": "..."}, ...]
            images: Optional list of numpy arrays (RGB format) representing images
                   to include with the request for multimodal LLM support.

        Yields:
            str: Partial response content chunks as they are generated.

        Raises:
            ConnectionError: If unable to connect to the llama.cpp endpoint.
            requests.RequestException: For other HTTP-related errors.
            RuntimeError: If an unexpected error occurs during inference.
        """
        # Convert messages to multimodal format if images are provided
        processed_messages = convert_messages_to_multimodal(messages, images)
        
        # Build the API request URL
        endpoint_url = self.config["endpoint_url"]

        # llama.cpp server typically uses /v1/chat/completions for OpenAI-compatible mode
        api_url = f"{endpoint_url}/v1/chat/completions"

        # Prepare the request payload (OpenAI-compatible format)
        payload = {
            "model": self.config.get("model_name", ""),
            "messages": processed_messages,
            "stream": True,
        }

        # Set up headers for OpenAI-compatible API
        headers = {"Content-Type": "application/json"}

        try:
            # Make the streaming request to llama.cpp server
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
                f"Failed to connect to llama.cpp server at {endpoint_url}. "
                "Please ensure the llama.cpp server is running and the endpoint URL is correct."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "Request timed out. The model may be taking too long to generate a response."
            ) from e
        except requests.exceptions.RequestException as e:
            # Handle specific error cases like 404 (wrong endpoint path)
            if isinstance(e, requests.exceptions.HTTPError):
                if e.response is not None and e.response.status_code == 404:
                    raise RuntimeError(
                        f"Endpoint not found at {api_url}. "
                        "The llama.cpp server might be using a different API path. "
                        "Try adjusting the endpoint URL or ensure the server supports OpenAI-compatible API."
                    ) from e
            raise RuntimeError(f"Error communicating with llama.cpp server: {str(e)}") from e

    def get_provider_type(self) -> str:
        """Return the provider type identifier.

        Returns:
            str: "llamacpp" - the identifier for this provider.
        """
        return self.provider_type

    def get_config_field_names(self) -> list[str]:
        """Return configuration field names for llama.cpp UI rendering.

        Returns:
            List of configuration field names to display in the sidebar.
        """
        return ["endpoint_url", "model_name"]

    def get_config_defaults(self) -> dict[str, str]:
        """Return default configuration values for llama.cpp.

        Returns:
            Dictionary with default endpoint URL and empty model name.
        """
        return {"endpoint_url": "http://localhost:8080", "model_name": ""}

    def fetch_models(self) -> list[str]:
        """Fetch available models from llama.cpp /v1/models endpoint.

        Queries the OpenAI-compatible /v1/models endpoint to retrieve a list of
        available model identifiers that can be used for chat completions.

        Returns:
            List of model identifiers (e.g., ['local-model-1', 'model-2']).
            Empty list if no models are found or endpoint doesn't support listing.

        Raises:
            ConnectionError: If llama.cpp server is unreachable.
            RuntimeError: If response format is invalid or unexpected.
            requests.Timeout: If the request times out after 10 seconds.
        """
        endpoint_url = self.config["endpoint_url"]
        # llama.cpp typically uses /v1/models for OpenAI-compatible mode
        models_url = f"{endpoint_url}/v1/models"

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
                "Unexpected response format from /v1/models endpoint. "
                "Expected OpenAI-compatible format with 'data' array."
            )

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to llama.cpp server at {endpoint_url}. "
                "Please ensure the llama.cpp server is running and the endpoint URL is correct."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Request timed out after 10 seconds while fetching models from {endpoint_url}"
            ) from e
        except ValueError as e:
            # JSON decode error
            raise RuntimeError(
                "Invalid JSON response from llama.cpp /v1/models endpoint. "
                "The server may not be responding correctly."
            ) from e
