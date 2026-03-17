# Modular LLM Chat Application

A modular Gradio-based chat interface supporting multiple local LLM providers with streaming responses. Built strictly adhering to **Gradio 6.9.0** syntax conventions.

## Features

- **Modular Provider Architecture**: Clean separation of provider implementations in the `providers/` directory
- **Persistent Sidebar**: User-friendly provider selection using Gradio's native `gr.Sidebar` component
- **Dynamic Configuration**: UI fields automatically update based on selected provider type using `@gr.render` decorator
- **Streaming Responses**: Real-time text generation display using generator functions with `yield`
- **Session Persistence**: State management using `gr.State` for maintaining conversation history and configuration
- **Robust Error Handling**: Comprehensive error handling for connection issues, configuration errors, and runtime exceptions

## Supported Providers

### Local Providers (Current Implementation)

1. **LM Studio** - Connect to a locally running LM Studio server
2. **llama.cpp** - Connect to a locally running llama.cpp server with OpenAI-compatible API

### Future Extensibility

The modular architecture supports easy addition of cloud providers:
- OpenAI ChatGPT
- Anthropic Claude
- Google Gemini

## Project Structure

```
Aestant/
├── app.py                          # Main Gradio application entry point
├── main.py                         # Entry script that launches the app
├── providers/                      # Provider modules directory
│   ├── __init__.py                # Package initialization and exports
│   ├── base.py                    # Abstract base class for all providers
│   ├── lmstudio_provider.py       # LM Studio provider implementation
│   └── llamacpp_provider.py       # llama.cpp provider implementation
├── chatbot.py                      # Legacy file (preserved)
├── pyproject.toml                  # Project dependencies
├── README.md                       # This documentation
└── plans/                          # Architecture documentation
    └── llm-chat-app-architecture.md
```

## Installation

### Prerequisites

- Python 3.12 or higher
- `uv` package manager (recommended) or pip

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   cd /home/santa/Documents/github-repos/Aestant
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

Run the application using one of these methods:

```bash
# Using uv (recommended)
uv run python main.py

# Or directly with Python
python main.py
```

The application will launch at `http://127.0.0.1:7860`

### Using LM Studio Provider

1. **Start LM Studio**:
   - Open LM Studio desktop application
   - Download and load a model in the "Local Server" tab
   - Start the local server (default port: 1234)

2. **Configure in the App**:
   - Select "lmstudio" from the provider dropdown in the sidebar
   - Set Endpoint URL to `http://localhost:1234/v1` (or your custom port)
   - Optionally specify a model name if you have multiple models loaded

3. **Start Chatting**:
   - Type your message in the text box
   - Click "Send" or press Enter
   - Watch the response stream in real-time

### Using llama.cpp Provider

1. **Start llama.cpp Server**:
   ```bash
   # Build llama.cpp first (if not already built)
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make
   
   # Start the server with your model
   ./server -m /path/to/your/model.gguf --port 8080
   ```

2. **Configure in the App**:
   - Select "llamacpp" from the provider dropdown in the sidebar
   - Set Endpoint URL to `http://localhost:8080` (or your custom port)
   - Optionally specify a model name if you have multiple models loaded

3. **Start Chatting**:
   - Type your message in the text box
   - Click "Send" or press Enter
   - Watch the response stream in real-time

## API Reference

### Provider Interface

All providers implement the `BaseProvider` interface defined in [`providers/base.py`](providers/base.py):

```python
from providers.base import BaseProvider

class BaseProvider(ABC):
    def __init__(self, config: dict) -> None:
        """Initialize provider with configuration."""
        pass
    
    @abstractmethod
    def stream_chat(self, messages: list[dict]) -> Any:
        """Generate streaming response for chat history.
        
        Args:
            messages: List of {"role": "user|assistant", "content": "..."} dicts
        
        Yields:
            str: Partial response content chunks
        """
        pass
    
    @abstractmethod
    def get_provider_type(self) -> str:
        """Return provider identifier string."""
        pass
```

### Creating Provider Instances

Use the factory function to create provider instances:

```python
from providers import get_provider

# Create LM Studio provider
lmstudio = get_provider("lmstudio", {
    "endpoint_url": "http://localhost:1234/v1",
    "model_name": ""  # Optional
})

# Create llama.cpp provider
llamacpp = get_provider("llamacpp", {
    "endpoint_url": "http://localhost:8080",
    "model_name": ""  # Optional
})

# Use the provider
messages = [{"role": "user", "content": "Hello, how are you?"}]
for chunk in lmstudio.stream_chat(messages):
    print(chunk, end="", flush=True)
```

### Available Providers

```python
from providers import get_available_providers

providers = get_available_providers()  # Returns: ['lmstudio', 'llamacpp']
```

## Configuration Options

### LM Studio Provider

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `endpoint_url` | str | Yes | `http://localhost:1234/v1` | URL of the LM Studio server with `/v1` suffix |
| `model_name` | str | No | `` (empty) | Name of the model to use (uses default if empty) |

### llama.cpp Provider

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `endpoint_url` | str | Yes | `http://localhost:8080` | Base URL of the llama.cpp server (without `/v1`) |
| `model_name` | str | No | `` (empty) | Name of the model to use (uses default if empty) |

## Error Handling

The application handles various error scenarios gracefully:

- **Connection Errors**: Displayed as "❌ Connection Error" with helpful troubleshooting tips
- **Configuration Errors**: Displayed as "⚠️ Configuration Error" when invalid settings are detected
- **Runtime Errors**: Displayed as "🔧 Runtime Error" for inference-related issues
- **Unexpected Errors**: Displayed as "💥 Unexpected Error" with error type and message

## Troubleshooting

### Connection Refused Errors

**Problem**: "Failed to connect to [provider] at [URL]"

**Solutions**:
1. Ensure the provider server is running
2. Verify the endpoint URL is correct (including port number)
3. Check if the server is bound to `localhost` or a specific interface
4. Verify no firewall is blocking the connection

### Model Not Found Errors

**Problem**: "Model not found" or similar errors from the provider

**Solutions**:
1. Ensure a model is loaded in your provider (LM Studio or llama.cpp)
2. Specify the correct model name in the configuration
3. Check that the model file exists and is properly formatted

### Streaming Not Working

**Problem**: Response appears all at once instead of streaming

**Solutions**:
1. Verify the provider supports streaming (both LM Studio and llama.cpp do)
2. Check your network connection stability
3. Ensure you're using recent versions of the provider software

## Development

### Adding a New Provider

To add support for a new LLM provider:

1. **Create a new module** in `providers/`:
   ```python
   # providers/new_provider.py
   from providers.base import BaseProvider
   
   class NewProvider(BaseProvider):
       def __init__(self, config: dict) -> None:
           super().__init__(config)
           self.provider_type = "newprovider"
       
       def _validate_config(self) -> None:
           # Validate configuration
           pass
       
       def stream_chat(self, messages: list[dict]) -> Any:
           # Implement streaming logic
           pass
       
       def get_provider_type(self) -> str:
           return self.provider_type
   ```

2. **Register the provider** in `providers/__init__.py`:
   ```python
   from providers.new_provider import NewProvider
   
   PROVIDER_REGISTRY = {
       "lmstudio": LMStudioProvider,
       "llamacpp": LlamaCppProvider,
       "newprovider": NewProvider,  # Add new provider
   }
   ```

3. **Update the UI** in `app.py` to handle the new provider's configuration fields

## Gradio Version Compatibility

This application is built specifically for **Gradio 6.9.0**. Key syntax conventions used:

- No `type` argument in `gr.Chatbot()` component
- `@gr.render()` decorator for dynamic UI rendering
- Native `gr.Sidebar()` component for persistent side panels
- Generator functions with `yield` for streaming responses
- Event chaining with `.then()` for sequential handling

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

---

**Note**: This application is designed for local development and testing. For production use, consider implementing additional security measures such as authentication, rate limiting, and input validation.
