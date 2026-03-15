"""Chat interface module for Gradio application."""

from typing import List, Tuple, Optional, Any
from llm_handler import send_to_llm


def process_chat_message(
    message: str,
    history: List[Tuple[str, str]],
    base64_image_a: Optional[str],
    base64_image_b: Optional[str],
    provider: str,
    model: str
) -> str:
    """Process a chat message and return the LLM response.
    
    Args:
        message: User's input message
        history: Chat history as list of (user_message, assistant_response) tuples
        base64_image_a: Base64 encoded string of Image A
        base64_image_b: Base64 encoded string of Image B
        provider: LLM provider name ('openai', 'anthropic', 'google', or 'local')
        model: Model name to use for the selected provider
        
    Returns:
        str: The LLM's response to the message and images
    """
    try:
        response = send_to_llm(provider, base64_image_a, base64_image_b, message, model)
        return response
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message
