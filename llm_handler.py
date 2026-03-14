"""LLM handler for sending images and prompts to various LLM providers."""

import os
from typing import Optional


def get_openai_client():
    """Get the OpenAI client instance.
    
    Returns:
        OpenAI client configured with API key from environment variable
    """
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # For local models like Ollama, LM Studio
    
    if base_url:
        return OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    return OpenAI(api_key=api_key)


def get_anthropic_client():
    """Get the Anthropic client instance.
    
    Returns:
        Anthropic client configured with API key from environment variable
    """
    from anthropic import Anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key)


def get_google_client():
    """Get the Google GenAI client instance.
    
    Returns:
        Google GenAI client configured with API key from environment variable
    """
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


def send_to_openai(base64_image_a: Optional[str], base64_image_b: Optional[str], 
                   user_prompt: str, model: str = "gpt-4o") -> str:
    """Send images and prompt to OpenAI API (or compatible local models).
    
    Args:
        base64_image_a: Base64 encoded string of the first image
        base64_image_b: Base64 encoded string of the second image
        user_prompt: User's text prompt/question
        model: Model name to use (default: gpt-4o)
        
    Returns:
        str: The LLM's response to the images and prompt
    """
    client = get_openai_client()
    
    # Build content array with text prompt and images
    content = [{"type": "input_text", "text": user_prompt}]
    
    # Add Image A if provided
    if base64_image_a:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_image_a}"
        })
    
    # Add Image B if provided
    if base64_image_b:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_image_b}"
        })
    
    # Send request to OpenAI API
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )
    
    return response.output_text


def send_to_anthropic(base64_image_a: Optional[str], base64_image_b: Optional[str], 
                      user_prompt: str, model: str = "claude-sonnet-4-5-20250929") -> str:
    """Send images and prompt to Anthropic Claude API.
    
    Args:
        base64_image_a: Base64 encoded string of the first image
        base64_image_b: Base64 encoded string of the second image
        user_prompt: User's text prompt/question
        model: Model name to use (default: claude-sonnet-4-5-20250929)
        
    Returns:
        str: The LLM's response to the images and prompt
    """
    client = get_anthropic_client()
    
    # Build content array with text prompt and images
    content = []
    
    # Add Image A if provided
    if base64_image_a:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64_image_a
            }
        })
    
    # Add Image B if provided
    if base64_image_b:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64_image_b
            }
        })
    
    # Add text prompt
    content.append({
        "type": "text",
        "text": user_prompt
    })
    
    # Send request to Anthropic API
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": content
        }]
    )
    
    return message.content[0].text


def send_to_google(base64_image_a: Optional[str], base64_image_b: Optional[str], 
                   user_prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Send images and prompt to Google Gemini API.
    
    Args:
        base64_image_a: Base64 encoded string of the first image
        base64_image_b: Base64 encoded string of the second image
        user_prompt: User's text prompt/question
        model: Model name to use (default: gemini-2.5-flash)
        
    Returns:
        str: The LLM's response to the images and prompt
    """
    from google.genai import types
    
    client = get_google_client()
    
    # Build contents array with text prompt and images
    contents = []
    
    # Add Image A if provided
    if base64_image_a:
        import base64 as b64
        image_bytes = b64.b64decode(base64_image_a)
        contents.append(types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ))
    
    # Add Image B if provided
    if base64_image_b:
        import base64 as b64
        image_bytes = b64.b64decode(base64_image_b)
        contents.append(types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ))
    
    # Add text prompt
    contents.append(user_prompt)
    
    # Send request to Google Gemini API
    response = client.models.generate_content(
        model=model,
        contents=contents,
    )
    
    return response.text


def send_to_llm(provider: str, base64_image_a: Optional[str], base64_image_b: Optional[str], 
                user_prompt: str, model: str) -> str:
    """Send images and prompt to the specified LLM provider.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'google', or 'local')
        base64_image_a: Base64 encoded string of the first image
        base64_image_b: Base64 encoded string of the second image
        user_prompt: User's text prompt/question
        model: Model name to use
        
    Returns:
        str: The LLM's response to the images and prompt
    """
    if provider in ('openai', 'local'):
        return send_to_openai(base64_image_a, base64_image_b, user_prompt, model)
    elif provider == 'anthropic':
        return send_to_anthropic(base64_image_a, base64_image_b, user_prompt, model)
    elif provider == 'google':
        return send_to_google(base64_image_a, base64_image_b, user_prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported providers: openai, anthropic, google, local")
