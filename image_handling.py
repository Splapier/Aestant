"""Image handling module for Gradio application."""

import base64
from typing import Optional, Tuple


def pil_to_base64(image) -> Optional[str]:
    """Convert a PIL Image to base64 encoded string.
    
    Args:
        image: PIL Image object or None
        
    Returns:
        Base64 encoded string of the image, or None if image is None
    """
    if image is not None:
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    return None


def get_image_data(image_a: Optional[object] = None, 
                   image_b: Optional[object] = None) -> Tuple[Optional[str], Optional[str]]:
    """Convert uploaded images to base64 encoded strings.
    
    Args:
        image_a: Uploaded Image A (PIL Image or file path)
        image_b: Uploaded Image B (PIL Image or file path)
        
    Returns:
        tuple: (base64_a, base64_b) - Base64 encoded strings of uploaded images
    """
    from PIL import Image
    
    # Handle image A
    if image_a is not None:
        if isinstance(image_a, str):
            # If it's a file path, load the image
            pil_image = Image.open(image_a)
        else:
            # Assume it's already a PIL Image
            pil_image = image_a
        base64_a = pil_to_base64(pil_image)
    else:
        base64_a = None
    
    # Handle image B
    if image_b is not None:
        if isinstance(image_b, str):
            # If it's a file path, load the image
            pil_image = Image.open(image_b)
        else:
            # Assume it's already a PIL Image
            pil_image = image_b
        base64_b = pil_to_base64(pil_image)
    else:
        base64_b = None
    
    return base64_a, base64_b
