"""Image handling component for Streamlit application."""

import base64
import streamlit as st


def file_to_base64(file):
    """Convert an uploaded file to base64 encoded string.
    
    Args:
        file: Uploaded file object from st.file_uploader
        
    Returns:
        Base64 encoded string of the file content, or None if file is None
    """
    if file is not None:
        return base64.b64encode(file.read()).decode("utf-8")
    return None


def render_image_uploaders():
    """Render side-by-side image uploaders.
    
    Creates two columns with file uploaders for Image A and Image B,
    displaying uploaded images when available.
    
    Returns:
        tuple: (base64_a, base64_b) - Base64 encoded strings of uploaded images
    """
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Image A")
        uploaded_file_a = st.file_uploader(
            "Upload Image A",
            type=["png", "jpg", "jpeg"],
            key="upload_a"
        )
        if uploaded_file_a is not None:
            st.image(uploaded_file_a, caption="Image A", use_container_width=True)
    
    with col2:
        st.header("Image B")
        uploaded_file_b = st.file_uploader(
            "Upload Image B",
            type=["png", "jpg", "jpeg"],
            key="upload_b"
        )
        if uploaded_file_b is not None:
            st.image(uploaded_file_b, caption="Image B", use_container_width=True)
    
    # Convert files to base64 and store in session state
    base64_a = file_to_base64(uploaded_file_a) if uploaded_file_a else None
    base64_b = file_to_base64(uploaded_file_b) if uploaded_file_b else None
    
    return base64_a, base64_b
