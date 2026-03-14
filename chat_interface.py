"""Chat interface component for Streamlit application."""

import streamlit as st

from llm_handler import send_to_llm


def render_chat_interface(base64_image_a=None, base64_image_b=None,
                          provider="local", model="llava"):
    """Render the chat interface section of the app.
    
    Initializes chat history in session state if not present,
    displays existing messages, and provides input for new messages.
    
    Args:
        base64_image_a (str, optional): Base64 encoded string of Image A
        base64_image_b (str, optional): Base64 encoded string of Image B
        provider (str): LLM provider name ('openai', 'anthropic', 'google', or 'local')
        model (str): Model name to use for the selected provider
    """
    st.header("Chat")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type a message..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response using LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(f"Thinking... (using {provider}/{model})")
            
            try:
                response = send_to_llm(provider, base64_image_a, base64_image_b, prompt, model)
                message_placeholder.markdown(response)
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.markdown(error_message)
                response = error_message
            
            st.session_state.messages.append({"role": "assistant", "content": response})
