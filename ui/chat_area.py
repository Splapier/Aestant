"""Main chat area layout and image handling.

This module builds the central chat interface including the chatbot display,
image annotation editors, message input, and action buttons. It also handles
image refresh event wiring which is local to this section.
"""

import gradio as gr
from dataclasses import dataclass

from chatbot.image_handler import scan_input_directory, load_images_from_directory
from chatbot.image_handler import load_image_as_numpy


@dataclass
class ChatAreaComponents:
    """Container for components created in the chat area.

    Attributes:
        chatbot_component: The chatbot display component.
        prompt_input: User message input textbox.
        send_button: Button to submit messages.
        clear_button: Button to clear chat history.
        image_editors_row: Row container for image editors (for visibility control).
        image_editor_1: First ImageEditor component.
        image_editor_2: Second ImageEditor component.
        image_status_display: Status message for image loading.
    """

    chatbot_component: gr.Chatbot
    prompt_input: gr.Textbox
    send_button: gr.Button
    clear_button: gr.Button
    image_editors_row: gr.Row
    image_editor_1: gr.ImageEditor
    image_editor_2: gr.ImageEditor
    image_status_display: gr.Markdown


def _load_initial_images() -> list:
    """Load initial images from input directory."""
    return load_images_from_directory(max_count=2)


def _show_image_row_if_images_exist() -> gr.update:
    """Show the image editors row if images are available in input/."""
    paths = scan_input_directory(max_count=2)
    if paths:
        return gr.update(visible=True)
    return gr.update(visible=False)


def _refresh_editor_values() -> tuple[gr.update, gr.update, list[str], str]:
    """Load images from input/ into ImageEditor components.

    Returns:
        Tuple of (editor1_update, editor2_update, paths_list, status_text).
    """
    paths = scan_input_directory(max_count=2)

    image_arrays = []
    for path in paths:
        arr = load_image_as_numpy(path)
        if arr is not None:
            image_arrays.append(arr)

    editor1_update = gr.update(
        value={
            "background": image_arrays[0] if len(image_arrays) >= 1 else None,
            "layers": [],
            "composite": image_arrays[0] if len(image_arrays) >= 1 else None,
        },
        visible=len(image_arrays) >= 1,
    )
    editor2_update = gr.update(
        value={
            "background": image_arrays[1] if len(image_arrays) >= 2 else None,
            "layers": [],
            "composite": image_arrays[1] if len(image_arrays) >= 2 else None,
        },
        visible=len(image_arrays) >= 2,
    )

    if image_arrays:
        status = f"✅ Loaded {len(image_arrays)} image(s) from input/ directory."
    else:
        status = "ℹ️ No images found in input/ directory."

    return editor1_update, editor2_update, paths, status


def create_chat_area(image_paths_state: gr.State) -> ChatAreaComponents:
    """Create the main chat interface area.

    Must be called within a gr.Blocks context. Builds the chatbot, image editors,
    prompt input, and buttons. Wires up image refresh events internally.

    Args:
        image_paths_state: Gradio state for tracking loaded image paths.

    Returns:
        ChatAreaComponents containing all created components.
    """
    with gr.Column(scale=3):
        chatbot_component = gr.Chatbot(label="Conversation", height=500)

        # Image annotation section
        gr.Markdown("## 🖼️ Image Annotation (Optional)")
        gr.Markdown(
            "Place images in the `input/` directory and draw rectangles to highlight regions of interest."
        )

        with gr.Row():
            image_refresh_button = gr.Button("🖼️ Refresh Images", variant="secondary")

        initial_images = _load_initial_images()

        if initial_images:
            image_status_display = gr.Markdown(
                value=f"✅ Loaded {len(initial_images)} image(s) from input/ directory.",
                label="Status",
            )
        else:
            image_status_display = gr.Markdown(
                value="ℹ️ No images loaded. Click 'Refresh Images' or place files in `input/` directory.",
                label="Status",
            )

        # Pre-create up to 2 ImageEditor components
        with gr.Row(visible=bool(initial_images)) as image_editors_row:
            editor1_value = {
                "background": initial_images[0] if len(initial_images) >= 1 else None,
                "layers": [],
                "composite": initial_images[0] if len(initial_images) >= 1 else None,
            }
            editor2_value = {
                "background": initial_images[1] if len(initial_images) >= 2 else None,
                "layers": [],
                "composite": initial_images[1] if len(initial_images) >= 2 else None,
            }

            image_editor_1 = gr.ImageEditor(
                label="Image 1 (Draw rectangles)",
                type="numpy",
                brush=gr.Brush(
                    default_color="#FF0000",
                    colors=["#FF0000", "#00FF00", "#0000FF"],
                ),
                interactive=True,
                value=editor1_value,
            )

            image_editor_2 = gr.ImageEditor(
                label="Image 2 (Draw rectangles)",
                type="numpy",
                brush=gr.Brush(
                    default_color="#FF0000",
                    colors=["#FF0000", "#00FF00", "#0000FF"],
                ),
                interactive=True,
                value=editor2_value,
            )

        # Input and buttons
        prompt_input = gr.Textbox(
            label="Your Message",
            lines=2,
            placeholder="Type your message here...",
            container=True,
        )

        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear Chat", variant="secondary")

    # Wire image refresh events (two-step: show row, then set values)
    # Bundling visibility + value in a single gr.update() causes ImageEditor to
    # get stuck in "processing" on the first click.
    image_refresh_button.click(
        fn=_show_image_row_if_images_exist,
        inputs=None,
        outputs=[image_editors_row],
    ).then(
        fn=_refresh_editor_values,
        inputs=None,
        outputs=[
            image_editor_1,
            image_editor_2,
            image_paths_state,
            image_status_display,
        ],
    )

    return ChatAreaComponents(
        chatbot_component=chatbot_component,
        prompt_input=prompt_input,
        send_button=send_button,
        clear_button=clear_button,
        image_editors_row=image_editors_row,
        image_editor_1=image_editor_1,
        image_editor_2=image_editor_2,
        image_status_display=image_status_display,
    )
