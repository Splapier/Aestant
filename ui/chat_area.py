"""Main chat area layout and image handling.

This module builds the central chat interface including the chatbot display,
image annotation editors, message input, and action buttons. It also handles
image refresh event wiring which is local to this section.

Image annotation uses custom HTML5 Canvas rectangle tools built on gr.HTML,
replacing the previous gr.ImageEditor with freeform brush.
"""

import gradio as gr
from dataclasses import dataclass

from chatbot.image_handler import scan_input_directory, load_images_from_directory
from chatbot.image_handler import load_image_as_numpy, RectangleTool


# Pre-compute HTML templates once (they don't change between updates)
_tool_config = RectangleTool()
_update_template = _tool_config.get_html_update()
_HTML_TEMPLATE = _update_template["html_template"]
_CSS_TEMPLATE = _update_template["css_template"]
_JS_TEMPLATE = _update_template["js_on_load"]


@dataclass
class ChatAreaComponents:
    """Container for components created in the chat area.

    Attributes:
        image_editors_row: Row container for image editors (for visibility control).
        image_editor_1: First HTML rectangle tool component.
        image_editor_2: Second HTML rectangle tool component.
        image_status_display: Status message for image loading.
        run_election_button: Button to run preference election.
        election_status: Status display for election progress.
    """

    image_editors_row: gr.Row
    image_editor_1: gr.HTML
    image_editor_2: gr.HTML
    image_status_display: gr.Markdown
    run_election_button: gr.Button
    election_status: gr.Textbox


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
    """Load images from input/ into rectangle tool components.

    Returns:
        Tuple of (editor1_update, editor2_update, paths_list, status_text).
    """
    paths = scan_input_directory(max_count=2)

    image_arrays = []
    for path in paths:
        arr = load_image_as_numpy(path)
        if arr is not None:
            image_arrays.append(arr)

    # Build value dicts
    tool1 = RectangleTool(label="Image 1 - Rectangle Tool")
    tool2 = RectangleTool(label="Image 2 - Rectangle Tool")

    if len(image_arrays) >= 1:
        tool1.set_background(image_arrays[0])
    if len(image_arrays) >= 2:
        tool2.set_background(image_arrays[1])

    editor1_update = gr.update(
        value=tool1._get_value(),
        html_template=_HTML_TEMPLATE,
        css_template=_CSS_TEMPLATE,
        js_on_load=_JS_TEMPLATE,
        visible=len(image_arrays) >= 1,
    )
    editor2_update = gr.update(
        value=tool2._get_value(),
        html_template=_HTML_TEMPLATE,
        css_template=_CSS_TEMPLATE,
        js_on_load=_JS_TEMPLATE,
        visible=len(image_arrays) >= 2,
    )

    if image_arrays:
        status = f"Loaded {len(image_arrays)} image(s) from input/ directory."
    else:
        status = "No images found in input/ directory."

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
        # Image annotation section
        gr.Markdown("## Image Annotation (Optional)")
        gr.Markdown(
            "Place images in the `input/` directory and draw rectangles to highlight regions of interest."
        )

        with gr.Row():
            image_refresh_button = gr.Button("Refresh Images", variant="secondary")

        initial_images = _load_initial_images()

        if initial_images:
            image_status_display = gr.Markdown(
                value=f"Loaded {len(initial_images)} image(s) from input/ directory.",
                label="Status",
            )
        else:
            image_status_display = gr.Markdown(
                value="No images loaded. Click 'Refresh Images' or place files in `input/` directory.",
                label="Status",
            )

        # Build initial values
        init_tool1 = RectangleTool(
            label="Image 1 - Rectangle Tool",
            visible=len(initial_images) >= 1,
        )
        init_tool2 = RectangleTool(
            label="Image 2 - Rectangle Tool",
            visible=len(initial_images) >= 2,
        )
        if len(initial_images) >= 1:
            init_tool1.set_background(initial_images[0])
        if len(initial_images) >= 2:
            init_tool2.set_background(initial_images[1])

        # Pre-create up to 2 HTML rectangle tool components
        with gr.Row(visible=bool(initial_images)) as image_editors_row:
            image_editor_1 = gr.HTML(
                value=init_tool1._get_value(),
                html_template=_HTML_TEMPLATE,
                css_template=_CSS_TEMPLATE,
                js_on_load=_JS_TEMPLATE,
                label="Image 1 - Rectangle Tool",
                elem_id="rect-tool-1",
                visible=len(initial_images) >= 1,
            )

            image_editor_2 = gr.HTML(
                value=init_tool2._get_value(),
                html_template=_HTML_TEMPLATE,
                css_template=_CSS_TEMPLATE,
                js_on_load=_JS_TEMPLATE,
                label="Image 2 - Rectangle Tool",
                elem_id="rect-tool-2",
                visible=len(initial_images) >= 2,
            )

        gr.Markdown("---")
        gr.Markdown("## Preference Election")

        with gr.Row():
            run_election_button = gr.Button(
                "Run Preference Election", variant="primary"
            )

        election_status = gr.Textbox(
            label="Election Status",
            lines=2,
            interactive=False,
            visible=False,
        )

    # Wire image refresh events (two-step: show row, then set values)
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
        image_editors_row=image_editors_row,
        image_editor_1=image_editor_1,
        image_editor_2=image_editor_2,
        image_status_display=image_status_display,
        run_election_button=run_election_button,
        election_status=election_status,
    )
