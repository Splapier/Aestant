"""Tagging tab UI component.

This module provides a dedicated tab for tagging single images:
- Shows one image at a time from input directory
- Rectangle drawing tool for bounding box selection
- Send to Tag button (VLM tagging)
- Previous/Next navigation (jumps to first untagged by default)
- Embedding controls for image and tag vectors
- Editable tag fields display
- Add key dropdown for unfilled schema keys
- Status and progress info
"""

import gradio as gr
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatbot.image_modules.rectangle_tool import RectangleTool
from chatbot.image_modules.image_loading import (
    scan_input_directory,
    load_image_as_numpy,
)
from chatbot.schema_manager import (
    load_master_schema,
    get_descriptions_from_yaml,
    flatten_schema_keys,
)
from chatbot.tagging_engine import (
    tag_image_concurrent,
    save_tagged_dataset,
    scan_untagged_images,
    scan_all_images,
    find_first_untagged_index,
)
from chatbot.embedding_engine import (
    embed_image,
    embed_tags,
    save_embeddings,
    batch_embed_images,
)
from chatbot.config_manager import load_provider_config
from chatbot import get_provider


# Pre-compute HTML templates once (they don't change between updates)
_tool_config = RectangleTool()
_update_template = _tool_config.get_html_update()
_HTML_TEMPLATE = _update_template["html_template"]
_CSS_TEMPLATE = _update_template["css_template"]
_JS_TEMPLATE = _update_template["js_on_load"]


@dataclass
class TaggingTabComponents:
    """Container for tagging tab components."""

    image_viewer: Any
    tag_button: Any
    prev_button: Any
    next_button: Any
    save_button: Any
    embed_dropdown: Any
    embed_button: Any
    status_display: Any
    fields_display: Any
    add_key_dropdown: Any
    add_key_value: Any
    add_key_button: Any
    image_index_state: Any
    image_path_state: Any
    tags_state: Any
    raw_responses_state: Any
    prompts_state: Any


def _scan_and_skip_tagged() -> list[str]:
    """Scan for untagged images."""
    return scan_untagged_images()


def _scan_all_and_load() -> tuple[list[str], int]:
    """Scan all images and find index of first untagged.

    Returns:
        Tuple of (all_paths, first_untagged_index)
    """
    all_paths = scan_all_images()
    if not all_paths:
        return [], 0
    idx = find_first_untagged_index(all_paths)
    return all_paths, idx


def _load_current_image_all(
    index: int,
) -> tuple[str, np.ndarray | None, gr.update, str, int]:
    """Load image at current index using all images list.

    Args:
        index: Current image index.

    Returns:
        Tuple of (image_path, image_array, editor_update, status, total_count).
    """
    all_paths = scan_all_images()

    if not all_paths:
        return (
            "",
            None,
            gr.update(visible=False),
            "No images to tag. Add images to input/ directory.",
            0,
        )

    if index < 0:
        index = len(all_paths) - 1
    if index >= len(all_paths):
        index = 0

    image_path = all_paths[index] if all_paths else ""
    image_array = load_image_as_numpy(image_path) if image_path else None

    tool = RectangleTool(label="Tagging Image", visible=image_array is not None)
    if image_array is not None:
        tool.set_background(image_array)

    editor_update = gr.update(
        value=tool._get_value(),
        html_template=_HTML_TEMPLATE,
        css_template=_CSS_TEMPLATE,
        js_on_load=_JS_TEMPLATE,
        visible=image_array is not None,
    )

    status = f"Image {index + 1} of {len(all_paths)}"

    return image_path, image_array, editor_update, status, len(all_paths)


def _render_tag_fields_html(
    tags: dict,
    schema: dict,
    descriptions: dict,
) -> str:
    """Render editable tag fields as HTML.

    Args:
        tags: Currently filled tags dictionary.
        schema: Master schema for reference.
        descriptions: Descriptions from YAML comments.

    Returns:
        HTML string for tag fields.
    """
    if not tags:
        return '<p class="no-tags">No tags yet. Click "Send to Tag" to analyze this image.</p>'

    css = """
    <style>
    .tag-fields-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        max-height: 400px;
        overflow-y: auto;
        padding: 12px;
        background: #1e1e1e;
        border: 1px solid #444;
        border-radius: 6px;
    }
    .tag-category {
        margin-bottom: 16px;
        border: 1px solid #333;
        border-radius: 4px;
        overflow: hidden;
    }
    .tag-category-header {
        background: #2a2a2a;
        padding: 8px 12px;
        font-weight: 600;
        color: #e0e0e0;
        border-bottom: 1px solid #333;
    }
    .tag-category-items {
        padding: 8px;
        background: #1a1a1a;
    }
    .tag-field-row {
        display: flex;
        align-items: flex-start;
        padding: 6px 8px;
        margin-bottom: 4px;
        background: #252525;
        border-radius: 4px;
    }
    .tag-field-row:hover {
        background: #2a2a2a;
    }
    .tag-field-key {
        min-width: 200px;
        color: #aaa;
        font-size: 12px;
        padding-top: 4px;
    }
    .tag-field-key .full-path {
        color: #4da6ff;
        font-weight: 600;
    }
    .tag-field-key .description {
        color: #666;
        font-size: 10px;
        display: block;
        margin-top: 2px;
    }
    .tag-field-input {
        flex: 1;
    }
    .tag-field-input input,
    .tag-field-input textarea {
        width: 100%;
        padding: 6px 8px;
        border: 1px solid #444;
        border-radius: 4px;
        background: #2a2a2a;
        color: #e0e0e0;
        font-size: 13px;
    }
    .tag-field-input input:focus,
    .tag-field-input textarea:focus {
        outline: none;
        border-color: #007bff;
    }
    .tag-field-input textarea {
        min-height: 60px;
        resize: vertical;
    }
    .no-tags {
        color: #888;
        text-align: center;
        padding: 40px;
    }
    </style>
    """

    def format_value(val) -> str:
        if val is None:
            return ""
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        if isinstance(val, bool):
            return str(val)
        return str(val)

    def parse_value(val_str: str, field_type: str):
        if not val_str.strip():
            return None
        if field_type == "boolean":
            return val_str.lower() in ("true", "1", "yes")
        if field_type == "array":
            items = [x.strip() for x in val_str.split(",") if x.strip()]
            return items if items else None
        return val_str

    html_parts = [css, '<div class="tag-fields-container">']

    def get_parent_key(full_path: str) -> str:
        """Get the parent key name for display (e.g., 'eyes' from 'head.eyes.shape')."""
        parts = full_path.split(".")
        if len(parts) >= 2:
            return parts[-2]
        return ""

    for category, category_data in sorted(tags.items()):
        if not category_data:
            continue

        html_parts.append(f'<div class="tag-category">')
        html_parts.append(f'<div class="tag-category-header">{category}</div>')
        html_parts.append('<div class="tag-category-items">')

        def traverse(obj, path_prefix, depth=0):
            for key, value in sorted(obj.items()):
                full_path = f"{path_prefix}.{key}" if path_prefix else key
                desc = descriptions.get(full_path, "")

                if isinstance(value, dict):
                    traverse(value, full_path, depth + 1)
                else:
                    if value is None or value == "" or value == [] or value == False:
                        continue

                    field_type = "string"
                    if full_path in schema:
                        field_type = schema[full_path].get("type", "string")

                    value_str = format_value(value)
                    parent_key = get_parent_key(full_path)

                    html_parts.append(f'''
                    <div class="tag-field-row" data-path="{full_path}" data-type="{field_type}">
                        <div class="tag-field-key">
                            <span class="full-path">{parent_key}.{key}</span>
                            {f'<span class="description">{desc}</span>' if desc else ""}
                        </div>
                        <div class="tag-field-input">
                            <input type="text" 
                                   class="tag-value-input" 
                                   data-path="{full_path}"
                                   data-type="{field_type}"
                                   value="{value_str.replace('"', "&quot;")}"
                                   placeholder="Enter value...">
                        </div>
                    </div>
                    ''')

        traverse(category_data, category)
        html_parts.append("</div></div>")

    html_parts.append("</div>")

    html_parts.append("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.tag-value-input').forEach(function(input) {
            input.addEventListener('change', function() {
                const path = this.dataset.path;
                const type = this.dataset.type;
                const value = this.value;
                
                const hiddenInput = document.getElementById('tag-update-input');
                if (hiddenInput) {
                    hiddenInput.value = JSON.stringify({path: path, type: type, value: value});
                    hiddenInput.dispatchEvent(new Event('change'));
                }
            });
        });
    });
    </script>
    """)

    return "\n".join(html_parts)


def _get_unfilled_schema_keys(tags: dict, schema: dict) -> list[str]:
    """Get schema keys that haven't been filled yet.

    Args:
        tags: Currently filled tags.
        schema: Master schema.

    Returns:
        List of unfilled key paths.
    """
    flat_schema = flatten_schema_keys(schema)
    flat_tags = flatten_schema_keys(tags) if tags else {}

    unfilled = []
    for path in flat_schema:
        if path not in flat_tags or not flat_tags[path]:
            unfilled.append(path)

    return unfilled


def _load_current_image(index: int) -> tuple[str, np.ndarray | None, gr.update, str]:
    """Load image at current index.

    Args:
        index: Current image index.

    Returns:
        Tuple of (image_path, image_array, editor_update, status).
    """
    paths = _scan_and_skip_tagged()

    if not paths:
        return (
            "",
            None,
            gr.update(visible=False),
            "No images to tag. Add images to input/ directory.",
        )

    if index < 0:
        index = 0
    if index >= len(paths):
        index = len(paths) - 1

    image_path = paths[index] if paths else ""
    image_array = load_image_as_numpy(image_path) if image_path else None

    tool = RectangleTool(label="Tagging Image", visible=image_array is not None)
    if image_array is not None:
        tool.set_background(image_array)

    editor_update = gr.update(
        value=tool._get_value(),
        html_template=_HTML_TEMPLATE,
        css_template=_CSS_TEMPLATE,
        js_on_load=_JS_TEMPLATE,
        visible=image_array is not None,
    )

    status = f"Image {index + 1} of {len(paths)}"

    return image_path, image_array, editor_update, status


def _handle_send_to_tag(
    image_path: str,
    editor_value: dict | None,
    provider: str,
    endpoint: str,
    model: str,
) -> tuple[gr.update, str, dict, dict, dict]:
    """Handle send to tag button click.

    Args:
        image_path: Current image path.
        editor_value: Rectangle tool value with rectangles.
        provider: Provider type.
        endpoint: Endpoint URL.
        model: Model name.

    Returns:
        Tuple of (fields_html, status, tags, raw_responses, prompts).
    """
    if not image_path:
        return gr.update(), "No image loaded", {}, {}, {}

    config = {
        "endpoint_url": endpoint.strip() if endpoint else "",
        "model_name": model.strip() if model else "",
    }

    image_array = load_image_as_numpy(image_path)
    if image_array is None:
        return gr.update(), "Failed to load image", {}, {}, {}

    rectangles = []
    if editor_value and isinstance(editor_value, dict):
        rectangles = editor_value.get("rects", [])

    try:
        result = tag_image_concurrent(image_array, rectangles, provider, config)

        schema = load_master_schema()
        descriptions = get_descriptions_from_yaml()

        fields_html = _render_tag_fields_html(result["tags"], schema, descriptions)

        status = "Tagging complete!"

        return (
            fields_html,
            status,
            result["tags"],
            result["raw_responses"],
            result["prompts"],
        )
    except Exception as e:
        return (
            gr.update(),
            f"Error: {str(e)[:200]}",
            {},
            {},
            {},
        )


def _handle_next(
    current_index: int,
    current_image_path: str,
    tags: dict,
    raw_responses: dict,
    prompts: dict,
) -> tuple[int, str, gr.update, str, dict, dict, dict]:
    """Handle next button click.

    Args:
        current_index: Current image index.
        current_image_path: Current image path.
        tags: Current tags (to save).
        raw_responses: Current raw responses.
        prompts: Current prompts.

    Returns:
        Tuple of (new_index, image_path, editor_update, status, empty_tags, empty_raw, empty_prompts).
    """
    if current_image_path and tags:
        try:
            save_tagged_dataset(current_image_path, tags, raw_responses, prompts)
        except Exception:
            pass

    next_index = current_index + 1
    new_image_path, new_array, editor_update, status, _total = _load_current_image_all(
        next_index
    )

    return (
        next_index,
        new_image_path,
        editor_update,
        status,
        {},
        {},
        {},
    )


def _handle_prev(
    current_index: int,
    current_image_path: str,
    tags: dict,
    raw_responses: dict,
    prompts: dict,
) -> tuple[int, str, gr.update, str, dict, dict, dict]:
    """Handle previous button click.

    Args:
        current_index: Current image index.
        current_image_path: Current image path.
        tags: Current tags (to save).
        raw_responses: Current raw responses.
        prompts: Current prompts.

    Returns:
        Tuple of (new_index, image_path, editor_update, status, empty_tags, empty_raw, empty_prompts).
    """
    if current_image_path and tags:
        try:
            save_tagged_dataset(current_image_path, tags, raw_responses, prompts)
        except Exception:
            pass

    prev_index = current_index - 1
    new_image_path, new_array, editor_update, status, _total = _load_current_image_all(
        prev_index
    )

    return (
        prev_index,
        new_image_path,
        editor_update,
        status,
        {},
        {},
        {},
    )


def _handle_save_current_tags(
    image_path: str,
    tags: dict,
    raw_responses: dict,
    prompts: dict,
) -> str:
    """Save current tags to dataset.

    Args:
        image_path: Current image path.
        tags: Current tags.
        raw_responses: Raw responses.
        prompts: Prompts.

    Returns:
        Status message.
    """
    if not image_path:
        return "No image to save"

    if not tags:
        return "No tags to save"

    try:
        save_tagged_dataset(image_path, tags, raw_responses, prompts)
        return "Tags saved to dataset/"
    except Exception as e:
        return f"Error saving: {str(e)[:100]}"


def _handle_create_embeddings(
    image_path: str,
    tags: dict,
    embed_type: str,
) -> str:
    """Handle embedding creation.

    Args:
        image_path: Current image path.
        tags: Current tags.
        embed_type: One of 'Embed Image', 'Embed Tags', 'Embed Both'.

    Returns:
        Status message.
    """
    if not image_path:
        return "No image loaded"

    needs_tags = embed_type in ("Embed Tags", "Embed Both")
    if needs_tags and not tags:
        return "No tags to embed. Tag the image first."

    try:
        image_emb = None
        tag_emb = None

        if embed_type in ("Embed Image", "Embed Both"):
            image_emb = embed_image(image_path)

        if needs_tags:
            tag_emb = embed_tags(tags)

        save_embeddings(image_path, image_emb, tag_emb)

        if embed_type == "Embed Image":
            return "Image embedding created"
        elif embed_type == "Embed Tags":
            return "Tag embedding created"
        else:
            return "Both embeddings created"
    except Exception as e:
        return f"Embedding error: {str(e)[:100]}"


def create_tagging_tab(
    provider_selector: gr.Radio,
    endpoint_state: gr.State,
    model_state: gr.State,
) -> "TaggingTabComponents":
    """Create the tagging tab UI.

    Args:
        provider_selector: Provider selection radio from sidebar.
        endpoint_state: Endpoint URL state.
        model_state: Model name state.

    Returns:
        TaggingTabComponents containing all created components.
    """
    initial_paths, initial_index = _scan_all_and_load()

    if initial_paths:
        initial_image_path = initial_paths[initial_index]
        initial_array = load_image_as_numpy(initial_image_path)
    else:
        initial_image_path = ""
        initial_array = None

    image_index_state = gr.State(initial_index)
    image_path_state = gr.State(initial_image_path)
    tags_state = gr.State({})
    raw_responses_state = gr.State({})
    prompts_state = gr.State({})

    with gr.Column():
        gr.Markdown("## Image Tagging")
        gr.Markdown(
            "Tag images for dataset creation. Place images in `input/` directory. "
            "Tagged images are saved to `dataset/` as YAML files."
        )

        with gr.Row():
            refresh_button = gr.Button("🔄 Refresh Images", variant="secondary")
            image_counter = gr.Markdown(
                value=f"Image 1 of {len(initial_paths)}"
                if initial_paths
                else "No images",
            )

        init_tool = RectangleTool(
            label="Tagging Image",
            visible=initial_array is not None,
        )
        if initial_array is not None:
            init_tool.set_background(initial_array)

        image_viewer = gr.HTML(
            value=init_tool._get_value(),
            html_template=_HTML_TEMPLATE,
            css_template=_CSS_TEMPLATE,
            js_on_load=_JS_TEMPLATE,
            label="Image Viewer",
            visible=initial_array is not None,
        )

        with gr.Row():
            prev_button = gr.Button("⬅️ Previous", variant="secondary", scale=1)
            next_button = gr.Button("➡️ Next", variant="secondary", scale=1)

        with gr.Row():
            tag_button = gr.Button("🏷️ Send to Tag", variant="primary")
            save_button = gr.Button("💾 Save Tags", variant="secondary")

        with gr.Row():
            embed_dropdown = gr.Dropdown(
                choices=["Embed Image", "Embed Tags", "Embed Both"],
                value="Embed Both",
                label="Embeddings",
            )
            embed_button = gr.Button("🔢 Create Embeddings", variant="secondary")
            batch_embed_button = gr.Button(
                "🔄 Batch Embed All Images", variant="secondary"
            )

        status_display = gr.Markdown(
            value="Ready to tag" if initial_paths else "No images found",
        )

        batch_embed_status = gr.Markdown(value="")

        fields_display = gr.HTML(
            value='<p class="no-tags">No tags yet. Click "Send to Tag" to analyze this image.</p>',
            label="Tag Fields",
        )

        gr.Markdown("### Add Tag Key")

        with gr.Row():
            search_key_input = gr.Textbox(
                label="Search keys",
                placeholder="Type to search...",
                scale=1,
            )
            add_key_dropdown = gr.Dropdown(
                choices=[],
                label="Select key to add",
                scale=2,
            )
            add_key_value = gr.Textbox(
                label="Value",
                placeholder="Enter value...",
                scale=2,
            )
            add_key_button = gr.Button("➕ Add", variant="primary", scale=1)

        add_status = gr.Markdown(value="")

        hidden_update_input = gr.Textbox(
            elem_id="tag-update-input",
            visible=False,
        )

    def handle_refresh():
        paths = _scan_and_skip_tagged()
        index = 0
        if paths:
            img_path = paths[index]
            img_arr = load_image_as_numpy(img_path)
        else:
            img_path = ""
            img_arr = None

        tool = RectangleTool(label="Tagging Image", visible=img_arr is not None)
        if img_arr is not None:
            tool.set_background(img_arr)

        editor_update = gr.update(
            value=tool._get_value(),
            html_template=_HTML_TEMPLATE,
            css_template=_CSS_TEMPLATE,
            js_on_load=_JS_TEMPLATE,
            visible=img_arr is not None,
        )

        status = f"Image 1 of {len(paths)}" if paths else "No images"

        schema = load_master_schema()
        descriptions = get_descriptions_from_yaml()
        unfilled = _get_unfilled_schema_keys({}, schema)

        return (
            index,
            img_path,
            editor_update,
            status,
            gr.update(choices=unfilled),
            "",
        )

    refresh_button.click(
        fn=handle_refresh,
        inputs=[],
        outputs=[
            image_index_state,
            image_path_state,
            image_viewer,
            image_counter,
            add_key_dropdown,
            status_display,
        ],
    )

    def handle_send_to_tag_fn(
        img_path,
        editor_val,
        provider,
        endpoint,
        model,
    ):
        return _handle_send_to_tag(img_path, editor_val, provider, endpoint, model)

    tag_button.click(
        fn=handle_send_to_tag_fn,
        inputs=[
            image_path_state,
            image_viewer,
            provider_selector,
            endpoint_state,
            model_state,
        ],
        outputs=[
            fields_display,
            status_display,
            tags_state,
            raw_responses_state,
            prompts_state,
        ],
    )

    def handle_next(idx, img_path, tags, raw_resp, prompts):
        return _handle_next(idx, img_path, tags, raw_resp, prompts)

    next_button.click(
        fn=handle_next,
        inputs=[
            image_index_state,
            image_path_state,
            tags_state,
            raw_responses_state,
            prompts_state,
        ],
        outputs=[
            image_index_state,
            image_path_state,
            image_viewer,
            image_counter,
            tags_state,
            raw_responses_state,
            prompts_state,
        ],
    ).then(
        fn=lambda: '<p class="no-tags">No tags yet. Click "Send to Tag" to analyze this image.</p>',
        outputs=[fields_display],
    )

    def handle_prev(idx, img_path, tags, raw_resp, prompts):
        return _handle_prev(idx, img_path, tags, raw_resp, prompts)

    prev_button.click(
        fn=handle_prev,
        inputs=[
            image_index_state,
            image_path_state,
            tags_state,
            raw_responses_state,
            prompts_state,
        ],
        outputs=[
            image_index_state,
            image_path_state,
            image_viewer,
            image_counter,
            tags_state,
            raw_responses_state,
            prompts_state,
        ],
    ).then(
        fn=lambda: '<p class="no-tags">No tags yet. Click "Send to Tag" to analyze this image.</p>',
        outputs=[fields_display],
    )

    def handle_embed(img_path, tags, embed_type):
        return _handle_create_embeddings(img_path, tags, embed_type)

    embed_button.click(
        fn=handle_embed,
        inputs=[image_path_state, tags_state, embed_dropdown],
        outputs=[status_display],
    )

    def handle_batch_embed():
        """Handle batch embedding of all images in input directory."""
        try:
            count = batch_embed_images()
            return f"✅ Batch embed complete: {count} new image(s) embedded"
        except Exception as e:
            return f"❌ Batch embed error: {str(e)[:200]}"

    batch_embed_button.click(
        fn=handle_batch_embed,
        inputs=[],
        outputs=[batch_embed_status],
    )

    save_button.click(
        fn=_handle_save_current_tags,
        inputs=[image_path_state, tags_state, raw_responses_state, prompts_state],
        outputs=[status_display],
    )

    def handle_add_key(
        key_path, value, current_tags, current_raw, current_prompts, current_img_path
    ):
        """Add a new key to current tags."""
        if not key_path or not value:
            return (
                current_tags,
                current_raw,
                current_prompts,
                "Please select a key and enter a value",
            )

        parts = key_path.split(".")

        new_tags = {k: v for k, v in current_tags.items()}
        current = new_tags

        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        field_type = "string"
        schema = load_master_schema()
        if key_path in schema:
            field_type = schema[key_path].get("type", "string")

        if field_type == "array":
            current[parts[-1]] = [x.strip() for x in value.split(",") if x.strip()]
        elif field_type == "boolean":
            current[parts[-1]] = value.lower() in ("true", "1", "yes")
        else:
            current[parts[-1]] = value

        descriptions = get_descriptions_from_yaml()
        fields_html = _render_tag_fields_html(new_tags, schema, descriptions)

        return new_tags, current_raw, current_prompts, fields_html

    add_key_button.click(
        fn=handle_add_key,
        inputs=[
            add_key_dropdown,
            add_key_value,
            tags_state,
            raw_responses_state,
            prompts_state,
            image_path_state,
        ],
        outputs=[tags_state, raw_responses_state, prompts_state, fields_display],
    )

    add_key_button.click(
        fn=lambda: "",
        outputs=[add_key_value],
    )

    def on_tags_changed(new_tags):
        """Update dropdown when tags change."""
        schema = load_master_schema()
        unfilled = _get_unfilled_schema_keys(new_tags, schema)
        return gr.update(choices=unfilled)

    tags_state.change(
        fn=on_tags_changed,
        inputs=[tags_state],
        outputs=[add_key_dropdown],
    )

    def handle_search_keys(search_text: str):
        """Filter schema keys based on search text."""
        if not search_text:
            schema = load_master_schema()
            unfilled = _get_unfilled_schema_keys({}, schema)
            return gr.update(choices=unfilled, value="")

        search_lower = search_text.lower()
        schema = load_master_schema()
        descriptions = get_descriptions_from_yaml()
        flat_schema = flatten_schema_keys(schema)

        filtered = []
        for path in flat_schema:
            path_lower = path.lower()
            desc = descriptions.get(path, "").lower()
            if search_lower in path_lower or search_lower in desc:
                filtered.append(path)

        filtered = sorted(filtered)
        return gr.update(choices=filtered, value=search_text)

    search_key_input.change(
        fn=handle_search_keys,
        inputs=[search_key_input],
        outputs=[add_key_dropdown],
    )

    class TaggingTabComponents:
        def __init__(self):
            self.image_viewer = image_viewer
            self.tag_button = tag_button
            self.prev_button = prev_button
            self.next_button = next_button
            self.save_button = save_button
            self.embed_dropdown = embed_dropdown
            self.embed_button = embed_button
            self.status_display = status_display
            self.fields_display = fields_display
            self.add_key_dropdown = add_key_dropdown
            self.add_key_value = add_key_value
            self.add_key_button = add_key_button
            self.image_index_state = image_index_state
            self.image_path_state = image_path_state
            self.tags_state = tags_state
            self.raw_responses_state = raw_responses_state
            self.prompts_state = prompts_state

    return TaggingTabComponents()


__all__ = ["create_tagging_tab", "TaggingTabState"]
