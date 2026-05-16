"""Preference comparison tab for winners/losers selection.

This module provides a UI for:
- Selecting two images and choosing a winner/loser
- Drawing rectangles to highlight regions of attention
- Building winners and losers pools with entry-based tracking
- Entry-by-entry scoring (optional)
- Auto-advancing to next pair after comparison
- Running inference to find best candidate based on pools
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np

from chatbot.average_embeddings import get_pool_status
from chatbot.similarity_search import find_winner
from chatbot.tagging_engine import DATASET_DIR
from chatbot.image_modules.rectangle_tool import RectangleTool
from chatbot.image_modules.image_loading import load_image_as_numpy
from chatbot.embedding_engine import (
    add_entry_to_embedding,
    get_entry_embedding,
    get_all_entries,
    load_embedding_file,
)


_HTML_TEMPLATE = None
_CSS_TEMPLATE = None
_JS_TEMPLATE = None

_tool_config = RectangleTool()
_update_template = _tool_config.get_html_update()
_HTML_TEMPLATE = _update_template["html_template"]
_CSS_TEMPLATE = _update_template["css_template"]
_JS_TEMPLATE = _update_template["js_on_load"]


def scan_dataset_images():
    """Scan dataset directory for images with embeddings."""
    if not DATASET_DIR.exists():
        return []
    return sorted(
        [
            f.name.removesuffix(".embedding.json")
            for f in DATASET_DIR.glob("*.embedding.json")
        ]
    )


def load_image_for_display(image_name: str):
    """Load image for display by reading image_path from embedding file."""
    if not image_name:
        return None

    emb_file = DATASET_DIR / f"{image_name}.embedding.json"
    if not emb_file.exists():
        return None

    with open(emb_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = data.get("image_path")
    if image_path and Path(image_path).exists():
        return str(image_path)

    possible_ext = [".jpg", ".jpeg", ".png", ".webp", ".gif"]
    for ext in possible_ext:
        img_path = DATASET_DIR / f"{image_name}{ext}"
        if img_path.exists():
            return str(img_path)

    return None


def format_pool_status(status: dict) -> str:
    """Format pool status for display."""
    winners = status.get("winners", {})
    losers = status.get("losers", {})
    win_entries = winners.get("entries", [])
    lose_entries = losers.get("entries", [])

    def format_entry(entry: dict) -> str:
        image_name = entry.get("image_name", "unknown")
        entry_id = entry.get("entry_id")
        if entry_id:
            return f"{image_name}[{entry_id}]"
        return f"{image_name}[full]"

    win_formatted = (
        ", ".join(format_entry(e) for e in win_entries) if win_entries else "None"
    )
    lose_formatted = (
        ", ".join(format_entry(e) for e in lose_entries) if lose_entries else "None"
    )

    return f"""
### Pool Status
**Winners:** {winners.get("entry_count", 0)} entries
- {win_formatted}

**Losers:** {losers.get("entry_count", 0)} entries
- {lose_formatted}
"""


_last_comparison = {"winner": None, "loser": None}
_compared_images = set()


def _get_next_pair(current_images, exclude=None):
    """Get the next pair of images to compare, skipping already-compared ones."""
    if exclude is None:
        exclude = set()
    all_excluded = _compared_images | exclude

    available = [img for img in current_images if img not in all_excluded]

    if len(available) < 2:
        return None, None
    return available[0], available[1]


def _create_rectangle_tool_html(image_name: str | None, label: str) -> dict:
    """Create rectangle tool HTML component value for an image."""
    image_path = load_image_for_display(image_name) if image_name else None

    tool = RectangleTool(label=label, visible=image_path is not None)
    if image_path:
        img_array = load_image_as_numpy(image_path)
        if img_array is not None:
            tool.set_background(img_array)

    return gr.update(
        value=tool._get_value(),
        html_template=_HTML_TEMPLATE,
        css_template=_CSS_TEMPLATE,
        js_on_load=_JS_TEMPLATE,
        visible=image_path is not None,
    )


def _extract_rectangles(editor_value) -> list[dict]:
    """Extract rectangles from rectangle tool editor value."""
    if not editor_value or not isinstance(editor_value, dict):
        return []
    return editor_value.get("rects", [])


def _get_or_create_entry(image_name: str, rects: list[dict]) -> str | None:
    """Get or create an entry for the given rectangles on an image.

    Args:
        image_name: Name of image (without extension).
        rects: List of rectangle dicts. Empty/None means full image.

    Returns:
        Entry ID if rects provided, None if no rects (full image).
        Creates new entry if rects don't match existing.
    """
    if not rects:
        return None

    return add_entry_to_embedding(image_name, rects)


def _get_embedding_for_entry(
    image_name: str, entry_id: str | None
) -> list[float] | None:
    """Get embedding for a specific entry or full image.

    Args:
        image_name: Name of image.
        entry_id: Entry ID, or None for full image embedding.

    Returns:
        Image embedding list.
    """
    if entry_id is None:
        data = load_embedding_file(image_name)
        return data.get("image_embedding")

    return get_entry_embedding(image_name, entry_id)


def _format_entries_for_display(image_name: str) -> str:
    """Get formatted string of entries for an image."""
    entries = get_all_entries(image_name)
    if not entries:
        return "No entries"

    parts = ["full image (default)"]
    for entry in sorted(entries, key=lambda e: e.get("rect_index", 0)):
        parts.append(f"{entry['id']}")
    return ", ".join(parts)


def handle_a_wins(name_a: str, name_b: str, editor_value_a, editor_value_b):
    """Mark A as winner, B as loser. Auto-advance to next pair."""
    global _last_comparison, _compared_images

    if not name_a or not name_b:
        return ("Both images must be selected.", name_a, name_b, None, None)
    if name_a == name_b:
        return (
            "Cannot compare the same image with itself.",
            name_a,
            name_b,
            None,
            None,
        )

    rects_a = _extract_rectangles(editor_value_a)
    rects_b = _extract_rectangles(editor_value_b)

    emb_file_a = DATASET_DIR / f"{name_a}.embedding.json"
    emb_file_b = DATASET_DIR / f"{name_b}.embedding.json"

    if not emb_file_a.exists():
        return (f"Embedding file not found for {name_a}", name_a, name_b, None, None)
    if not emb_file_b.exists():
        return (f"Embedding file not found for {name_b}", name_a, name_b, None, None)

    with open(emb_file_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(emb_file_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    entry_id_a = _get_or_create_entry(name_a, rects_a)
    entry_id_b = _get_or_create_entry(name_b, rects_b)

    emb_a = _get_embedding_for_entry(name_a, entry_id_a)
    emb_b = _get_embedding_for_entry(name_b, entry_id_b)

    if emb_a is None or emb_b is None:
        return (
            "Failed to get embeddings for entries.",
            name_a,
            name_b,
            None,
            None,
        )

    from chatbot.average_embeddings import add_winner_entry, add_loser_entry

    add_winner_entry(name_a, entry_id_a, emb_a, data_a.get("tag_embedding"))
    add_loser_entry(name_b, entry_id_b, emb_b, data_b.get("tag_embedding"))

    _last_comparison = {"winner": name_a, "loser": name_b}
    _compared_images.add(name_a)
    _compared_images.add(name_b)

    all_images = scan_dataset_images()
    next_a, next_b = _get_next_pair(all_images)
    next_img_a = _create_rectangle_tool_html(next_a, "Image A")
    next_img_b = _create_rectangle_tool_html(next_b, "Image B")

    entry_info_a = f"({entry_id_a})" if entry_id_a else "(full image)"
    entry_info_b = f"({entry_id_b})" if entry_id_b else "(full image)"

    return (
        f"✅ {name_a}{entry_info_a} → winners, {name_b}{entry_info_b} → losers. Auto-advancing...",
        next_a,
        next_b,
        next_img_a,
        next_img_b,
    )


def handle_b_wins(name_a: str, name_b: str, editor_value_a, editor_value_b):
    """Mark B as winner, A as loser. Auto-advance to next pair."""
    global _last_comparison, _compared_images

    if not name_a or not name_b:
        return ("Both images must be selected.", name_a, name_b, None, None)
    if name_a == name_b:
        return (
            "Cannot compare the same image with itself.",
            name_a,
            name_b,
            None,
            None,
        )

    rects_a = _extract_rectangles(editor_value_a)
    rects_b = _extract_rectangles(editor_value_b)

    emb_file_a = DATASET_DIR / f"{name_a}.embedding.json"
    emb_file_b = DATASET_DIR / f"{name_b}.embedding.json"

    if not emb_file_a.exists():
        return (f"Embedding file not found for {name_a}", name_a, name_b, None, None)
    if not emb_file_b.exists():
        return (f"Embedding file not found for {name_b}", name_a, name_b, None, None)

    with open(emb_file_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(emb_file_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    entry_id_a = _get_or_create_entry(name_a, rects_a)
    entry_id_b = _get_or_create_entry(name_b, rects_b)

    emb_a = _get_embedding_for_entry(name_a, entry_id_a)
    emb_b = _get_embedding_for_entry(name_b, entry_id_b)

    if emb_a is None or emb_b is None:
        return (
            "Failed to get embeddings for entries.",
            name_a,
            name_b,
            None,
            None,
        )

    from chatbot.average_embeddings import add_winner_entry, add_loser_entry

    add_winner_entry(name_b, entry_id_b, emb_b, data_b.get("tag_embedding"))
    add_loser_entry(name_a, entry_id_a, emb_a, data_a.get("tag_embedding"))

    _last_comparison = {"winner": name_b, "loser": name_a}
    _compared_images.add(name_a)
    _compared_images.add(name_b)

    all_images = scan_dataset_images()
    next_a, next_b = _get_next_pair(all_images)
    next_img_a = _create_rectangle_tool_html(next_a, "Image A")
    next_img_b = _create_rectangle_tool_html(next_b, "Image B")

    entry_info_a = f"({entry_id_a})" if entry_id_a else "(full image)"
    entry_info_b = f"({entry_id_b})" if entry_id_b else "(full image)"

    return (
        f"✅ {name_b}{entry_info_b} → winners, {name_a}{entry_info_a} → losers. Auto-advancing...",
        next_a,
        next_b,
        next_img_a,
        next_img_b,
    )


def handle_swap(name_a: str, name_b: str, editor_value_a, editor_value_b):
    """Swap the last comparison result (winner becomes loser and vice versa)."""
    global _last_comparison

    if _last_comparison["winner"] is None:
        return ("No previous comparison to swap.", name_a, name_b, None, None)

    old_winner = _last_comparison["winner"]
    old_loser = _last_comparison["loser"]

    emb_file_winner = DATASET_DIR / f"{old_winner}.embedding.json"
    emb_file_loser = DATASET_DIR / f"{old_loser}.embedding.json"

    if not emb_file_winner.exists() or not emb_file_loser.exists():
        return ("Cannot swap: embedding files missing.", name_a, name_b, None, None)

    with open(emb_file_winner, "r", encoding="utf-8") as f:
        data_w = json.load(f)
    with open(emb_file_loser, "r", encoding="utf-8") as f:
        data_l = json.load(f)

    img_emb_w = data_w.get("image_embedding")
    img_emb_l = data_l.get("image_embedding")

    if img_emb_w is None or img_emb_l is None:
        return ("Cannot swap: embeddings missing.", name_a, name_b, None, None)

    from chatbot.average_embeddings import add_winner_entry, add_loser_entry

    add_loser_entry(old_winner, None, img_emb_w, data_w.get("tag_embedding"))
    add_winner_entry(old_loser, None, img_emb_l, data_l.get("tag_embedding"))

    _last_comparison = {"winner": old_loser, "loser": old_winner}

    img_a = _create_rectangle_tool_html(name_a, "Image A")
    img_b = _create_rectangle_tool_html(name_b, "Image B")

    return (
        f"🔄 Swapped: {old_loser} → winners, {old_winner} → losers",
        name_a,
        name_b,
        img_a,
        img_b,
    )


def handle_run_inference(use_entry_scoring: bool):
    """Run inference to find best candidate."""
    from chatbot.average_embeddings import load_state

    state = load_state()
    winners = state.get("winners", {})
    losers = state.get("losers", {})

    if winners.get("image_count", 0) == 0:
        return "No winners yet. Add some winners first.", None

    result = find_winner(
        DATASET_DIR, winners, losers, use_entry_scoring=use_entry_scoring
    )

    winner_name = result.get("winner", "Unknown")
    score = result.get("score", 0.0)

    winner_stem = winner_name.replace(".embedding.json", "")
    winner_img = load_image_for_display(winner_stem)

    scoring_type = "with entry scoring" if use_entry_scoring else "standard"
    return (
        f"🏆 Winner: {winner_name}\nScore: {score:.4f} ({scoring_type})",
        winner_img,
    )


def update_image_a(name_a: str):
    """Update rectangle tool A when dropdown changes."""
    return _create_rectangle_tool_html(name_a, "Image A")


def update_image_b(name_b: str):
    """Update rectangle tool B when dropdown changes."""
    return _create_rectangle_tool_html(name_b, "Image B")


def get_formatted_pool_status():
    """Get formatted pool status for display."""
    status = get_pool_status()
    return format_pool_status(status)


def create_comparison_tab():
    """Create the preference comparison tab UI."""

    with gr.Tab("Comparison"):
        gr.Markdown("## Preference Comparison & Inference")
        gr.Markdown(
            "Select two images, choose a winner, and build pools for inference. "
            "Draw rectangles on images to create region entries. "
            "After each comparison, the next pair is loaded automatically."
        )

        all_images = scan_dataset_images()
        default_a = all_images[0] if len(all_images) > 0 else None
        default_b = all_images[1] if len(all_images) > 1 else None

        init_tool_a = _create_rectangle_tool_html(default_a, "Image A")
        init_tool_b = _create_rectangle_tool_html(default_b, "Image B")

        pool_status = gr.Markdown(value=get_formatted_pool_status())

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Image A")
                image_viewer_a = gr.HTML(
                    value=init_tool_a["value"],
                    html_template=_HTML_TEMPLATE or "",
                    css_template=_CSS_TEMPLATE or "",
                    js_on_load=_JS_TEMPLATE or "",
                    label="Image A Viewer",
                    visible=init_tool_a["visible"],
                )
                choices_a = gr.Dropdown(
                    choices=all_images,
                    value=default_a,
                    label="Select Image A",
                    allow_custom_value=True,
                )
                if default_a:
                    entries_display_a = gr.Markdown(
                        value=f"**Entries:** {_format_entries_for_display(default_a)}"
                    )

            with gr.Column(scale=1):
                gr.Markdown("### Image B")
                image_viewer_b = gr.HTML(
                    value=init_tool_b["value"],
                    html_template=_HTML_TEMPLATE or "",
                    css_template=_CSS_TEMPLATE or "",
                    js_on_load=_JS_TEMPLATE or "",
                    label="Image B Viewer",
                    visible=init_tool_b["visible"],
                )
                choices_b = gr.Dropdown(
                    choices=all_images,
                    value=default_b,
                    label="Select Image B",
                    allow_custom_value=True,
                )
                if default_b:
                    entries_display_b = gr.Markdown(
                        value=f"**Entries:** {_format_entries_for_display(default_b)}"
                    )

        with gr.Row():
            a_wins_btn = gr.Button("👑 A Wins", variant="primary", scale=1)
            b_wins_btn = gr.Button("👑 B Wins", variant="primary", scale=1)
            swap_btn = gr.Button("🔄 Swap", variant="secondary", scale=1)

        action_status = gr.Markdown(value="")

        gr.Markdown("---")
        gr.Markdown("### Run Inference")
        with gr.Row():
            refresh_pools_btn = gr.Button("🔄 Refresh Pools", variant="secondary")
            entry_scoring_checkbox = gr.Checkbox(
                label="Use entry-by-entry scoring",
                value=False,
                info="Compare candidates against each individual entry (slower but more detailed)",
            )
            run_inference_btn = gr.Button("🏆 Find Winner", variant="primary")

        inference_status = gr.Markdown(value="")
        inference_result = gr.Image(
            type="filepath", label="Inference Winner", height=400
        )

        choices_a.change(
            fn=update_image_a,
            inputs=[choices_a],
            outputs=[image_viewer_a],
        )

        choices_b.change(
            fn=update_image_b,
            inputs=[choices_b],
            outputs=[image_viewer_b],
        )

        a_wins_btn.click(
            fn=handle_a_wins,
            inputs=[choices_a, choices_b, image_viewer_a, image_viewer_b],
            outputs=[
                action_status,
                choices_a,
                choices_b,
                image_viewer_a,
                image_viewer_b,
            ],
        ).then(fn=get_formatted_pool_status, outputs=[pool_status])

        b_wins_btn.click(
            fn=handle_b_wins,
            inputs=[choices_a, choices_b, image_viewer_a, image_viewer_b],
            outputs=[
                action_status,
                choices_a,
                choices_b,
                image_viewer_a,
                image_viewer_b,
            ],
        ).then(fn=get_formatted_pool_status, outputs=[pool_status])

        swap_btn.click(
            fn=handle_swap,
            inputs=[choices_a, choices_b, image_viewer_a, image_viewer_b],
            outputs=[
                action_status,
                choices_a,
                choices_b,
                image_viewer_a,
                image_viewer_b,
            ],
        ).then(fn=get_formatted_pool_status, outputs=[pool_status])

        refresh_pools_btn.click(fn=get_formatted_pool_status, outputs=[pool_status])

        run_inference_btn.click(
            fn=handle_run_inference,
            inputs=[entry_scoring_checkbox],
            outputs=[inference_status, inference_result],
        )

    return {
        "pool_status": pool_status,
        "choices_a": choices_a,
        "choices_b": choices_b,
        "image_viewer_a": image_viewer_a,
        "image_viewer_b": image_viewer_b,
    }


__all__ = ["create_comparison_tab"]
