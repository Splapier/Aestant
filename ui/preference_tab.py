"""Preference comparison tab for winners/losers selection.

This module provides a UI for:
- Selecting two images and choosing a winner/loser
- Building winners and losers embedding pools
- Running inference to find best candidate based on pools
"""

import json

import gradio as gr
from pathlib import Path

from chatbot.average_embeddings import load_state, add_winner, add_loser
from chatbot.similarity_search import find_winner
from chatbot.embedding_engine import embed_image, embed_tags, save_embeddings
from chatbot.tagging_engine import DATASET_DIR


def scan_dataset_images():
    """Scan dataset directory for images with embeddings."""
    if not DATASET_DIR.exists():
        return []
    return sorted([f.stem for f in DATASET_DIR.glob("*.embedding.json")])


def load_image_for_comparison(image_name: str):
    """Load image from dataset for display."""
    if not image_name:
        return None
    # Find original image in input/ or dataset/
    input_dir = Path(__file__).resolve().parent.parent.parent / "input"
    possible_ext = [".jpg", ".jpeg", ".png", ".webp", ".gif"]

    for ext in possible_ext:
        img_path = input_dir / f"{image_name}{ext}"
        if img_path.exists():
            return str(img_path)
        img_path = DATASET_DIR / f"{image_name}{ext}"
        if img_path.exists():
            return str(img_path)

    return None


def handle_add_winner(image_name: str, tags: dict | None):
    """Add winner's embeddings to winners pool."""
    if not image_name:
        return "No image selected"

    emb_file = DATASET_DIR / f"{image_name}.embedding.json"
    if not emb_file.exists():
        return f"Embedding file not found for {image_name}"

    import json

    with open(emb_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_emb = data.get("image_embedding")
    tag_emb = data.get("tag_embedding")

    if img_emb is None:
        return "No image embedding found. Run embedding first."

    add_winner(img_emb, tag_emb)
    return f"✅ Added {image_name} to winners pool"


def handle_add_loser(image_name: str, tags: dict | None):
    """Add loser's embeddings to losers pool."""
    if not image_name:
        return "No image selected"

    emb_file = DATASET_DIR / f"{image_name}.embedding.json"
    if not emb_file.exists():
        return f"Embedding file not found for {image_name}"

    import json

    with open(emb_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_emb = data.get("image_embedding")
    tag_emb = data.get("tag_embedding")

    if img_emb is None:
        return "No image embedding found. Run embedding first."

    add_loser(img_emb, tag_emb)
    return f"✅ Added {image_name} to losers pool"


def handle_run_inference():
    """Run inference to find best candidate."""
    state = load_state()
    winners = state.get("winners", {})
    losers = state.get("losers", {})

    if winners.get("image_count", 0) == 0:
        return "No winners yet. Add some winners first.", None

    result = find_winner(DATASET_DIR, winners, losers)

    winner_name = result.get("winner", "Unknown")
    score = result.get("score", 0.0)

    # Load the winning image for display
    winner_img = load_image_for_comparison(winner_name.replace(".embedding.json", ""))

    return (
        f"🏆 Winner: {winner_name}\nScore: {score:.4f}",
        winner_img,
    )


def get_pool_status():
    """Get current winners/losers pool status."""
    state = load_state()
    winners = state.get("winners", {})
    losers = state.get("losers", {})

    return f"""
### Pool Status
**Winners:**
- Images: {winners.get("image_count", 0)}
- Tags: {winners.get("tag_count", 0)}

**Losers:**
- Images: {losers.get("image_count", 0)}
- Tags: {losers.get("tag_count", 0)}
"""


def create_preference_tab():
    """Create the preference comparison tab UI."""

    with gr.Tab("Preference Comparison"):
        gr.Markdown("## Preference Comparison & Inference")
        gr.Markdown(
            "Select two images, choose a winner/loser, and build pools for inference."
        )

        # Pool status
        pool_status = gr.Markdown(value=get_pool_status())

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Image A")
                image_a = gr.Image(type="filepath", label="Image A", height=300)
                choices_a = gr.Dropdown(
                    choices=scan_dataset_images(),
                    label="Select Image A",
                    allow_custom_value=True,
                )
                winner_a_btn = gr.Button("👑 Winner A", variant="primary")
                loser_a_btn = gr.Button("❌ Loser A", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### Image B")
                image_b = gr.Image(type="filepath", label="Image B", height=300)
                choices_b = gr.Dropdown(
                    choices=scan_dataset_images(),
                    label="Select Image B",
                    allow_custom_value=True,
                )
                winner_b_btn = gr.Button("👑 Winner B", variant="primary")
                loser_b_btn = gr.Button("❌ Loser B", variant="secondary")

        # Status displays
        with gr.Row():
            status_a = gr.Markdown(value="")
            status_b = gr.Markdown(value="")

        # Inference section
        gr.Markdown("### Run Inference")
        with gr.Row():
            refresh_pools_btn = gr.Button("🔄 Refresh Pools", variant="secondary")
            run_inference_btn = gr.Button("🏆 Find Winner", variant="primary")

        inference_status = gr.Markdown(value="")
        inference_result = gr.Image(
            type="filepath", label="Inference Winner", height=400
        )

        # State for storing tags (simplified - just use None for now)
        tags_a_state = gr.State(None)
        tags_b_state = gr.State(None)

        # Event handlers
        def update_image_a(name):
            img = load_image_for_comparison(name)
            return img

        def update_image_b(name):
            img = load_image_for_comparison(name)
            return img

        choices_a.change(fn=update_image_a, inputs=[choices_a], outputs=[image_a])
        choices_b.change(fn=update_image_b, inputs=[choices_b], outputs=[image_b])

        def add_winner_a(name):
            return handle_add_winner(name, None)

        def add_loser_a(name):
            return handle_add_loser(name, None)

        def add_winner_b(name):
            return handle_add_winner(name, None)

        def add_loser_b(name):
            return handle_add_loser(name, None)

        winner_a_btn.click(
            fn=add_winner_a,
            inputs=[choices_a],
            outputs=[status_a],
        ).then(fn=get_pool_status, outputs=[pool_status])

        loser_a_btn.click(
            fn=add_loser_a,
            inputs=[choices_a],
            outputs=[status_a],
        ).then(fn=get_pool_status, outputs=[pool_status])

        winner_b_btn.click(
            fn=add_winner_b,
            inputs=[choices_b],
            outputs=[status_b],
        ).then(fn=get_pool_status, outputs=[pool_status])

        loser_b_btn.click(
            fn=add_loser_b,
            inputs=[choices_b],
            outputs=[status_b],
        ).then(fn=get_pool_status, outputs=[pool_status])

        refresh_pools_btn.click(fn=get_pool_status, outputs=[pool_status])

        run_inference_btn.click(
            fn=handle_run_inference,
            inputs=[],
            outputs=[inference_status, inference_result],
        )

    return {
        "pool_status": pool_status,
        "choices_a": choices_a,
        "choices_b": choices_b,
    }


__all__ = ["create_preference_tab"]
