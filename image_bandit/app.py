"""Gradio UI for the DINOv2 + LinUCB image preference loop.

Shows two candidate images (the top-2 bandit scores from a random batch),
lets the user pick the one they prefer, updates the LinUCB profile
(+1.0 / -1.0), saves it, and advances to the next pair until the program is
shut down or the image pool runs out.
"""

from pathlib import Path

import gradio as gr

from image_bandit.feature_store import DINOV2_MODEL, FEATURE_DIM, FeatureStore
from image_bandit.linucb import LinUCBUser
from image_bandit.preference_profile import load_preference_profile
from image_bandit.recommender import BanditRecommender

DEFAULT_IMAGES_DIR = "images"
DEFAULT_DATA_DIR = "data/bandit"
DEFAULT_BATCH_SIZE = 8

_HIDDEN = (gr.update(visible=False), gr.update(visible=False))
_INACTIVE = (gr.update(interactive=False), gr.update(interactive=False))
_ACTIVE = (gr.update(interactive=True), gr.update(interactive=True))


def _stats_line(stats: dict) -> str:
    line = (
        f"🔍 Scanned {stats['scanned']} images · {stats['new']} new · "
        f"{stats['cached']} cached · {stats['reprocessed']} reprocessed · "
        f"{stats['removed']} removed"
    )
    if stats.get("errors"):
        line += f" · ⚠️ {stats['errors']} errors"
    return line


def create_bandit_app(
    images_dir=DEFAULT_IMAGES_DIR,
    data_dir=DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = DINOV2_MODEL,
) -> gr.Blocks:
    """Build the image preference bandit Blocks application."""
    images_dir = Path(images_dir)
    data_dir = Path(data_dir)
    profile_path = data_dir / "preference_profile.npz"

    store = FeatureStore(data_dir / "features")
    profile = load_preference_profile(profile_path, expected_dim=FEATURE_DIM)
    if profile is None:
        profile = LinUCBUser(FEATURE_DIM)
    recommender = BanditRecommender(
        store,
        profile,
        images_dir,
        batch_size=batch_size,
        profile_path=profile_path,
    )

    def pair_outputs(pair) -> tuple:
        return (
            str(recommender.path_for(pair.left_key)),
            str(recommender.path_for(pair.right_key)),
            *_ACTIVE,
        )

    def no_pair_message(reason: str) -> tuple:
        return (reason, *_HIDDEN, *_INACTIVE)

    def setup():
        yield (
            "⏳ Scanning images and computing dense features…",
            *_HIDDEN,
            *_INACTIVE,
        )
        stats = store.ensure_features(images_dir)
        pair = recommender.next_pair()
        if pair is None:
            yield no_pair_message(
                "❌ No images found in the images directory. "
                "Add images and click Scan."
            )
            return
        yield (
            f"{_stats_line(stats)}\n\n{recommender.status()}",
            *pair_outputs(pair),
        )

    def on_choice(side: str) -> tuple:
        if recommender._current_pair is None:
            return no_pair_message(
                "⚠️ No pending pair. Click Scan for new images."
            )
        result = recommender.choose(side)
        pair = recommender.next_pair()
        if pair is None:
            return no_pair_message(
                f"🏆 Winner: {recommender.display_name(result['winner'])} · "
                f"❌ Loser: {recommender.display_name(result['loser'])}\n\n"
                f"⏸ No more candidates to show. Add images to the directory "
                f"and click Scan.\n\n{recommender.status()}"
            )
        return (
            f"🏆 Winner: {recommender.display_name(result['winner'])} · "
            f"❌ Loser: {recommender.display_name(result['loser'])}\n\n"
            f"{recommender.status()}",
            *pair_outputs(pair),
        )

    def on_rescan():
        yield no_pair_message("⏳ Scanning images…")
        stats = store.ensure_features(images_dir)
        pair = recommender._current_pair or recommender.next_pair()
        if pair is None:
            yield no_pair_message(
                "❌ No images found in the images directory."
            )
            return
        yield (
            f"{_stats_line(stats)}\n\n{recommender.status()}",
            *pair_outputs(pair),
        )

    with gr.Blocks(title="Image Preference Bandit") as demo:
        gr.Markdown("# 🖼️ DINOv2 Image Preference Bandit")
        gr.Markdown(
            "Two candidate images are scored with LinUCB. Pick the one you "
            "prefer — the bandit updates your preference profile (+1.0 / "
            "-1.0) and saves it after every choice."
        )
        status_md = gr.Markdown("Starting…")

        with gr.Row():
            with gr.Column():
                img_left = gr.Image(label="Candidate A", value=None, visible=False)
                btn_left = gr.Button(
                    "👈 I prefer this image", variant="primary", interactive=False
                )
            with gr.Column():
                img_right = gr.Image(label="Candidate B", value=None, visible=False)
                btn_right = gr.Button(
                    "I prefer this image 👉", variant="primary", interactive=False
                )

        scan_btn = gr.Button("🔄 Scan for new images")

        gr.Markdown(
            f"Images: `{images_dir}/` · Features & profile: `{data_dir}/` · "
            f"Batch size: {batch_size}"
        )

        outputs = [status_md, img_left, img_right, btn_left, btn_right]
        demo.load(setup, outputs=outputs)
        btn_left.click(lambda: on_choice("left"), outputs=outputs)
        btn_right.click(lambda: on_choice("right"), outputs=outputs)
        scan_btn.click(on_rescan, outputs=outputs)

    return demo


__all__ = [
    "create_bandit_app",
    "DEFAULT_IMAGES_DIR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_BATCH_SIZE",
]
