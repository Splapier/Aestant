"""Gradio UI for the DINOv2 + LinUCB image preference loop.

Shows two candidate images (the top-2 bandit scores from a random batch),
lets the user pick the one they prefer, updates the LinUCB profile
(+1.0 / -1.0), saves it, and advances to the next pair until the program is
shut down or the image pool runs out. The user can also skip a pair (the
next top-2 of the batch is shown) or delete either image (the file is
removed, the other image stays in place, and a replacement fills the empty
slot).
"""

from pathlib import Path

import gradio as gr

from image_bandit.feature_store import DINOV2_MODEL, FEATURE_DIM, FeatureStore
from image_bandit.linucb import LinUCBUser
from image_bandit.manual_inference import (
    ContentFeatureStore,
    run_manual_inference,
)
from image_bandit.preference_profile import load_preference_profile
from image_bandit.recommender import BanditRecommender

DEFAULT_IMAGES_DIR = "images"
DEFAULT_INPUT_DIR = "input"
DEFAULT_DATA_DIR = "data/bandit"
DEFAULT_BATCH_SIZE = 8
DEFAULT_TOP_N = 5

_ROW_HIDDEN = gr.update(visible=False)
_ROW_SHOWN = gr.update(visible=True)
_CLEAR_IMAGES = (gr.update(value=None), gr.update(value=None))
_SKIP_HIDDEN = gr.update(visible=False)
_SKIP_SHOWN = gr.update(visible=True)
# One slot per action button: prefer left/right, delete left/right, skip.
_BUTTON_SLOTS = 5
_INACTIVE = tuple(gr.update(interactive=False) for _ in range(_BUTTON_SLOTS))
_ACTIVE = tuple(gr.update(interactive=True) for _ in range(_BUTTON_SLOTS))


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
    input_dir=DEFAULT_INPUT_DIR,
    data_dir=DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    top_n: int = DEFAULT_TOP_N,
    model_name: str = DINOV2_MODEL,
) -> gr.Blocks:
    """Build the image preference bandit Blocks application."""
    images_dir = Path(images_dir)
    input_dir = Path(input_dir)
    data_dir = Path(data_dir)
    profile_path = data_dir / "preference_profile.npz"

    store = FeatureStore(data_dir / "features")
    input_store = ContentFeatureStore(data_dir / "input_features")
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
            _ROW_SHOWN,
            gr.update(value=str(recommender.path_for(pair.left_key))),
            gr.update(value=str(recommender.path_for(pair.right_key))),
            *_ACTIVE,
            _SKIP_SHOWN,
        )

    def no_pair_message(reason: str) -> tuple:
        return (reason, _ROW_HIDDEN, *_CLEAR_IMAGES, *_INACTIVE, _SKIP_HIDDEN)

    def setup():
        yield (
            "⏳ Scanning images and computing dense features…",
            _ROW_HIDDEN,
            *_CLEAR_IMAGES,
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

    def on_skip() -> tuple:
        if recommender._current_pair is None:
            return no_pair_message(
                "⚠️ No pending pair. Click Scan for new images."
            )
        result = recommender.skip()
        pair = recommender.next_pair()
        skipped = (
            f"⏭️ Skipped: {recommender.display_name(result['left'])} vs "
            f"{recommender.display_name(result['right'])}"
        )
        if pair is None:
            return no_pair_message(
                f"{skipped}\n\n⏸ No more candidates to show. Add images to "
                f"the directory and click Scan.\n\n{recommender.status()}"
            )
        return (
            f"{skipped}\n\n{recommender.status()}",
            *pair_outputs(pair),
        )

    def on_delete(side: str) -> tuple:
        if recommender._current_pair is None:
            return no_pair_message(
                "⚠️ No pending pair. Click Scan for new images."
            )
        result = recommender.delete_image(side)
        pair = recommender._current_pair
        deleted = f"🗑️ Deleted: {recommender.display_name(result['deleted'])}"
        if pair is None:
            return no_pair_message(
                f"{deleted}\n\n⏸ No more candidates to show. Add images to "
                f"the directory and click Scan.\n\n{recommender.status()}"
            )
        return (
            f"{deleted}\n\n{recommender.status()}",
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

    def _top5_lines(top5: list) -> str:
        lines = ["**Top 5 scores:**"]
        for rank, (name, score) in enumerate(top5, 1):
            lines.append(f"{rank}. `{name}` — {score:.4f}")
        return "\n".join(lines)

    def _no_images_inference() -> tuple:
        return (
            f"❌ No images found in `{input_dir}/`. Add images and click Run.",
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value=""),
        )

    def run_inference():
        yield (
            f"⏳ Scanning `{input_dir}/` and computing dense features…",
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value=""),
        )
        result = run_manual_inference(input_dir, input_store, profile, top_n=top_n)
        if result is None:
            yield _no_images_inference()
            return
        stats = result.stats
        line = (
            f"🔍 Scanned {stats.scanned} images · {stats.new} new · "
            f"{stats.cached} cached"
        )
        if stats.errors:
            line += f" · ⚠️ {stats.errors} errors"
        yield (
            line,
            gr.update(value=str(result.top_path)),
            gr.update(
                value=(
                    f"🏆 **Top image:** `{result.top_path.name}`  \n"
                    f"**Score:** {result.top_score:.4f}"
                )
            ),
            gr.update(value=_top5_lines(result.top5)),
        )

    with gr.Blocks(title="Image Preference Bandit") as demo:
        gr.Markdown("# 🖼️ DINOv2 Image Preference Bandit")
        gr.Markdown(
            "Two candidate images are scored with LinUCB. Pick the one you "
            "prefer — the bandit updates your preference profile (+1.0 / "
            "-1.0) and saves it after every choice. You can also skip a pair "
            "(the next two from the batch are shown) or delete an image (it "
            "is removed from disk, its slot is refilled, and the other image "
            "stays in place)."
        )
        status_md = gr.Markdown("Starting…")

        pair_row = gr.Row(visible=False)
        with pair_row:
            with gr.Column():
                img_left = gr.Image(label="Candidate A", value=None, visible=True)
                btn_left = gr.Button(
                    "👈 I prefer this image", variant="primary", interactive=False
                )
                del_left = gr.Button(
                    "🗑️ Delete this image", variant="secondary", interactive=False
                )
            with gr.Column():
                img_right = gr.Image(label="Candidate B", value=None, visible=True)
                btn_right = gr.Button(
                    "I prefer this image 👉", variant="primary", interactive=False
                )
                del_right = gr.Button(
                    "🗑️ Delete this image", variant="secondary", interactive=False
                )

        skip_btn = gr.Button("⏭️ Skip this pair", visible=False)
        scan_btn = gr.Button("🔄 Scan for new images")

        gr.Markdown(
            f"Images: `{images_dir}/` · Features & profile: `{data_dir}/` · "
            f"Batch size: {batch_size}"
        )

        gr.Markdown("---")
        gr.Markdown("## 🔍 Manual Inference")
        gr.Markdown(
            f"Scores every image in `{input_dir}/` with the current preference "
            "profile and shows the top image with its score, plus the top-5 "
            "scores. This only reads the profile - it never updates it, so it "
            "does not interfere with the bandit loop."
        )
        infer_status_md = gr.Markdown("Starting…")
        infer_btn = gr.Button("🔍 Run manual inference")
        with gr.Row():
            top_image = gr.Image(label="Top image", value=None, visible=True)
            with gr.Column():
                top_score_md = gr.Markdown("")
                top5_md = gr.Markdown("")

        outputs = [
            status_md,
            pair_row,
            img_left,
            img_right,
            btn_left,
            btn_right,
            del_left,
            del_right,
            skip_btn,
        ]
        infer_outputs = [infer_status_md, top_image, top_score_md, top5_md]
        demo.load(setup, outputs=outputs)
        demo.load(run_inference, outputs=infer_outputs)
        btn_left.click(lambda: on_choice("left"), outputs=outputs)
        btn_right.click(lambda: on_choice("right"), outputs=outputs)
        del_left.click(lambda: on_delete("left"), outputs=outputs)
        del_right.click(lambda: on_delete("right"), outputs=outputs)
        skip_btn.click(on_skip, outputs=outputs)
        scan_btn.click(on_rescan, outputs=outputs)
        infer_btn.click(run_inference, outputs=infer_outputs)

    return demo


__all__ = [
    "create_bandit_app",
    "DEFAULT_IMAGES_DIR",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TOP_N",
]
