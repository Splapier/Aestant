"""Tests for manual inference over the input directory.

Like the bandit tests, these never touch torch or the real DINOv2 model:
``ContentFeatureStore`` accepts an injectable extractor and the LinUCB math is
pure numpy. The fake extractor maps each image's channel means to the first 3
coordinates of an 8-d vector, so with a clean profile the score is
``alpha * ||v||`` and the ranking is by vector norm.
"""

import numpy as np
import pytest
from PIL import Image

from image_bandit.feature_store import file_fingerprint
from image_bandit.linucb import LinUCBUser
from image_bandit.manual_inference import (
    ContentFeatureStore,
    get_profile,
    run_manual_inference,
)
from image_bandit.preference_profile import save_preference_profile

DIM = 8

# Fixture colors chosen so vector norms rank:
# img_0 > img_1 > img_7 > img_5 > img_3 > img_4 > img_2 > img_6
FIXTURE_COLORS = {
    "img_0": (250, 30, 10),
    "img_1": (20, 240, 10),
    "img_2": (100, 100, 100),
    "img_3": (180, 50, 10),
    "img_4": (50, 170, 10),
    "img_5": (50, 50, 180),
    "img_6": (10, 10, 10),
    "img_7": (120, 120, 120),
}
EXPECTED_TOP5 = [
    "img_0.png",
    "img_1.png",
    "img_7.png",
    "img_5.png",
    "img_3.png",
]


def make_extractor(dim=DIM):
    """Fake feature extractor: channel means -> first 3 coordinates."""
    calls = {"count": 0}

    def extract(img):
        calls["count"] += 1
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        channel_mean = arr.reshape(-1, 3).mean(axis=0)
        vec = np.zeros(dim, dtype=np.float32)
        vec[:3] = channel_mean / 255.0
        return vec

    extract.calls = calls
    return extract


def write_image(path, color):
    Image.new("RGB", (8, 8), color).save(path)


def make_checkerboard(size=16, cell=2):
    """Return an RGB image of a black/white checkerboard (high detail)."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x // cell + y // cell) % 2 == 0:
                arr[y, x] = 255
    return Image.fromarray(arr)


def write_gif(path, frames):
    """Save a list of same-size RGB frames as an animated GIF."""
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )


@pytest.fixture
def input_dir(tmp_path):
    d = tmp_path / "input"
    d.mkdir()
    for name, color in FIXTURE_COLORS.items():
        write_image(d / f"{name}.png", color)
    return d


@pytest.fixture
def content_store(tmp_path, input_dir):
    ext = make_extractor()
    store = ContentFeatureStore(tmp_path / "store", extractor=ext)
    store.ensure_features(input_dir)
    return store, ext


class TestContentFeatureStore:
    def test_extracts_new_images_keyed_by_hash(self, tmp_path, input_dir):
        store = ContentFeatureStore(tmp_path / "store", extractor=make_extractor())
        result = store.ensure_features(input_dir)
        assert result.scanned == 8
        assert result.new == 8
        assert result.cached == 0
        assert result.errors == 0
        assert len(store.keys) == 8
        # keys are content hashes, not relative paths
        for key in store.keys:
            assert key != "img_0.png"
            assert len(key) == 64  # sha256 hex digest length

    def test_second_run_does_not_reprocess(self, content_store, input_dir):
        store, ext = content_store
        calls_after_first = ext.calls["count"]
        result = store.ensure_features(input_dir)
        assert result.new == 0
        assert result.cached == 8
        assert ext.calls["count"] == calls_after_first

    def test_renamed_image_same_content_not_reextracted(
        self, tmp_path, input_dir
    ):
        ext = make_extractor()
        store = ContentFeatureStore(tmp_path / "store", extractor=ext)
        store.ensure_features(input_dir)
        calls_before = ext.calls["count"]

        # Move the file to a new name; content is unchanged.
        (input_dir / "img_0.png").rename(input_dir / "renamed.png")
        result = store.ensure_features(input_dir)
        assert result.new == 0
        assert result.cached == 8
        assert ext.calls["count"] == calls_before
        # The feature for the (renamed) image is still present.
        digest = file_fingerprint(input_dir / "renamed.png")
        assert store.has(digest)
        assert result.path_to_hash[input_dir / "renamed.png"] == digest

    def test_removed_image_is_not_pruned(self, tmp_path, input_dir):
        store = ContentFeatureStore(tmp_path / "store", extractor=make_extractor())
        store.ensure_features(input_dir)
        kept_keys = set(store.keys)
        assert len(kept_keys) == 8

        (input_dir / "img_5.png").unlink()
        store.ensure_features(input_dir)
        # Non-destructive: the removed image's content hash is retained.
        assert set(store.keys) == kept_keys
        assert len(store.keys) == 8

    def test_features_persist_across_instances(self, tmp_path, input_dir):
        ContentFeatureStore(tmp_path / "store", extractor=make_extractor()).ensure_features(
            input_dir
        )
        reloaded = ContentFeatureStore(tmp_path / "store")
        assert len(reloaded.keys) == 8
        digest = file_fingerprint(input_dir / "img_0.png")
        feat = reloaded.feature(digest)
        assert feat.shape == (DIM,)
        assert feat[0] == pytest.approx(250 / 255.0, abs=1e-5)

    def test_missing_dir_returns_empty(self, tmp_path):
        store = ContentFeatureStore(tmp_path / "store", extractor=make_extractor())
        result = store.ensure_features(tmp_path / "nope")
        assert result.scanned == 0
        assert store.keys == []

    def test_gif_features_use_best_frame(self, tmp_path):
        d = tmp_path / "input"
        d.mkdir()
        # Frame 0 is flat blue; frame 1 is a half-white/half-black
        # checkerboard (the most detailed frame).
        write_gif(
            d / "anim.gif",
            [Image.new("RGB", (16, 16), (0, 0, 255)), make_checkerboard()],
        )
        store = ContentFeatureStore(tmp_path / "store", extractor=make_extractor())
        result = store.ensure_features(d)
        assert result.scanned == 1
        assert result.new == 1
        assert result.errors == 0
        feat = store.feature(file_fingerprint(d / "anim.gif"))
        # The checkerboard's channel means are ~0.5 on every channel, while
        # the blue first frame would give (0.0, 0.0, 1.0).
        assert feat[0] == pytest.approx(0.5, abs=0.05)
        assert feat[1] == pytest.approx(0.5, abs=0.05)
        assert feat[2] == pytest.approx(0.5, abs=0.05)


class TestRunManualInference:
    def test_top_image_and_top5_order(self, content_store, input_dir):
        store, _ = content_store
        profile = LinUCBUser(DIM)
        result = run_manual_inference(input_dir, store, profile)
        assert result is not None
        assert result.top_path.name == "img_0.png"
        assert [name for name, _ in result.top5] == EXPECTED_TOP5
        assert len(result.top5) == 5
        # scores are descending
        scores = [score for _, score in result.top5]
        assert scores == sorted(scores, reverse=True)
        assert result.top_score == pytest.approx(scores[0])

    def test_top_n_limits_list(self, content_store, input_dir):
        store, _ = content_store
        profile = LinUCBUser(DIM)
        result = run_manual_inference(input_dir, store, profile, top_n=3)
        assert [name for name, _ in result.top5] == EXPECTED_TOP5[:3]

    def test_empty_dir_returns_none(self, tmp_path, content_store):
        store, _ = content_store
        profile = LinUCBUser(DIM)
        empty = tmp_path / "empty"
        empty.mkdir()
        assert run_manual_inference(empty, store, profile) is None

    def test_does_not_modify_profile(self, content_store, input_dir):
        store, _ = content_store
        profile = LinUCBUser(DIM)
        A_before = profile.A.copy()
        b_before = profile.b.copy()
        updates_before = profile.updates

        run_manual_inference(input_dir, store, profile)

        assert np.allclose(profile.A, A_before)
        assert np.allclose(profile.b, b_before)
        assert profile.updates == updates_before

    def test_uses_existing_profile(self, content_store, input_dir):
        store, _ = content_store
        # A profile with a strong prior toward img_5 (blue) should flip the
        # ranking away from the clean-profile top (img_0). img_5 has the
        # largest dot product with its own feature direction, so a large
        # reward on it makes it the top.
        profile = LinUCBUser(DIM)
        v5 = store.feature(file_fingerprint(input_dir / "img_5.png"))
        profile.update_preferences(v5, 1000.0)
        result = run_manual_inference(input_dir, store, profile)
        assert result.top_path.name == "img_5.png"


class TestGetProfile:
    def test_returns_saved_profile(self, tmp_path):
        profile = LinUCBUser(DIM)
        profile.update_preferences(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]), 1.0)
        path = save_preference_profile(profile, tmp_path / "prof.npz")
        loaded = get_profile(path, expected_dim=DIM)
        assert loaded is not None
        assert np.allclose(loaded.A, profile.A)
        assert np.allclose(loaded.b, profile.b)

    def test_returns_clean_profile_when_absent(self, tmp_path):
        loaded = get_profile(tmp_path / "nope.npz", expected_dim=DIM)
        assert loaded is not None
        assert np.allclose(loaded.b, np.zeros(DIM))
        assert np.allclose(loaded.A, np.eye(DIM))


class TestAppConstruction:
    def test_create_bandit_app_with_input_dir(self, tmp_path):
        from image_bandit.app import create_bandit_app

        demo = create_bandit_app(
            images_dir=tmp_path / "imgs",
            input_dir=tmp_path / "input",
            data_dir=tmp_path / "data",
        )
        assert demo is not None
