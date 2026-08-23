"""Tests for the DINOv2 + LinUCB image preference bandit package.

These tests never touch torch or the real DINOv2 model: FeatureStore accepts
an injectable extractor and the LinUCB math is pure numpy. The fake extractor
maps each image's channel means to the first 3 coordinates of an 8-d vector,
so image "preference" is determined by vector norm.
"""

import os
import time

import numpy as np
import pytest
from PIL import Image

from image_bandit.feature_store import (
    FEATURE_DIM,
    FeatureStore,
    scan_image_directory,
)
from image_bandit.linucb import LinUCBUser
from image_bandit.preference_profile import (
    load_preference_profile,
    save_preference_profile,
)
from image_bandit.recommender import BanditRecommender

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


def write_future_mtime(path):
    future = time.time() + 10
    os.utime(path, (future, future))


@pytest.fixture
def images_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name, color in FIXTURE_COLORS.items():
        write_image(d / f"{name}.png", color)
    return d


@pytest.fixture
def seeded_store(tmp_path, images_dir):
    ext = make_extractor()
    store = FeatureStore(tmp_path / "store", extractor=ext)
    store.ensure_features(images_dir)
    return store, ext


@pytest.fixture
def seeded_recommender(tmp_path, images_dir, seeded_store):
    store, _ = seeded_store
    profile = LinUCBUser(DIM)
    rec = BanditRecommender(
        store,
        profile,
        images_dir,
        batch_size=8,
        rng=np.random.default_rng(7),
    )
    return rec


class TestScanImageDirectory:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert scan_image_directory(tmp_path / "nope") == []

    def test_ignores_non_images(self, tmp_path):
        write_image(tmp_path / "a.png", (1, 2, 3))
        (tmp_path / "notes.txt").write_text("text")
        names = [p.name for p in scan_image_directory(tmp_path)]
        assert names == ["a.png"]

    def test_sorted_by_filename(self, tmp_path):
        write_image(tmp_path / "b.png", (1, 1, 1))
        write_image(tmp_path / "a.jpg", (1, 1, 1))
        names = [p.name for p in scan_image_directory(tmp_path)]
        assert names == ["a.jpg", "b.png"]


class TestFeatureStore:
    def test_extracts_new_images(self, tmp_path, images_dir):
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        stats = store.ensure_features(images_dir)
        assert stats["scanned"] == 8
        assert stats["new"] == 8
        assert stats["cached"] == 0
        assert stats["errors"] == 0
        assert len(store.keys) == 8

    def test_second_run_does_not_reprocess(self, tmp_path, images_dir):
        ext = make_extractor()
        store = FeatureStore(tmp_path / "store", extractor=ext)
        store.ensure_features(images_dir)
        calls_after_first = ext.calls["count"]

        stats = store.ensure_features(images_dir)
        assert stats["new"] == 0
        assert stats["cached"] == 8
        assert stats["reprocessed"] == 0
        assert ext.calls["count"] == calls_after_first

    def test_features_persist_across_instances(self, tmp_path, images_dir):
        FeatureStore(tmp_path / "store", extractor=make_extractor()).ensure_features(
            images_dir
        )
        reloaded = FeatureStore(tmp_path / "store")
        assert reloaded.keys == [f"img_{i}.png" for i in range(8)]
        feat = reloaded.feature("img_0.png")
        assert feat.shape == (DIM,)
        # img_0 is (250, 30, 10) -> first coordinate is the largest
        assert feat[0] == pytest.approx(250 / 255.0, abs=1e-5)

    def test_changed_image_is_reprocessed(self, tmp_path, images_dir):
        ext = make_extractor()
        store = FeatureStore(tmp_path / "store", extractor=ext)
        store.ensure_features(images_dir)
        calls_before = ext.calls["count"]

        write_image(images_dir / "img_2.png", (200, 0, 255))
        write_future_mtime(images_dir / "img_2.png")

        stats = store.ensure_features(images_dir)
        assert stats["reprocessed"] == 1
        assert stats["cached"] == 7
        assert ext.calls["count"] == calls_before + 1
        assert store.feature("img_2.png")[1] == pytest.approx(0.0, abs=1e-5)
        assert store.feature("img_2.png")[2] == pytest.approx(255 / 255.0, abs=1e-5)

    def test_deleted_image_is_removed(self, tmp_path, images_dir):
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        store.ensure_features(images_dir)

        (images_dir / "img_5.png").unlink()
        stats = store.ensure_features(images_dir)
        assert stats["removed"] == 1
        assert "img_5.png" not in store.keys
        assert len(store.keys) == 7

    def test_broken_image_counted_as_error(self, tmp_path, images_dir):
        (images_dir / "corrupt.png").write_text("not an image")
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        stats = store.ensure_features(images_dir)
        assert stats["errors"] == 1
        assert "corrupt.png" not in store.keys

    def test_missing_images_dir(self, tmp_path):
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        stats = store.ensure_features(tmp_path / "nope")
        assert stats["scanned"] == 0
        assert store.keys == []

    def test_random_batch(self, seeded_store):
        store, _ = seeded_store
        batch = store.random_batch(5, rng=np.random.default_rng(42))
        assert len(batch) == 5
        assert len(set(batch)) == 5
        assert set(batch) <= set(store.keys)

    def test_random_batch_larger_than_pool(self, seeded_store):
        store, _ = seeded_store
        batch = store.random_batch(100, rng=np.random.default_rng(42))
        assert sorted(batch) == sorted(store.keys)

    def test_random_batch_empty_store(self, tmp_path):
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        assert store.random_batch(5) == []

    def test_remove_drops_key_and_persists(self, tmp_path, images_dir):
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        store.ensure_features(images_dir)
        store.remove("img_5.png")
        assert "img_5.png" not in store.keys
        assert len(store.keys) == 7

        reloaded = FeatureStore(tmp_path / "store")
        assert "img_5.png" not in reloaded.keys
        assert len(reloaded.keys) == 7

    def test_remove_missing_key_is_noop(self, tmp_path, images_dir):
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        store.ensure_features(images_dir)
        store.remove("nope.png")
        assert len(store.keys) == 8


class TestLinUCBUser:
    def test_initial_score_is_alpha_times_norm(self):
        profile = LinUCBUser(4, alpha=0.1)
        assert profile.score_image(np.array([3.0, 4.0, 0.0, 0.0])) == pytest.approx(0.5)

    def test_score_images_matches_score_image(self):
        profile = LinUCBUser(3)
        x1 = np.array([1.0, 2.0, 0.0])
        x2 = np.array([0.0, 0.5, 1.0])
        profile.update_preferences(x1, 0.7)
        batched = profile.score_images([x1, x2])
        assert np.allclose(batched, [profile.score_image(x1), profile.score_image(x2)])

    def test_update_matches_dino_math(self):
        profile = LinUCBUser(2)
        v = np.array([1.0, 2.0])
        profile.update_preferences(v, 1.0)
        assert np.allclose(profile.A, np.eye(2) + np.outer(v, v))
        assert np.allclose(profile.b, v)
        assert profile.updates == 1


class TestPreferenceProfile:
    def test_roundtrip(self, tmp_path):
        profile = LinUCBUser(4, alpha=0.3)
        profile.update_preferences(np.array([1.0, 0.0, 0.0, 0.0]), 1.0)
        profile.update_preferences(np.array([0.0, 2.0, 0.0, 0.0]), -1.0)
        path = save_preference_profile(profile, tmp_path / "prof.npz")

        loaded = load_preference_profile(path, expected_dim=4)
        assert loaded is not None
        assert loaded.alpha == 0.3
        assert loaded.d == 4
        assert loaded.updates == 2
        assert np.allclose(loaded.A, profile.A)
        assert np.allclose(loaded.b, profile.b)

    def test_load_missing_returns_none(self, tmp_path):
        assert load_preference_profile(tmp_path / "nope.npz") is None

    def test_load_dim_mismatch_returns_none(self, tmp_path):
        profile = LinUCBUser(4)
        path = save_preference_profile(profile, tmp_path / "p.npz")
        assert load_preference_profile(path, expected_dim=FEATURE_DIM) is None


class TestBanditRecommender:
    def test_first_pair_is_top2(self, seeded_recommender):
        pair = seeded_recommender.next_pair()
        assert pair.left_key == "img_0.png"
        assert pair.right_key == "img_1.png"

    def test_pairs_follow_score_order(self, seeded_recommender):
        rec = seeded_recommender
        assert (rec.next_pair().left_key, rec.next_pair().left_key) == (
            "img_0.png",
            "img_7.png",
        )
        assert (rec.next_pair().left_key, rec.next_pair().left_key) == (
            "img_3.png",
            "img_2.png",
        )

    def test_choose_updates_profile_and_saves(
        self, tmp_path, images_dir, seeded_store
    ):
        store, _ = seeded_store
        profile_path = tmp_path / "profile.npz"
        profile = LinUCBUser(DIM)
        rec = BanditRecommender(
            store,
            profile,
            images_dir,
            batch_size=8,
            profile_path=profile_path,
            rng=np.random.default_rng(7),
        )
        pair = rec.next_pair()
        f_win = store.feature(pair.left_key)
        f_lose = store.feature(pair.right_key)
        score_win_before = profile.score_image(f_win)
        score_lose_before = profile.score_image(f_lose)

        result = rec.choose("left")
        assert result == {"winner": pair.left_key, "loser": pair.right_key}
        assert np.allclose(profile.b, f_win - f_lose)
        assert np.allclose(
            profile.A,
            np.eye(DIM) + np.outer(f_win, f_win) + np.outer(f_lose, f_lose),
        )
        theta = np.linalg.inv(profile.A) @ profile.b
        assert theta @ f_win > 0
        assert theta @ f_lose < 0
        assert profile.score_image(f_win) > score_win_before
        assert profile.score_image(f_lose) < score_lose_before

        loaded = load_preference_profile(profile_path, expected_dim=DIM)
        assert loaded is not None
        assert loaded.updates == 2
        assert np.allclose(loaded.A, profile.A)
        assert np.allclose(loaded.b, profile.b)

    def test_choose_right_marks_right_as_winner(
        self, tmp_path, images_dir, seeded_store
    ):
        store, _ = seeded_store
        profile = LinUCBUser(DIM)
        rec = BanditRecommender(
            store, profile, images_dir, batch_size=8, rng=np.random.default_rng(7)
        )
        pair = rec.next_pair()
        result = rec.choose("right")
        assert result == {"winner": pair.right_key, "loser": pair.left_key}

    def test_profile_reloaded_on_restart(
        self, tmp_path, images_dir, seeded_store
    ):
        store, _ = seeded_store
        profile_path = tmp_path / "profile.npz"
        rec = BanditRecommender(
            store,
            LinUCBUser(DIM),
            images_dir,
            batch_size=8,
            profile_path=profile_path,
            rng=np.random.default_rng(7),
        )
        rec.next_pair()
        rec.choose("left")

        reloaded = load_preference_profile(profile_path, expected_dim=DIM)
        rec2 = BanditRecommender(store, reloaded, images_dir, batch_size=8)
        assert rec2.profile.updates == 2
        assert not np.allclose(rec2.profile.b, np.zeros(DIM))

    def test_batch_exhaustion_draws_new_batch(
        self, tmp_path, images_dir, seeded_store
    ):
        store, _ = seeded_store
        profile = LinUCBUser(DIM)
        rec = BanditRecommender(
            store,
            profile,
            images_dir,
            batch_size=4,
            rng=np.random.default_rng(7),
        )
        rec.next_pair()
        assert len(rec._batch) == 2
        rec.next_pair()
        assert len(rec._batch) == 0
        pair = rec.next_pair()
        assert pair is not None
        assert len(rec._batch) == 2  # fresh batch drawn and 2 consumed

    def test_pairs_never_repeat_within_batch(self, tmp_path, images_dir, seeded_store):
        store, _ = seeded_store
        rec = BanditRecommender(
            store,
            LinUCBUser(DIM),
            images_dir,
            batch_size=8,
            rng=np.random.default_rng(7),
        )
        shown = []
        for _ in range(4):
            pair = rec.next_pair()
            shown.extend([pair.left_key, pair.right_key])
        assert len(set(shown)) == 8

    def test_next_pair_empty_store(self, tmp_path):
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        rec = BanditRecommender(store, LinUCBUser(DIM), tmp_path / "images")
        assert rec.next_pair() is None

    def test_choose_without_pair_raises(self, seeded_recommender):
        with pytest.raises(RuntimeError):
            seeded_recommender.choose("left")

    def test_choose_bad_side_raises(self, seeded_recommender):
        seeded_recommender.next_pair()
        with pytest.raises(ValueError):
            seeded_recommender.choose("middle")

    def test_skip_returns_pair_and_clears_pending(self, seeded_recommender):
        rec = seeded_recommender
        pair = rec.next_pair()
        result = rec.skip()
        assert result == {"left": pair.left_key, "right": pair.right_key}
        assert rec._current_pair is None
        assert rec.skipped_pairs == 1
        assert rec.profile.updates == 0

    def test_skip_advances_to_next_pair(self, seeded_recommender):
        rec = seeded_recommender
        pair1 = rec.next_pair()
        assert (pair1.left_key, pair1.right_key) == ("img_0.png", "img_1.png")

        rec.skip()
        pair2 = rec.next_pair()
        assert (pair2.left_key, pair2.right_key) == ("img_7.png", "img_5.png")

    def test_skipped_images_are_not_shown_again(self, seeded_recommender):
        rec = seeded_recommender
        rec.next_pair()
        rec.skip()
        shown = []
        for _ in range(3):
            pair = rec.next_pair()
            shown.extend([pair.left_key, pair.right_key])
        assert "img_0.png" not in shown
        assert "img_1.png" not in shown
        assert len(set(shown)) == 6

    def test_skip_without_pair_raises(self, seeded_recommender):
        with pytest.raises(RuntimeError):
            seeded_recommender.skip()

    def test_delete_left_refills_left_slot(self, seeded_recommender, images_dir):
        rec = seeded_recommender
        pair1 = rec.next_pair()
        assert (pair1.left_key, pair1.right_key) == ("img_0.png", "img_1.png")

        result = rec.delete_image("left")
        assert result == {
            "deleted": "img_0.png",
            "deleted_name": "img_0.png",
            "kept": "img_1.png",
            "replacement": "img_7.png",
        }
        assert not (images_dir / "img_0.png").exists()
        assert (images_dir / "img_1.png").exists()
        assert "img_0.png" not in rec.store.keys
        # The kept image stays on the right; the replacement takes the left.
        assert (rec._current_pair.left_key, rec._current_pair.right_key) == (
            "img_7.png",
            "img_1.png",
        )
        assert rec.profile.updates == 0
        assert rec.deleted_images == 1

    def test_delete_right_keeps_left_in_place(self, seeded_recommender, images_dir):
        rec = seeded_recommender
        pair1 = rec.next_pair()
        assert (pair1.left_key, pair1.right_key) == ("img_0.png", "img_1.png")

        result = rec.delete_image("right")
        assert result["deleted"] == "img_1.png"
        assert result["kept"] == "img_0.png"
        assert not (images_dir / "img_1.png").exists()
        assert "img_1.png" not in rec.store.keys
        assert (rec._current_pair.left_key, rec._current_pair.right_key) == (
            "img_0.png",
            result["replacement"],
        )
        assert rec._current_pair.right_key != "img_0.png"

    def test_delete_draws_fresh_batch_when_exhausted(
        self, tmp_path, images_dir, seeded_store
    ):
        store, _ = seeded_store
        profile = LinUCBUser(DIM)
        rec = BanditRecommender(
            store, profile, images_dir, batch_size=4, rng=np.random.default_rng(7)
        )
        rec.next_pair()
        rec.next_pair()
        assert len(rec._batch) == 0
        pair = rec._current_pair

        result = rec.delete_image("left")
        assert result["replacement"] is not None
        assert result["replacement"] != pair.right_key
        assert rec._current_pair is not None
        assert rec._current_pair.right_key == pair.right_key
        assert profile.updates == 0

    def test_delete_with_no_candidates_clears_pair(self, tmp_path):
        d = tmp_path / "two"
        d.mkdir()
        write_image(d / "a.png", (250, 30, 10))
        write_image(d / "b.png", (20, 240, 10))
        store = FeatureStore(tmp_path / "store", extractor=make_extractor())
        store.ensure_features(d)
        rec = BanditRecommender(
            store, LinUCBUser(DIM), d, batch_size=8, rng=np.random.default_rng(7)
        )
        pair = rec.next_pair()
        assert pair is not None

        result = rec.delete_image("left")
        assert result["replacement"] is None
        assert rec._current_pair is None
        assert not (d / pair.left_key).exists()
        assert (d / pair.right_key).exists()

    def test_delete_without_pair_raises(self, seeded_recommender):
        with pytest.raises(RuntimeError):
            seeded_recommender.delete_image("left")

    def test_delete_bad_side_raises(self, seeded_recommender):
        seeded_recommender.next_pair()
        with pytest.raises(ValueError):
            seeded_recommender.delete_image("middle")

    def test_path_for_and_display_name(self, seeded_recommender, images_dir):
        rec = seeded_recommender
        assert rec.path_for("img_0.png") == images_dir / "img_0.png"
        assert rec.display_name("img_0.png") == "img_0.png"


class TestConstants:
    def test_feature_dim(self):
        assert FEATURE_DIM == 1152


class TestAppConstruction:
    def test_create_bandit_app_builds(self, tmp_path):
        from image_bandit.app import create_bandit_app

        demo = create_bandit_app(
            images_dir=tmp_path / "imgs", data_dir=tmp_path / "data"
        )
        assert demo is not None
