"""Random-batch recommendation loop: score a batch, show the top 2, learn.

Each round the bandit scores every remaining image in the current random
batch and the two highest-scoring images are shown to the user. The user's
choice updates the LinUCB profile (+1.0 winner / -1.0 loser), the profile is
saved to disk, and the next pair is drawn from the same batch until it is
exhausted - at which point a fresh random batch is retrieved.

The user can also skip a pair (both images are discarded from the batch and
the next top-2 is shown) or delete one of the two images (the file is removed
from disk and the store, the other image stays in place, and the empty slot
is filled with the next-highest-scoring image from the batch).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image_bandit.feature_store import FeatureStore
from image_bandit.linucb import LinUCBUser
from image_bandit.preference_profile import save_preference_profile

WINNER_REWARD = 1.0
LOSER_REWARD = -1.0


@dataclass
class Pair:
    left_key: str
    right_key: str


class BanditRecommender:
    """Drives the score -> show top 2 -> choose -> update loop."""

    def __init__(
        self,
        store: FeatureStore,
        profile: LinUCBUser,
        images_dir,
        batch_size: int = 8,
        profile_path=None,
        rng: np.random.Generator | None = None,
    ):
        self.store = store
        self.profile = profile
        self.images_dir = Path(images_dir)
        self.batch_size = batch_size
        self.profile_path = Path(profile_path) if profile_path is not None else None
        self.rng = rng if rng is not None else np.random.default_rng()
        self._batch: list[str] = []
        self._current_pair: Pair | None = None
        self.shown_pairs = 0
        self.skipped_pairs = 0
        self.deleted_images = 0

    def _draw_batch(self) -> None:
        self._batch = self.store.random_batch(self.batch_size, self.rng)
        self._current_pair = None

    def next_pair(self) -> Pair | None:
        """Score the batch and return its top-2 images (higher score on the left)."""
        if len(self._batch) < 2:
            self._draw_batch()
        if len(self._batch) < 2:
            return None
        vectors = np.stack([self.store.feature(k) for k in self._batch])
        scores = self.profile.score_images(vectors)
        order = np.argsort(scores)[::-1]
        top1 = self._batch[int(order[0])]
        top2 = self._batch[int(order[1])]
        self._batch.remove(top1)
        self._batch.remove(top2)
        self._current_pair = Pair(top1, top2)
        return self._current_pair

    def choose(self, side: str) -> dict:
        """Apply the user's choice (+1.0 winner / -1.0 loser) and save the profile."""
        if self._current_pair is None:
            raise RuntimeError("No pending pair to choose from")
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if side == "left":
            winner_key, loser_key = self._current_pair.left_key, self._current_pair.right_key
        else:
            winner_key, loser_key = self._current_pair.right_key, self._current_pair.left_key
        self.profile.update_preferences(self.store.feature(winner_key), WINNER_REWARD)
        self.profile.update_preferences(self.store.feature(loser_key), LOSER_REWARD)
        if self.profile_path is not None:
            save_preference_profile(self.profile, self.profile_path)
        self._current_pair = None
        self.shown_pairs += 1
        return {"winner": winner_key, "loser": loser_key}

    def skip(self) -> dict:
        """Discard the current pair (no preference update) so the next
        top-2 of the batch can be shown."""
        if self._current_pair is None:
            raise RuntimeError("No pending pair to skip")
        result = {
            "left": self._current_pair.left_key,
            "right": self._current_pair.right_key,
        }
        self._current_pair = None
        self.skipped_pairs += 1
        return result

    def delete_image(self, side: str) -> dict:
        """Delete one image of the current pair from disk and the feature
        store. The other image stays in place; the empty slot is filled with
        the next-highest-scoring image from the batch (a fresh batch is drawn
        if the current one is exhausted). No preference update is applied.

        Returns ``{"deleted", "kept", "replacement"}``; ``replacement`` is
        ``None`` (and the pending pair is cleared) when no other image is
        available.
        """
        if self._current_pair is None:
            raise RuntimeError("No pending pair to delete from")
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        pair = self._current_pair
        deleted_key = pair.left_key if side == "left" else pair.right_key
        kept_key = pair.right_key if side == "left" else pair.left_key

        self.path_for(deleted_key).unlink(missing_ok=True)
        self.store.remove(deleted_key)
        self.deleted_images += 1

        if len(self._batch) < 1:
            self._draw_batch()
        # The kept image is already on screen, so it is never a candidate.
        self._batch = [k for k in self._batch if k != kept_key]
        if not self._batch:
            self._current_pair = None
            return {"deleted": deleted_key, "kept": kept_key, "replacement": None}

        vectors = np.stack([self.store.feature(k) for k in self._batch])
        scores = self.profile.score_images(vectors)
        replacement = self._batch[int(np.argmax(scores))]
        self._batch.remove(replacement)
        if side == "left":
            self._current_pair = Pair(replacement, kept_key)
        else:
            self._current_pair = Pair(kept_key, replacement)
        return {"deleted": deleted_key, "kept": kept_key, "replacement": replacement}

    def path_for(self, key: str) -> Path:
        return self.images_dir / self.store.rel_path(key)

    def display_name(self, key: str) -> str:
        return Path(self.store.rel_path(key)).name

    def status(self) -> str:
        return (
            f"Batch remaining: {len(self._batch)} · "
            f"Pairs shown: {self.shown_pairs} · "
            f"Skipped: {self.skipped_pairs} · "
            f"Deleted: {self.deleted_images} · "
            f"Preference updates: {self.profile.updates}"
        )
