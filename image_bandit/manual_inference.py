"""Manual inference: score every image in the input directory with the
current LinUCB preference profile and report the top image plus the top-5
scores.

Unlike the bandit loop (``FeatureStore``), which keys features by relative
path over the images directory and prunes entries on rescan, this module keys
features by the SHA-256 of the file *contents*. That makes the cache safe to
store without reliance on an image's location: an image that is moved,
renamed, or re-added under a new name is recognized by its content and never
re-extracted, and a scan never deletes entries because input images are
expected to be moved around.

Manual inference only *reads* the preference profile - it never calls
``update_preferences`` - so it cannot interfere with the bandit loop or the
profile it persists.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from image_bandit.feature_store import (
    FEATURE_DIM,
    extract_dense_features,
    file_fingerprint,
    load_best_frame,
    scan_image_directory,
)
from image_bandit.linucb import LinUCBUser
from image_bandit.preference_profile import load_preference_profile

CONTENT_FEATURES_FILE = "content_features.npz"
CONTENT_INDEX_FILE = "content_index.json"


@dataclass
class ScanResult:
    """Outcome of scanning the input directory for cached/new features."""

    scanned: int
    new: int
    cached: int
    errors: int
    # Maps each scanned image path to its SHA-256 content hash.
    path_to_hash: dict = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Outcome of manual inference over the input directory."""

    top_path: Path
    top_score: float
    # Top-N (filename, score) pairs, best first.
    top5: list
    stats: ScanResult


class ContentFeatureStore:
    """On-disk cache mapping image content (SHA-256) to feature vectors.

    Entries are keyed by the SHA-256 of the file contents rather than by path,
    so images may be moved, renamed, or re-added without losing or
    re-extracting their features. A scan never removes entries (unlike
    ``FeatureStore``), because input images are expected to be moved around.
    """

    def __init__(self, store_dir, extractor=None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.store_dir / CONTENT_FEATURES_FILE
        self.index_path = self.store_dir / CONTENT_INDEX_FILE
        self._extractor = extractor
        self._keys: list[str] = []
        self._features: dict[str, np.ndarray] = {}
        self._index: dict[str, dict] = {}
        self._load()

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not (self.features_path.exists() and self.index_path.exists()):
            return
        with np.load(self.features_path) as data:
            keys = [str(k) for k in data["keys"]]
            matrix = data["features"]
        with open(self.index_path, "r", encoding="utf-8") as f:
            self._index = json.load(f)
        self._keys = keys
        self._features = {key: matrix[i] for i, key in enumerate(keys)}

    def save(self) -> None:
        if not self._keys:
            self.features_path.unlink(missing_ok=True)
            self.index_path.unlink(missing_ok=True)
            return
        matrix = np.stack([self._features[k] for k in self._keys]).astype(np.float32)
        np.savez_compressed(
            self.features_path, keys=np.array(self._keys), features=matrix
        )
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, sort_keys=True)

    # -- accessors -----------------------------------------------------

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def feature(self, key: str) -> np.ndarray:
        return self._features[key]

    def has(self, key: str) -> bool:
        return key in self._features

    # -- extraction ----------------------------------------------------

    def _extractor_fn(self):
        if self._extractor is None:
            self._extractor = extract_dense_features
        return self._extractor

    def ensure_features(self, images_dir) -> ScanResult:
        """Extract features for images in ``images_dir`` not already cached.

        Images are identified by their SHA-256 content hash, so an image that
        is moved or renamed but whose content is unchanged is never
        re-extracted. Existing entries are never removed.
        """
        images_dir = Path(images_dir)
        paths = scan_image_directory(images_dir)
        extract = self._extractor_fn()
        result = ScanResult(
            scanned=len(paths), new=0, cached=0, errors=0, path_to_hash={}
        )
        for path in paths:
            try:
                digest = file_fingerprint(path)
            except OSError:
                result.errors += 1
                continue
            result.path_to_hash[path] = digest
            if digest in self._features:
                result.cached += 1
                entry = self._index.setdefault(digest, {})
                entry["rel"] = path.name
                continue
            result.new += 1
            try:
                img = load_best_frame(path)
                self._features[digest] = np.asarray(extract(img), dtype=np.float32)
            except Exception:
                result.errors += 1
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self._index[digest] = {"rel": path.name, "size": size}

        self._keys = sorted(self._features)
        self.save()
        return result


def get_profile(profile_path, expected_dim: int = FEATURE_DIM) -> LinUCBUser:
    """Return the saved preference profile, or a clean one if absent/mismatched."""
    profile = load_preference_profile(profile_path, expected_dim=expected_dim)
    if profile is None:
        profile = LinUCBUser(expected_dim)
    return profile


def run_manual_inference(
    input_dir, store: ContentFeatureStore, profile: LinUCBUser, top_n: int = 5
) -> InferenceResult | None:
    """Score every image in ``input_dir`` with ``profile`` and return the top
    image plus the top-N scores.

    Returns ``None`` when the directory holds no images. Only reads
    ``profile``; never updates it, so the bandit loop is left untouched.
    """
    scan = store.ensure_features(input_dir)
    if not scan.path_to_hash:
        return None

    hash_to_path = {digest: path for path, digest in scan.path_to_hash.items()}
    hashes = list(hash_to_path)
    vectors = np.stack([store.feature(digest) for digest in hashes])
    scores = profile.score_images(vectors)
    order = np.argsort(scores)[::-1]

    top5 = [
        (Path(hash_to_path[hashes[i]]).name, float(scores[i]))
        for i in order[:top_n]
    ]
    top_i = int(order[0])
    return InferenceResult(
        top_path=hash_to_path[hashes[top_i]],
        top_score=float(scores[top_i]),
        top5=top5,
        stats=scan,
    )


__all__ = [
    "CONTENT_FEATURES_FILE",
    "CONTENT_INDEX_FILE",
    "ContentFeatureStore",
    "InferenceResult",
    "ScanResult",
    "get_profile",
    "run_manual_inference",
]
