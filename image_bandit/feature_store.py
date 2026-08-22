"""Dense feature extraction and on-disk caching for the bandit recommender.

Extracts DINOv2 multi-layer dense features (as in dino.py) from every image
in the images directory and caches them in a compressed numpy archive. A JSON
index maps each feature vector to its source image and stores a content
fingerprint, so already-processed images are never re-extracted.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

DINOV2_MODEL = "facebook/dinov2-small"
FEATURE_DIM = 1152  # 3 tapped layers (3, 7, 11) x 384 (dinov2-small hidden size)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FEATURES_FILE = "dense_features.npz"
INDEX_FILE = "feature_index.json"

_BACKBONE = None


def scan_image_directory(images_dir) -> list[Path]:
    """Return sorted absolute paths of all supported images in a directory."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return []
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_fingerprint(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def get_backbone(model_name: str = DINOV2_MODEL) -> tuple:
    """Lazily load the frozen DINOv2 processor/model once per process."""
    global _BACKBONE
    if _BACKBONE is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        _BACKBONE = (processor, model, device, model_name)
    return _BACKBONE


def extract_dense_features(image, backbone: tuple | None = None) -> np.ndarray:
    """Extract the 1152-d dense DINOv2 feature vector for a PIL image.

    Concatenates mean-pooled features from an early, middle, and final
    transformer layer (same as extract_dense_features in dino.py).
    """
    import torch

    if backbone is None:
        backbone = get_backbone()
    processor, model, device, _ = backbone
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    early_features = hidden_states[3].mean(dim=1)
    mid_features = hidden_states[7].mean(dim=1)
    late_features = hidden_states[11].mean(dim=1)
    combined = torch.cat([early_features, mid_features, late_features], dim=1)
    return combined.squeeze().cpu().numpy().astype(np.float32)


class FeatureStore:
    """On-disk cache mapping image keys (relative paths) to feature vectors."""

    def __init__(self, store_dir, extractor=None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.store_dir / FEATURES_FILE
        self.index_path = self.store_dir / INDEX_FILE
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

    def rel_path(self, key: str) -> str:
        return self._index[key]["rel"]

    def random_batch(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> list[str]:
        """Return a random sample of image keys (without replacement)."""
        if not self._keys:
            return []
        rng = rng if rng is not None else np.random.default_rng()
        count = min(batch_size, len(self._keys))
        picked = rng.choice(len(self._keys), size=count, replace=False)
        return [self._keys[i] for i in picked]

    # -- extraction ----------------------------------------------------

    def _extractor_fn(self):
        if self._extractor is None:
            self._extractor = extract_dense_features
        return self._extractor

    def ensure_features(self, images_dir) -> dict:
        """Extract features for new/changed images and drop deleted ones.

        Images whose content fingerprint (SHA-256) matches a cached entry are
        skipped, so already-processed images are never re-extracted.
        """
        images_dir = Path(images_dir)
        paths = scan_image_directory(images_dir)
        extract = self._extractor_fn()
        stats = {
            "scanned": len(paths),
            "new": 0,
            "reprocessed": 0,
            "cached": 0,
            "removed": 0,
            "errors": 0,
        }
        seen: set[str] = set()
        for path in paths:
            key = str(path.relative_to(images_dir))
            seen.add(key)
            try:
                st = path.stat()
            except OSError:
                stats["errors"] += 1
                continue
            entry = self._index.get(key)
            cached = False
            digest = None
            if entry is not None and key in self._features:
                if (
                    entry.get("size") == st.st_size
                    and abs(float(entry.get("mtime", -1)) - st.st_mtime) < 1e-3
                ):
                    cached = True
                else:
                    digest = file_fingerprint(path)
                    if entry.get("sha256") == digest:
                        entry["size"] = st.st_size
                        entry["mtime"] = st.st_mtime
                        cached = True
            if cached:
                stats["cached"] += 1
                continue
            if entry is None:
                stats["new"] += 1
            else:
                stats["reprocessed"] += 1
            try:
                img = Image.open(path).convert("RGB")
                self._features[key] = np.asarray(extract(img), dtype=np.float32)
            except Exception:
                stats["errors"] += 1
                continue
            if digest is None:
                digest = file_fingerprint(path)
            self._index[key] = {
                "rel": key,
                "sha256": digest,
                "size": st.st_size,
                "mtime": st.st_mtime,
            }

        removed = [k for k in self._index if k not in seen]
        for k in removed:
            del self._index[k]
            self._features.pop(k, None)
            stats["removed"] += 1

        self._keys = sorted(self._features)
        self.save()
        return stats
