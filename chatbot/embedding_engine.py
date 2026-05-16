"""Embedding engine for creating image and tag embeddings.

This module handles:
- Loading CLIP model for embedding generation
- Creating image embeddings from files
- Creating tag embeddings from tag dictionaries
- Saving embeddings to JSON files in dataset/ directory
- Managing multiple entries (rectangle-based regions) per image

Usage:
    >>> from chatbot.embedding_engine import embed_image, embed_tags, save_embeddings
    >>> img_emb = embed_image("/path/to/image.jpg")
    >>> tag_emb = embed_tags({"style": "casual"})
    >>> save_embeddings("/path/to/image.jpg", img_emb, tag_emb)
    >>> from chatbot.embedding_engine import add_entry_to_embedding, get_entry_embedding
    >>> entry_id = add_entry_to_embedding("image", [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}])
    >>> entry_emb = get_entry_embedding("image", entry_id)
"""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from chatbot.tagging_engine import DATASET_DIR, ensure_dataset_dir

EMBEDDING_MODEL_NAME = "openai/clip-vit-base-patch32"

RECT_MATCH_TOLERANCE = 5


@lru_cache(maxsize=1)
def get_embedding_model():
    """Get cached CLIP embedding model.

    Returns:
        Tuple of (model, processor)
    """
    model = CLIPModel.from_pretrained(EMBEDDING_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
    return model, processor


def embed_image(
    image_path: str | Path,
    rectangles: list[dict] | None = None,
) -> list[float]:
    """Create embedding for an image file, optionally from a cropped region.

    Args:
        image_path: Path to the image file.
        rectangles: Optional list of rectangle dicts for cropping to specific region.
                   If provided and contains exactly one rectangle, the image is
                   cropped to that bounding box before embedding.

    Returns:
        Embedding as list of floats.

    Raises:
        FileNotFoundError: If image file doesn't exist.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, processor = get_embedding_model()

    from chatbot.image_modules.image_loading import load_image_cropped

    image_array = load_image_cropped(image_path, rectangles)

    pil_img = Image.fromarray(image_array)
    inputs = processor(images=pil_img, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        features = outputs.pooler_output

    embedding = features.detach().numpy()[0]

    return embedding.tolist()


def embed_from_array(
    image_array: np.ndarray,
    rectangles: list[dict] | None = None,
) -> list[float]:
    """Create embedding from a numpy image array, optionally cropped.

    Args:
        image_array: RGB numpy array (H x W x 3).
        rectangles: Optional list of rectangle dicts for cropping.

    Returns:
        Embedding as list of floats.
    """
    from chatbot.tagging_engine import crop_to_bounding_box

    if rectangles and len(rectangles) == 1:
        image_array = crop_to_bounding_box(image_array, rectangles)

    model, processor = get_embedding_model()

    pil_img = Image.fromarray(image_array)
    inputs = processor(images=pil_img, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        features = outputs.pooler_output

    embedding = features.detach().numpy()[0]

    return embedding.tolist()


def serialize_tags(tags: dict) -> str:
    """Serialize tags dictionary to string for embedding.

    Args:
        tags: Tags dictionary.

    Returns:
        Serialized string representation.
    """
    if not tags:
        return ""

    def _flatten(d, prefix=""):
        items = []
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key))
            else:
                items.append(f"{new_key}: {v}")
        return items

    lines = _flatten(tags)
    return ", ".join(lines)


def embed_tags(tags: dict) -> list[float]:
    """Create embedding for tags.

    Args:
        tags: Tags dictionary.

    Returns:
        Embedding as list of floats.
    """
    model, processor = get_embedding_model()

    text = serialize_tags(tags)
    inputs = processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )

    model.eval()
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        features = outputs.pooler_output

    embedding = features.detach().numpy()[0]

    return embedding.tolist()


def create_embedding_filename(image_path: str) -> str:
    """Create embedding filename from image path.

    Args:
        image_path: Path to the image file.

    Returns:
        Filename like "image.embedding.json"
    """
    stem = Path(image_path).stem
    return f"{stem}.embedding.json"


def save_embeddings(
    image_path: str,
    image_embedding: list[float] | None,
    tag_embedding: list[float] | None,
    rectangles: list[dict] | None = None,
) -> str:
    """Save embeddings to JSON file in dataset directory.

    Args:
        image_path: Path to the original image.
        image_embedding: Image embedding as list of floats, or None.
        tag_embedding: Tag embedding as list of floats, or None.
        rectangles: Optional list of rectangle dicts that were used for cropping.
                   Stored with the embedding to enable attention-based comparison.

    Returns:
        Path to the saved JSON file.
    """
    ensure_dataset_dir()

    filename = create_embedding_filename(image_path)
    json_path = DATASET_DIR / filename

    data = {
        "image_path": image_path,
        "image_embedding": image_embedding,
        "tag_embedding": tag_embedding,
        "region_rects": rectangles,
        "created_at": datetime.now().isoformat(),
        "model": EMBEDDING_MODEL_NAME,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return str(json_path)


def batch_embed_images(input_dir=None):
    """Batch embed all images in input directory that don't have embeddings yet.

    Only creates image embeddings (not tag embeddings). Skips images that
    already have image_embeddings in their .embedding.json file.

    Args:
        input_dir: Directory containing images. Defaults to input/.

    Returns:
        Number of newly embedded images.
    """
    if input_dir is None:
        input_dir = Path(__file__).resolve().parent.parent / "input"
    else:
        input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ensure_dataset_dir()

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    image_files = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    embedded_count = 0
    skipped_count = 0

    for img_path in sorted(image_files):
        emb_filename = create_embedding_filename(str(img_path))
        emb_path = DATASET_DIR / emb_filename

        # Check if already has image embedding
        if emb_path.exists():
            with open(emb_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if existing_data.get("image_embedding") is not None:
                skipped_count += 1
                continue

        # Embed and save
        img_emb = embed_image(str(img_path))
        save_embeddings(str(img_path), img_emb, None)
        embedded_count += 1

        if embedded_count % 10 == 0:
            print(f"Embedded {embedded_count} images so far...")

    print(
        f"Done. Newly embedded: {embedded_count}, Skipped (already embedded): {skipped_count}"
    )
    return embedded_count


def _rects_match(
    rects1: list[dict] | None,
    rects2: list[dict] | None,
    tolerance: int = RECT_MATCH_TOLERANCE,
) -> bool:
    """Check if two rectangle sets are approximately equal.

    Args:
        rects1: First rectangle set.
        rects2: Second rectangle set.
        tolerance: Pixel tolerance for coordinate comparison.

    Returns:
        True if rectangles match within tolerance.
    """
    if rects1 is None and rects2 is None:
        return True
    if rects1 is None or rects2 is None:
        return False
    if len(rects1) != len(rects2):
        return False

    for r1, r2 in zip(rects1, rects2):
        if (
            abs(r1.get("x1", 0) - r2.get("x1", 0)) > tolerance
            or abs(r1.get("y1", 0) - r2.get("y1", 0)) > tolerance
            or abs(r1.get("x2", 0) - r2.get("x2", 0)) > tolerance
            or abs(r1.get("y2", 0) - r2.get("y2", 0)) > tolerance
        ):
            return False
    return True


def _find_matching_entry(entries: list[dict], rectangles: list[dict]) -> str | None:
    """Find entry ID matching the given rectangles.

    Args:
        entries: List of entry dicts from embedding file.
        rectangles: Rectangles to match.

    Returns:
        Entry ID if found, None otherwise.
    """
    if not entries or not rectangles:
        return None

    for entry in entries:
        if _rects_match(entry.get("region_rects"), rectangles):
            return entry["id"]
    return None


def load_embedding_file(image_name: str) -> dict:
    """Load embedding data from a JSON file.

    Args:
        image_name: Name of image (without extension).

    Returns:
        Dictionary with all embedding data.
    """
    emb_path = DATASET_DIR / f"{image_name}.embedding.json"
    with open(emb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embedding_file(image_name: str, data: dict) -> str:
    """Save embedding data to JSON file.

    Args:
        image_name: Name of image (without extension).
        data: Dictionary to save.

    Returns:
        Path to the saved file.
    """
    emb_path = DATASET_DIR / f"{image_name}.embedding.json"
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(emb_path)


def _get_next_rect_index(entries: list[dict]) -> int:
    """Get the next rect_index for a new entry.

    Args:
        entries: List of existing entries.

    Returns:
        Next available rect_index.
    """
    if not entries:
        return 0
    max_index = max(entry.get("rect_index", -1) for entry in entries)
    return max_index + 1


def add_entry_to_embedding(image_name: str, rectangles: list[dict]) -> str | None:
    """Add a new rectangle entry to an image's embedding file or return existing.

    If matching rectangles already exist as an entry, returns the existing entry_id.
    Otherwise creates a new entry with cropped embedding, saves, and returns new entry_id.

    Args:
        image_name: Name of image (without extension).
        rectangles: List of rectangle dicts defining the region.
                   If empty/None, returns None (full image only).

    Returns:
        Entry ID (e.g., "rect_0") if rectangles provided, None if no rectangles.

    Raises:
        FileNotFoundError: If embedding file doesn't exist.
    """
    if not rectangles:
        return None

    data = load_embedding_file(image_name)
    entries = data.get("entries", [])

    existing_id = _find_matching_entry(entries, rectangles)
    if existing_id:
        return existing_id

    from chatbot.image_modules.image_loading import load_image_cropped
    from chatbot.tagging_engine import crop_to_bounding_box

    image_path = data.get("image_path")
    if not image_path:
        raise ValueError(f"No image_path in embedding for {image_name}")

    img_array = load_image_cropped(image_path, rectangles)
    cropped = crop_to_bounding_box(img_array, rectangles)

    cropped_emb = embed_from_array(cropped)

    rect_index = _get_next_rect_index(entries)
    entry_id = f"rect_{rect_index}"

    new_entry = {
        "id": entry_id,
        "rect_index": rect_index,
        "region_rects": rectangles,
        "image_embedding": cropped_emb,
        "tag_embedding": None,
    }

    entries.append(new_entry)
    data["entries"] = entries

    save_embedding_file(image_name, data)

    return entry_id


def get_entry_embedding(image_name: str, entry_id: str) -> list[float] | None:
    """Get embedding for a specific entry.

    Args:
        image_name: Name of image (without extension).
        entry_id: Entry ID (e.g., "rect_0").

    Returns:
        Entry's image embedding, or None if not found.

    Raises:
        FileNotFoundError: If embedding file doesn't exist.
        ValueError: If entry_id not found in the image.
    """
    data = load_embedding_file(image_name)
    entries = data.get("entries", [])

    for entry in entries:
        if entry["id"] == entry_id:
            return entry.get("image_embedding")

    raise ValueError(f"Entry {entry_id} not found in {image_name}")


def get_all_entries(image_name: str) -> list[dict]:
    """Get all entries for an image.

    Args:
        image_name: Name of image (without extension).

    Returns:
        List of entry dicts, or empty list if none.

    Raises:
        FileNotFoundError: If embedding file doesn't exist.
    """
    data = load_embedding_file(image_name)
    return data.get("entries", [])


def get_latest_entry_id(image_name: str) -> str | None:
    """Get the entry ID of the most recently added entry.

    Args:
        image_name: Name of image (without extension).

    Returns:
        Entry ID of latest entry, or None if no entries exist.

    Raises:
        FileNotFoundError: If embedding file doesn't exist.
    """
    entries = get_all_entries(image_name)
    if not entries:
        return None

    sorted_entries = sorted(entries, key=lambda e: e.get("rect_index", -1))
    return sorted_entries[-1]["id"]


def get_entry_by_rectangles(image_name: str, rectangles: list[dict]) -> dict | None:
    """Get entry matching given rectangles.

    Args:
        image_name: Name of image (without extension).
        rectangles: Rectangles to match.

    Returns:
        Entry dict if found, None otherwise.

    Raises:
        FileNotFoundError: If embedding file doesn't exist.
    """
    entries = get_all_entries(image_name)

    for entry in entries:
        if _rects_match(entry.get("region_rects"), rectangles):
            return entry
    return None


__all__ = [
    "embed_image",
    "embed_from_array",
    "embed_tags",
    "save_embeddings",
    "serialize_tags",
    "create_embedding_filename",
    "get_embedding_model",
    "batch_embed_images",
    "load_embedding_file",
    "save_embedding_file",
    "add_entry_to_embedding",
    "get_entry_embedding",
    "get_all_entries",
    "get_latest_entry_id",
    "get_entry_by_rectangles",
]
