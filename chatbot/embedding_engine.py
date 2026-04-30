"""Embedding engine for creating image and tag embeddings.

This module handles:
- Loading CLIP model for embedding generation
- Creating image embeddings from files
- Creating tag embeddings from tag dictionaries
- Saving embeddings to JSON files in dataset/ directory

Usage:
    >>> from chatbot.embedding_engine import embed_image, embed_tags, save_embeddings
    >>> img_emb = embed_image("/path/to/image.jpg")
    >>> tag_emb = embed_tags({"style": "casual"})
    >>> save_embeddings("/path/to/image.jpg", img_emb, tag_emb)
"""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from chatbot.tagging_engine import DATASET_DIR, ensure_dataset_dir

EMBEDDING_MODEL_NAME = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Get cached CLIP embedding model.

    Returns:
        Tuple of (model, processor)
    """
    model = CLIPModel.from_pretrained(EMBEDDING_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
    return model, processor


def embed_image(image_path: str) -> list[float]:
    """Create embedding for an image file.

    Args:
        image_path: Path to the image file.

    Returns:
        Embedding as list of floats.

    Raises:
        FileNotFoundError: If image file doesn't exist.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, processor = get_embedding_model()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with model.eval():
        features = model.get_image_features(**inputs)

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
    inputs = processor(text=text, return_tensors="pt", padding=True)

    with model.eval():
        features = model.get_text_features(**inputs)

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
) -> str:
    """Save embeddings to JSON file in dataset directory.

    Args:
        image_path: Path to the original image.
        image_embedding: Image embedding as list of floats, or None.
        tag_embedding: Tag embedding as list of floats, or None.

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
        "created_at": datetime.now().isoformat(),
        "model": EMBEDDING_MODEL_NAME,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return str(json_path)


__all__ = [
    "embed_image",
    "embed_tags",
    "save_embeddings",
    "serialize_tags",
    "create_embedding_filename",
    "get_embedding_model",
]
