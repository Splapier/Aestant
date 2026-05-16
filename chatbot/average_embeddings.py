"""Winners/Losers average embeddings with persistent state.

This module handles:
- Maintaining separate running averages for winners and losers
- Adding entries (image+entry_id combinations) to pools
- Computing averages dynamically or using pre-computed values
- Persisting state to disk for incremental updates

Usage:
    >>> from chatbot.average_embeddings import add_winner_entry, add_loser_entry
    >>> add_winner_entry("image_a", "rect_0", image_emb, tag_emb)
    >>> add_loser_entry("image_b", None, image_emb, None)  # None = full image
    >>> from chatbot.average_embeddings import compute_image_embedding_avg
    >>> avg = compute_image_embedding_avg("winners", use_rectangles=True)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from chatbot.tagging_engine import DATASET_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "average_embeddings.json"


def _empty_pool():
    """Return an empty pool structure."""
    return {
        "image_embedding_avg": None,
        "tag_embedding_avg": None,
        "image_count": 0,
        "tag_count": 0,
        "entries": [],
    }


def load_state():
    """Load winners/losers embeddings state from disk.

    Returns:
        State dictionary with winners, losers, and last_updated.
    """
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        if "winners" not in state:
            state = {
                "winners": _empty_pool(),
                "losers": _empty_pool(),
                "last_updated": None,
            }
        if "entries" not in state.get("winners", {}):
            state["winners"]["entries"] = []
        if "entries" not in state.get("losers", {}):
            state["losers"]["entries"] = []
        return state

    return {
        "winners": _empty_pool(),
        "losers": _empty_pool(),
        "last_updated": None,
    }


def save_state(state):
    """Save winners/losers embeddings state to disk."""
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def update_running_average(current_avg, count, new_vector):
    """Update running average with a new vector.

    Args:
        current_avg: Current average vector (list or None).
        count: Number of vectors used so far.
        new_vector: New vector to add (list of floats).

    Returns:
        Updated average vector as list.
    """
    new_vector = np.array(new_vector)
    if current_avg is None:
        return new_vector.tolist()
    current_avg = np.array(current_avg)
    updated = (current_avg * count + new_vector) / (count + 1)
    return updated.tolist()


def _get_entry_embedding(image_name: str, entry_id: str | None) -> list[float] | None:
    """Get embedding for a specific image entry.

    Args:
        image_name: Name of image (without extension).
        entry_id: Entry ID (e.g., "rect_0"), or None for full image.

    Returns:
        Embedding list, or None if not found.

    Raises:
        FileNotFoundError: If embedding file doesn't exist.
        ValueError: If entry_id not found.
    """
    from chatbot.embedding_engine import get_entry_embedding, load_embedding_file

    if entry_id is None:
        data = load_embedding_file(image_name)
        return data.get("image_embedding")

    return get_entry_embedding(image_name, entry_id)


def add_winner_entry(
    image_name: str,
    entry_id: str | None,
    emb: list[float] | None = None,
    tag_emb: list[float] | None = None,
):
    """Add an entry to the winners pool.

    Args:
        image_name: Name of image (without extension).
        entry_id: Entry ID (e.g., "rect_0"), or None for full image.
        emb: Embedding to add. If None, fetches from embedding file.
        tag_emb: Tag embedding to add, or None.
    """
    state = load_state()

    if emb is None:
        emb = _get_entry_embedding(image_name, entry_id)

    if emb is not None:
        state["winners"]["image_embedding_avg"] = update_running_average(
            state["winners"]["image_embedding_avg"],
            state["winners"]["image_count"],
            emb,
        )
        state["winners"]["image_count"] += 1

    if tag_emb is not None:
        state["winners"]["tag_embedding_avg"] = update_running_average(
            state["winners"]["tag_embedding_avg"],
            state["winners"]["tag_count"],
            tag_emb,
        )
        state["winners"]["tag_count"] += 1

    state["winners"]["entries"].append(
        {
            "image_name": image_name,
            "entry_id": entry_id,
        }
    )

    save_state(state)


def add_loser_entry(
    image_name: str,
    entry_id: str | None,
    emb: list[float] | None = None,
    tag_emb: list[float] | None = None,
):
    """Add an entry to the losers pool.

    Args:
        image_name: Name of image (without extension).
        entry_id: Entry ID (e.g., "rect_0"), or None for full image.
        emb: Embedding to add. If None, fetches from embedding file.
        tag_emb: Tag embedding to add, or None.
    """
    state = load_state()

    if emb is None:
        emb = _get_entry_embedding(image_name, entry_id)

    if emb is not None:
        state["losers"]["image_embedding_avg"] = update_running_average(
            state["losers"]["image_embedding_avg"],
            state["losers"]["image_count"],
            emb,
        )
        state["losers"]["image_count"] += 1

    if tag_emb is not None:
        state["losers"]["tag_embedding_avg"] = update_running_average(
            state["losers"]["tag_embedding_avg"],
            state["losers"]["tag_count"],
            tag_emb,
        )
        state["losers"]["tag_count"] += 1

    state["losers"]["entries"].append(
        {
            "image_name": image_name,
            "entry_id": entry_id,
        }
    )

    save_state(state)


def compute_image_embedding_avg(
    pool_name: str, use_rectangles: bool = False
) -> list[float] | None:
    """Compute image embedding average for a pool.

    Args:
        pool_name: Either "winners" or "losers".
        use_rectangles: If True, recompute average from entries using their stored embeddings.
                       If False, return pre-computed average.

    Returns:
        Image embedding average, or None if pool is empty.

    Raises:
        ValueError: If pool_name is invalid.
    """
    state = load_state()

    if pool_name not in ["winners", "losers"]:
        raise ValueError(
            f"Invalid pool_name: {pool_name}. Must be 'winners' or 'losers'."
        )

    pool = state[pool_name]

    if not use_rectangles:
        return pool.get("image_embedding_avg")

    entries = pool.get("entries", [])
    if not entries:
        return pool.get("image_embedding_avg")

    embeddings = []
    for entry in entries:
        image_name = entry["image_name"]
        entry_id = entry["entry_id"]

        emb = _get_entry_embedding(image_name, entry_id)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return pool.get("image_embedding_avg")

    avg = np.mean(embeddings, axis=0)
    return avg.tolist()


def get_pool_entries(pool_name: str) -> list[dict]:
    """Get entries list for a pool.

    Args:
        pool_name: Either "winners" or "losers".

    Returns:
        List of entry dicts with image_name and entry_id.

    Raises:
        ValueError: If pool_name is invalid.
    """
    state = load_state()

    if pool_name not in ["winners", "losers"]:
        raise ValueError(
            f"Invalid pool_name: {pool_name}. Must be 'winners' or 'losers'."
        )

    return state[pool_name].get("entries", [])


def get_pool_status() -> dict:
    """Get current pool status for both winners and losers.

    Returns:
        Dict with winners and losers pool info including entry counts.
    """
    state = load_state()

    winners = state.get("winners", {})
    losers = state.get("losers", {})

    return {
        "winners": {
            "image_count": winners.get("image_count", 0),
            "tag_count": winners.get("tag_count", 0),
            "entry_count": len(winners.get("entries", [])),
            "entries": winners.get("entries", []),
        },
        "losers": {
            "image_count": losers.get("image_count", 0),
            "tag_count": losers.get("tag_count", 0),
            "entry_count": len(losers.get("entries", [])),
            "entries": losers.get("entries", []),
        },
        "last_updated": state.get("last_updated"),
    }


__all__ = [
    "load_state",
    "save_state",
    "update_running_average",
    "add_winner_entry",
    "add_loser_entry",
    "compute_image_embedding_avg",
    "get_pool_entries",
    "get_pool_status",
]
