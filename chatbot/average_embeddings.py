"""Winners/Losers average embeddings with persistent state.

This module handles:
- Maintaining separate running averages for winners and losers
- Adding embeddings to winners or losers pools with rectangle metadata
- Persisting state to disk for incremental updates

Usage:
    >>> from chatbot.average_embeddings import add_winner, add_loser, load_state
    >>> add_winner(image_emb, tag_emb, rectangles=[{"x1": 10, "y1": 20, "x2": 100, "y2": 200}])
    >>> state = load_state()
    >>> print(state["winners"]["last_rectangles"])
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "average_embeddings.json"


def _empty_pool():
    """Return an empty pool structure."""
    return {
        "image_embedding_avg": None,
        "tag_embedding_avg": None,
        "image_count": 0,
        "tag_count": 0,
        "last_rectangles": None,
    }


def load_state():
    """Load winners/losers embeddings state from disk.

    Returns:
        State dictionary with winners, losers, and last_updated.
    """
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Migrate old format or ensure new structure
        if "winners" not in state:
            state = {
                "winners": _empty_pool(),
                "losers": _empty_pool(),
                "last_updated": None,
            }
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


def add_winner(image_emb, tag_emb=None, rectangles=None):
    """Add a winner's embeddings to the winners pool.

    Args:
        image_emb: Image embedding as list of floats.
        tag_emb: Tag embedding as list of floats, or None.
        rectangles: Optional list of rectangle dicts used for cropping the image.
                   Stored with the pool for attention-based comparison.
    """
    state = load_state()

    if image_emb is not None:
        state["winners"]["image_embedding_avg"] = update_running_average(
            state["winners"]["image_embedding_avg"],
            state["winners"]["image_count"],
            image_emb,
        )
        state["winners"]["image_count"] += 1

    if tag_emb is not None:
        state["winners"]["tag_embedding_avg"] = update_running_average(
            state["winners"]["tag_embedding_avg"],
            state["winners"]["tag_count"],
            tag_emb,
        )
        state["winners"]["tag_count"] += 1

    state["winners"]["last_rectangles"] = rectangles

    save_state(state)


def add_loser(image_emb, tag_emb=None, rectangles=None):
    """Add a loser's embeddings to the losers pool.

    Args:
        image_emb: Image embedding as list of floats.
        tag_emb: Tag embedding as list of floats, or None.
        rectangles: Optional list of rectangle dicts used for cropping the image.
                   Stored with the pool for attention-based comparison.
    """
    state = load_state()

    if image_emb is not None:
        state["losers"]["image_embedding_avg"] = update_running_average(
            state["losers"]["image_embedding_avg"],
            state["losers"]["image_count"],
            image_emb,
        )
        state["losers"]["image_count"] += 1

    if tag_emb is not None:
        state["losers"]["tag_embedding_avg"] = update_running_average(
            state["losers"]["tag_embedding_avg"],
            state["losers"]["tag_count"],
            tag_emb,
        )
        state["losers"]["tag_count"] += 1

    state["losers"]["last_rectangles"] = rectangles

    save_state(state)
