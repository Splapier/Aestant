"""Similarity search and winner selection for embeddings.

This module handles:
- Cosine similarity calculations
- Finding the best candidate based on winners/losers pools
- Primary scoring: cosine similarity to average embeddings
- Secondary scoring (optional): sum of similarities to individual entries

Usage:
    >>> from chatbot.similarity_search import find_winner
    >>> winner = find_winner(dataset_dir, winners_pool, losers_pool)
    >>> winner = find_winner(dataset_dir, winners_pool, losers_pool, use_entry_scoring=True)
"""

import json
from pathlib import Path

import numpy as np

from chatbot.tagging_engine import DATASET_DIR


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector (list or np.array).
        b: Second vector (list or np.array).

    Returns:
        Cosine similarity as float between -1 and 1.
    """
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_embedding_file(filepath):
    """Load embedding data from a JSON file.

    Args:
        filepath: Path to .embedding.json file.

    Returns:
        Dictionary with image_embedding, tag_embedding, entries, etc.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_entry_embedding(image_name: str, entry_id: str | None) -> list[float] | None:
    """Get embedding for a specific image entry.

    Args:
        image_name: Name of image (without extension).
        entry_id: Entry ID (e.g., "rect_0"), or None for full image.

    Returns:
        Embedding list, or None if not found.
    """
    from chatbot.embedding_engine import (
        get_entry_embedding,
        load_embedding_file as load_emb,
    )

    if entry_id is None:
        data = load_emb(image_name)
        return data.get("image_embedding")

    return get_entry_embedding(image_name, entry_id)


def _compute_primary_score(
    img_emb, tag_emb, win_img_avg, win_tag_avg, lose_img_avg, lose_tag_avg
):
    """Compute primary score based on cosine similarity to average embeddings.

    Args:
        img_emb: Candidate image embedding.
        tag_emb: Candidate tag embedding.
        win_img_avg: Winners average image embedding.
        win_tag_avg: Winners average tag embedding.
        lose_img_avg: Losers average image embedding.
        lose_tag_avg: Losers average tag embedding.

    Returns:
        Primary score as float.
    """
    score = 0.0

    if img_emb is not None and win_img_avg is not None:
        score += cosine_similarity(win_img_avg, img_emb)
    if img_emb is not None and lose_img_avg is not None:
        score -= cosine_similarity(lose_img_avg, img_emb)
    if tag_emb is not None and win_tag_avg is not None:
        score += cosine_similarity(win_tag_avg, tag_emb)
    if tag_emb is not None and lose_tag_avg is not None:
        score -= cosine_similarity(lose_tag_avg, tag_emb)

    return score


def _compute_secondary_score(img_emb, tag_emb, winners_entries, losers_entries):
    """Compute secondary score based on sum of similarities to individual entries.

    Each entry contributes its own similarity to the candidate, summed separately
    for winners and losers.

    Args:
        img_emb: Candidate image embedding.
        tag_emb: Candidate tag embedding.
        winners_entries: List of winner entry dicts.
        losers_entries: List of loser entry dicts.

    Returns:
        Secondary score as float (winner_sum - loser_sum).
    """
    win_sum = 0.0
    lose_sum = 0.0

    for entry in winners_entries:
        entry_emb = _get_entry_embedding(entry["image_name"], entry["entry_id"])
        if entry_emb is not None and img_emb is not None:
            win_sum += cosine_similarity(img_emb, entry_emb)

    for entry in losers_entries:
        entry_emb = _get_entry_embedding(entry["image_name"], entry["entry_id"])
        if entry_emb is not None and img_emb is not None:
            lose_sum += cosine_similarity(img_emb, entry_emb)

    return win_sum - lose_sum


def find_winner(
    dataset_dir,
    winners_pool,
    losers_pool,
    use_entry_scoring: bool = False,
):
    """Find the winner image based on similarity to winners/losers pools.

    Primary scoring: Score = sim_to_winners_avg - sim_to_losers_avg
    Secondary scoring (optional): Adds sum of similarities to individual entries.

    Args:
        dataset_dir: Directory containing .embedding.json files.
        winners_pool: Dict with entries list and image_embedding_avg.
        losers_pool: Dict with entries list and image_embedding_avg.
        use_entry_scoring: If True, adds secondary entry-by-entry scoring.

    Returns:
        Dictionary with winner info: filename, scores.
    """
    dataset_dir = Path(dataset_dir)

    win_img_avg = winners_pool.get("image_embedding_avg")
    win_tag_avg = winners_pool.get("tag_embedding_avg")
    lose_img_avg = losers_pool.get("image_embedding_avg")
    lose_tag_avg = losers_pool.get("tag_embedding_avg")

    winners_entries = winners_pool.get("entries", [])
    losers_entries = losers_pool.get("entries", [])

    best_file = None
    best_score = -float("inf")

    for emb_file in sorted(dataset_dir.glob("*.embedding.json")):
        data = load_embedding_file(emb_file)

        img_emb = data.get("image_embedding")
        tag_emb = data.get("tag_embedding")

        if img_emb is None:
            continue

        primary = _compute_primary_score(
            img_emb, tag_emb, win_img_avg, win_tag_avg, lose_img_avg, lose_tag_avg
        )

        if use_entry_scoring and (winners_entries or losers_entries):
            secondary = _compute_secondary_score(
                img_emb, tag_emb, winners_entries, losers_entries
            )
            score = primary + secondary
        else:
            score = primary

        if score > best_score:
            best_score = score
            best_file = emb_file.name

    return {
        "winner": best_file,
        "score": best_score,
    }


def find_winner_with_entries(
    dataset_dir,
    winners_entries: list[dict],
    losers_entries: list[dict],
    use_entry_scoring: bool = False,
):
    """Find winner using explicit entries instead of pools.

    Args:
        dataset_dir: Directory containing .embedding.json files.
        winners_entries: List of winner entry dicts with image_name and entry_id.
        losers_entries: List of loser entry dicts with image_name and entry_id.
        use_entry_scoring: If True, use entry-by-entry scoring.

    Returns:
        Dictionary with winner info: filename, score.
    """
    from chatbot.average_embeddings import compute_image_embedding_avg

    dataset_dir = Path(dataset_dir)

    winners_pool = {
        "image_embedding_avg": compute_image_embedding_avg(
            "winners", use_rectangles=True
        ),
        "tag_embedding_avg": None,
        "entries": winners_entries,
    }
    losers_pool = {
        "image_embedding_avg": compute_image_embedding_avg(
            "losers", use_rectangles=True
        ),
        "tag_embedding_avg": None,
        "entries": losers_entries,
    }

    return find_winner(dataset_dir, winners_pool, losers_pool, use_entry_scoring)


__all__ = [
    "cosine_similarity",
    "find_winner",
    "find_winner_with_entries",
    "load_embedding_file",
]
