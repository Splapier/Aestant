"""Similarity search and winner selection for embeddings.

This module handles:
- Cosine similarity calculations
- Finding the best candidate based on winners/losers pools
- Scoring: higher similarity to winners, lower to losers

Usage:
    >>> from chatbot.similarity_search import find_winner
    >>> winner = find_winner(dataset_dir, winners_state, losers_state)
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
        Dictionary with image_embedding, tag_embedding, etc.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def find_winner(dataset_dir, winners_state, losers_state):
    """Find the winner image based on similarity to winners/losers pools.

    Score = sim_to_winners - sim_to_losers for both image and tag embeddings.
    Highest scoring candidate wins.

    Args:
        dataset_dir: Directory containing .embedding.json files.
        winners_state: Dict with "image_embedding_avg" and "tag_embedding_avg".
        losers_state: Dict with "image_embedding_avg" and "tag_embedding_avg".

    Returns:
        Dictionary with winner info: filename, scores.
    """
    dataset_dir = Path(dataset_dir)

    win_img_avg = winners_state.get("image_embedding_avg")
    win_tag_avg = winners_state.get("tag_embedding_avg")
    lose_img_avg = losers_state.get("image_embedding_avg")
    lose_tag_avg = losers_state.get("tag_embedding_avg")

    best_file = None
    best_score = -float("inf")

    for emb_file in sorted(dataset_dir.glob("*.embedding.json")):
        data = load_embedding_file(emb_file)
        score = 0.0
        has_any = False

        # Image similarity scoring
        img_emb = data.get("image_embedding")
        if img_emb is not None and win_img_avg is not None:
            score += cosine_similarity(win_img_avg, img_emb)
            has_any = True
        if img_emb is not None and lose_img_avg is not None:
            score -= cosine_similarity(lose_img_avg, img_emb)

        # Tag similarity scoring (optional)
        tag_emb = data.get("tag_embedding")
        if tag_emb is not None and win_tag_avg is not None:
            score += cosine_similarity(win_tag_avg, tag_emb)
        if tag_emb is not None and lose_tag_avg is not None:
            score -= cosine_similarity(lose_tag_avg, tag_emb)

        if has_any and score > best_score:
            best_score = score
            best_file = emb_file.name

    return {
        "winner": best_file,
        "score": best_score,
    }
