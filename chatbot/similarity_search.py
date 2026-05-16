"""Similarity search and winner selection for embeddings.

This module handles:
- Cosine similarity calculations
- Finding the best candidate based on winners/losers pools with attention masking
- Scoring: higher similarity to winners, lower to losers
- Rectangle-based attention for focusing comparison on specific regions

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
        Dictionary with image_embedding, tag_embedding, region_rects, etc.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_attention_rectangles(winners_state, losers_state):
    """Determine the attention rectangles to use for comparison.

    Uses the winner pool's rectangles as the primary attention mask.
    If winners have no rectangles, falls back to loser pool's rectangles.
    If neither has rectangles, returns None (use full image comparison).

    Args:
        winners_state: Dict with "image_embedding_avg" and "last_rectangles".
        losers_state: Dict with "image_embedding_avg" and "last_rectangles".

    Returns:
        Rectangle dict list or None.
    """
    win_rects = winners_state.get("last_rectangles")
    if win_rects and len(win_rects) > 0:
        return win_rects

    lose_rects = losers_state.get("last_rectangles")
    if lose_rects and len(lose_rects) > 0:
        return lose_rects

    return None


def _compute_scoring_embedding(
    img_emb: list | None,
    tag_emb: list | None,
    image_path: str,
    region_rects: list[dict] | None,
    attention_rects: list[dict] | None,
) -> list[float] | None:
    """Compute embedding for scoring, optionally using cropped region.

    If attention_rects are provided and the image has region_rects stored,
    crops the image to that region and re-embeds it for comparison.

    Args:
        img_emb: Original image embedding from the embedding file.
        tag_emb: Tag embedding from the embedding file.
        image_path: Path to the original image file.
        region_rects: Rectangles stored with this embedding file.
        attention_rects: Rectangles to use as attention mask.

    Returns:
        Embedding to use for scoring, or None if not computable.
    """
    if img_emb is None:
        return None

    if attention_rects is None or region_rects is None:
        return img_emb

    if not attention_rects or not region_rects:
        return img_emb

    try:
        from chatbot.image_modules.image_loading import load_image_cropped
        from chatbot.embedding_engine import embed_from_array

        img_array = load_image_cropped(image_path, region_rects)
        if img_array is not None:
            return embed_from_array(img_array, attention_rects)
    except Exception:
        pass

    return img_emb


def find_winner(dataset_dir, winners_state, losers_state):
    """Find the winner image based on similarity to winners/losers pools.

    Score = sim_to_winners - sim_to_losers for both image and tag embeddings.
    Uses rectangle-based attention when available in the pool state.
    Highest scoring candidate wins.

    Args:
        dataset_dir: Directory containing .embedding.json files.
        winners_state: Dict with "image_embedding_avg", "tag_embedding_avg", and
                      optionally "last_rectangles" for attention masking.
        losers_state: Dict with "image_embedding_avg", "tag_embedding_avg", and
                      optionally "last_rectangles" for attention masking.

    Returns:
        Dictionary with winner info: filename, scores.
    """
    dataset_dir = Path(dataset_dir)

    win_img_avg = winners_state.get("image_embedding_avg")
    win_tag_avg = winners_state.get("tag_embedding_avg")
    lose_img_avg = losers_state.get("image_embedding_avg")
    lose_tag_avg = losers_state.get("tag_embedding_avg")

    attention_rects = _get_attention_rectangles(winners_state, losers_state)

    best_file = None
    best_score = -float("inf")

    for emb_file in sorted(dataset_dir.glob("*.embedding.json")):
        data = load_embedding_file(emb_file)
        score = 0.0
        has_any = False

        img_emb = data.get("image_embedding")
        tag_emb = data.get("tag_embedding")
        region_rects = data.get("region_rects")

        if img_emb is not None and win_img_avg is not None:
            scoring_emb = _compute_scoring_embedding(
                img_emb,
                tag_emb,
                data.get("image_path", ""),
                region_rects,
                attention_rects,
            )
            if scoring_emb is not None:
                score += cosine_similarity(win_img_avg, scoring_emb)
                has_any = True

        if img_emb is not None and lose_img_avg is not None:
            scoring_emb = _compute_scoring_embedding(
                img_emb,
                tag_emb,
                data.get("image_path", ""),
                region_rects,
                attention_rects,
            )
            if scoring_emb is not None:
                score -= cosine_similarity(lose_img_avg, scoring_emb)

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
