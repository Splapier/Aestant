"""Tests for similarity_search module with attention masking.

Tests the winner selection logic when rectangles are used as
attention masks to focus comparison on specific regions.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from chatbot.similarity_search import (
    cosine_similarity,
    find_winner,
    load_embedding_file,
)


@pytest.fixture
def temp_dataset_dir(tmp_path, monkeypatch):
    """Create a temporary dataset directory."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    import chatbot.similarity_search

    monkeypatch.setattr(chatbot.similarity_search, "DATASET_DIR", dataset_dir)

    return dataset_dir


@pytest.fixture
def sample_embedding_files(temp_dataset_dir):
    """Create sample embedding files for testing."""
    files = {}
    for name in ["img_a", "img_b", "img_c"]:
        emb_path = temp_dataset_dir / f"{name}.embedding.json"
        data = {
            "image_path": f"/input/{name}.png",
            "image_embedding": [0.1 + i * 0.1] * 512,
            "tag_embedding": [0.5 + i * 0.05] * 512,
            "region_rects": None,
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)
        files[name] = emb_path
    return files


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_vectors_return_1(self):
        vec = [0.1, 0.2, 0.3, 0.4]
        result = cosine_similarity(vec, vec)
        assert abs(result - 1.0) < 0.001

    def test_opposite_vectors_return_minus_1(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        result = cosine_similarity(vec1, vec2)
        assert abs(result + 1.0) < 0.001

    def test_perpendicular_vectors_return_0(self):
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        result = cosine_similarity(vec1, vec2)
        assert abs(result) < 0.001

    def test_handles_numpy_arrays(self):
        result = cosine_similarity(np.array([1, 2, 3]), np.array([2, 4, 6]))
        assert abs(result - 1.0) < 0.001

    def test_handles_lists(self):
        result = cosine_similarity([1, 2, 3], [2, 4, 6])
        assert abs(result - 1.0) < 0.001


class TestLoadEmbeddingFile:
    """Test loading embedding data from JSON files."""

    def test_loads_existing_file(self, temp_dataset_dir):
        emb_path = temp_dataset_dir / "test.embedding.json"
        data = {
            "image_path": "/input/test.png",
            "image_embedding": [0.1, 0.2],
            "tag_embedding": [0.3, 0.4],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        result = load_embedding_file(emb_path)
        assert result["image_embedding"] == [0.1, 0.2]
        assert result["tag_embedding"] == [0.3, 0.4]

    def test_loads_region_rects(self, temp_dataset_dir):
        emb_path = temp_dataset_dir / "test.embedding.json"
        rects = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        data = {
            "image_path": "/input/test.png",
            "image_embedding": [0.1],
            "tag_embedding": None,
            "region_rects": rects,
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        result = load_embedding_file(emb_path)
        assert result["region_rects"] == rects


class TestFindWinnerWithAttentionMasking:
    """Test find_winner with rectangle-based attention masking."""

    def test_find_winner_without_rectangles(self, temp_dataset_dir):
        for name in ["winner_a", "winner_b", "loser_c"]:
            emb_path = temp_dataset_dir / f"{name}.embedding.json"
            emb = [0.9, 0.1] * 256 if "winner" in name else [0.1, 0.9] * 256
            data = {
                "image_path": f"/input/{name}.png",
                "image_embedding": emb,
                "tag_embedding": None,
                "region_rects": None,
            }
            with open(emb_path, "w") as f:
                json.dump(data, f)

        winners_state = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
        }
        losers_state = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
        }

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] in [
            "winner_a.embedding.json",
            "winner_b.embedding.json",
        ]

    def test_find_winner_with_rectangles_in_winner_pool(
        self, temp_dataset_dir, monkeypatch
    ):
        from chatbot.image_modules.image_loading import load_image_cropped

        emb_path_win = temp_dataset_dir / "winner.embedding.json"
        win_rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        data_win = {
            "image_path": str(temp_dataset_dir / "winner.png"),
            "image_embedding": [0.9, 0.1] * 256,
            "tag_embedding": None,
            "region_rects": win_rects,
        }
        with open(emb_path_win, "w") as f:
            json.dump(data_win, f)

        emb_path_cand = temp_dataset_dir / "candidate.embedding.json"
        cand_rects = [{"x1": 20, "y1": 20, "x2": 60, "y2": 60}]
        data_cand = {
            "image_path": str(temp_dataset_dir / "candidate.png"),
            "image_embedding": [0.85, 0.15] * 256,
            "tag_embedding": None,
            "region_rects": cand_rects,
        }
        with open(emb_path_cand, "w") as f:
            json.dump(data_cand, f)

        winners_state = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
            "last_rectangles": win_rects,
        }
        losers_state = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
        }

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] in ["winner.embedding.json", "candidate.embedding.json"]

    def test_find_winner_uses_attention_mask_when_candidate_has_rects(
        self, temp_dataset_dir
    ):
        for name in ["match_win", "no_match"]:
            emb_path = temp_dataset_dir / f"{name}.embedding.json"
            rects = (
                [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
                if name == "match_win"
                else None
            )
            emb = [0.95, 0.05] * 256 if name == "match_win" else [0.1, 0.9] * 256
            data = {
                "image_path": f"/input/{name}.png",
                "image_embedding": emb,
                "tag_embedding": None,
                "region_rects": rects,
            }
            with open(emb_path, "w") as f:
                json.dump(data, f)

        winners_state = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
            "last_rectangles": [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}],
        }
        losers_state = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
        }

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] == "match_win.embedding.json"

    def test_find_winner_handles_mixed_rectangles(self, temp_dataset_dir):
        emb_path_has_rects = temp_dataset_dir / "has_rects.embedding.json"
        data_has = {
            "image_path": "/input/has_rects.png",
            "image_embedding": [0.6, 0.4] * 256,
            "tag_embedding": None,
            "region_rects": [{"x1": 0, "y1": 0, "x2": 50, "y2": 50}],
        }
        with open(emb_path_has_rects, "w") as f:
            json.dump(data_has, f)

        emb_path_no_rects = temp_dataset_dir / "no_rects.embedding.json"
        data_no = {
            "image_path": "/input/no_rects.png",
            "image_embedding": [0.5, 0.5] * 256,
            "tag_embedding": None,
            "region_rects": None,
        }
        with open(emb_path_no_rects, "w") as f:
            json.dump(data_no, f)

        winners_state = {
            "image_embedding_avg": [0.7, 0.3] * 256,
            "tag_embedding_avg": None,
            "last_rectangles": [{"x1": 0, "y1": 0, "x2": 40, "y2": 40}],
        }
        losers_state = {
            "image_embedding_avg": [0.3, 0.7] * 256,
            "tag_embedding_avg": None,
        }

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] in [
            "has_rects.embedding.json",
            "no_rects.embedding.json",
        ]

    def test_find_winner_returns_none_when_no_candidates(self, temp_dataset_dir):
        winners_state = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": None,
        }
        losers_state = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": None,
        }

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] is None
        assert result["score"] == -float("inf")

    def test_find_winner_with_tag_embeddings(self, temp_dataset_dir):
        emb_path_a = temp_dataset_dir / "tag_a.embedding.json"
        data_a = {
            "image_path": "/input/tag_a.png",
            "image_embedding": [0.5] * 512,
            "tag_embedding": [0.9, 0.1] * 256,
            "region_rects": None,
        }
        with open(emb_path_a, "w") as f:
            json.dump(data_a, f)

        emb_path_b = temp_dataset_dir / "tag_b.embedding.json"
        data_b = {
            "image_path": "/input/tag_b.png",
            "image_embedding": [0.5] * 512,
            "tag_embedding": [0.1, 0.9] * 256,
            "region_rects": None,
        }
        with open(emb_path_b, "w") as f:
            json.dump(data_b, f)

        winners_state = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": [0.9, 0.1] * 256,
        }
        losers_state = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": [0.1, 0.9] * 256,
        }

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] == "tag_a.embedding.json"


class TestFindWinnerEdgeCases:
    """Test edge cases in find_winner."""

    def test_handles_missing_image_path(self, temp_dataset_dir):
        emb_path = temp_dataset_dir / "orphan.embedding.json"
        data = {
            "image_path": "/nonexistent/path.png",
            "image_embedding": [0.5] * 512,
            "tag_embedding": None,
            "region_rects": None,
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        winners_state = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": None,
        }
        losers_state = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": None,
        }

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] == "orphan.embedding.json"

    def test_handles_empty_winner_pool(self, temp_dataset_dir):
        winners_state = {
            "image_embedding_avg": None,
            "tag_embedding_avg": None,
        }
        losers_state = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
        }

        emb_path = temp_dataset_dir / "candidate.embedding.json"
        data = {
            "image_path": "/input/candidate.png",
            "image_embedding": [0.5] * 512,
            "tag_embedding": None,
            "region_rects": None,
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] is None
        assert result["score"] == -float("inf")

    def test_handles_empty_loser_pool(self, temp_dataset_dir):
        winners_state = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
        }
        losers_state = {
            "image_embedding_avg": None,
            "tag_embedding_avg": None,
        }

        for name in ["good", "bad"]:
            emb_path = temp_dataset_dir / f"{name}.embedding.json"
            emb = [0.95, 0.05] * 256 if name == "good" else [0.4, 0.6] * 256
            data = {
                "image_path": f"/input/{name}.png",
                "image_embedding": emb,
                "tag_embedding": None,
                "region_rects": None,
            }
            with open(emb_path, "w") as f:
                json.dump(data, f)

        result = find_winner(temp_dataset_dir, winners_state, losers_state)

        assert result["winner"] == "good.embedding.json"
