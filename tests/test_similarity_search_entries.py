"""Tests for similarity_search module with entry-based scoring.

Tests the winner selection logic including optional secondary scoring
that compares candidates against each individual entry.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

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
    monkeypatch.setattr("chatbot.tagging_engine.DATASET_DIR", dataset_dir)
    monkeypatch.setattr("chatbot.embedding_engine.DATASET_DIR", dataset_dir)

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
            "entries": [],
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


class TestLoadEmbeddingFile:
    """Test loading embedding data from JSON files."""

    def test_loads_existing_file(self, temp_dataset_dir):
        emb_path = temp_dataset_dir / "test.embedding.json"
        data = {
            "image_path": "/input/test.png",
            "image_embedding": [0.1, 0.2],
            "tag_embedding": [0.3, 0.4],
            "entries": [],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        result = load_embedding_file(emb_path)
        assert result["image_embedding"] == [0.1, 0.2]
        assert result["entries"] == []

    def test_loads_entries(self, temp_dataset_dir):
        emb_path = temp_dataset_dir / "test.embedding.json"
        entries = [
            {
                "id": "rect_0",
                "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
            }
        ]
        data = {
            "image_path": "/input/test.png",
            "image_embedding": [0.1],
            "entries": entries,
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        result = load_embedding_file(emb_path)
        assert len(result["entries"]) == 1
        assert result["entries"][0]["id"] == "rect_0"


class TestFindWinner:
    """Test find_winner function."""

    def test_find_winner_basic(self, temp_dataset_dir):
        """Test basic winner selection."""
        for name in ["winner_a", "winner_b", "loser_c"]:
            emb_path = temp_dataset_dir / f"{name}.embedding.json"
            emb = [0.9, 0.1] * 256 if "winner" in name else [0.1, 0.9] * 256
            data = {
                "image_path": f"/input/{name}.png",
                "image_embedding": emb,
                "tag_embedding": None,
                "entries": [],
            }
            with open(emb_path, "w") as f:
                json.dump(data, f)

        winners_pool = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
            "entries": [{"image_name": "winner_a", "entry_id": None}],
        }
        losers_pool = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
            "entries": [{"image_name": "loser_c", "entry_id": None}],
        }

        result = find_winner(temp_dataset_dir, winners_pool, losers_pool)

        assert result["winner"] in [
            "winner_a.embedding.json",
            "winner_b.embedding.json",
        ]

    def test_find_winner_with_use_entry_scoring_false(self, temp_dataset_dir):
        """Test that use_entry_scoring=False uses only primary scoring."""
        emb_path = temp_dataset_dir / "test.embedding.json"
        data = {
            "image_path": "/input/test.png",
            "image_embedding": [0.5] * 512,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        winners_pool = {
            "image_embedding_avg": [0.9] * 512,
            "tag_embedding_avg": None,
            "entries": [],
        }
        losers_pool = {
            "image_embedding_avg": [0.1] * 512,
            "tag_embedding_avg": None,
            "entries": [],
        }

        result = find_winner(
            temp_dataset_dir, winners_pool, losers_pool, use_entry_scoring=False
        )
        assert result["winner"] == "test.embedding.json"

    def test_find_winner_returns_none_when_no_candidates(self, temp_dataset_dir):
        """Test that None is returned when no candidates available."""
        winners_pool = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": None,
            "entries": [],
        }
        losers_pool = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": None,
            "entries": [],
        }

        result = find_winner(temp_dataset_dir, winners_pool, losers_pool)

        assert result["winner"] is None
        assert result["score"] == -float("inf")

    def test_find_winner_with_tag_embeddings(self, temp_dataset_dir):
        """Test winner selection with tag embeddings."""
        emb_path_a = temp_dataset_dir / "tag_a.embedding.json"
        data_a = {
            "image_path": "/input/tag_a.png",
            "image_embedding": [0.5] * 512,
            "tag_embedding": [0.9, 0.1] * 256,
            "entries": [],
        }
        with open(emb_path_a, "w") as f:
            json.dump(data_a, f)

        emb_path_b = temp_dataset_dir / "tag_b.embedding.json"
        data_b = {
            "image_path": "/input/tag_b.png",
            "image_embedding": [0.5] * 512,
            "tag_embedding": [0.1, 0.9] * 256,
            "entries": [],
        }
        with open(emb_path_b, "w") as f:
            json.dump(data_b, f)

        winners_pool = {
            "image_embedding_avg": None,
            "tag_embedding_avg": [0.9, 0.1] * 256,
            "entries": [],
        }
        losers_pool = {
            "image_embedding_avg": None,
            "tag_embedding_avg": [0.1, 0.9] * 256,
            "entries": [],
        }

        result = find_winner(temp_dataset_dir, winners_pool, losers_pool)

        assert result["winner"] == "tag_a.embedding.json"


class TestFindWinnerWithEntries:
    """Test find_winner with entry-based pools."""

    def test_find_winner_with_entries_in_pools(self, temp_dataset_dir):
        """Test winner selection when pools have entries."""
        for name, emb in [
            ("entry_win", [0.9, 0.1] * 256),
            ("no_entry", [0.5, 0.5] * 256),
        ]:
            emb_path = temp_dataset_dir / f"{name}.embedding.json"
            entries = (
                [{"id": "rect_0", "image_embedding": [0.95, 0.05] * 256}]
                if name == "entry_win"
                else []
            )
            data = {
                "image_path": f"/input/{name}.png",
                "image_embedding": emb,
                "tag_embedding": None,
                "entries": entries,
            }
            with open(emb_path, "w") as f:
                json.dump(data, f)

        winners_pool = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
            "entries": [
                {"image_name": "entry_win", "entry_id": "rect_0"},
            ],
        }
        losers_pool = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
            "entries": [],
        }

        result = find_winner(temp_dataset_dir, winners_pool, losers_pool)

        assert result["winner"] is not None

    def test_find_winner_with_entry_scoring_enabled(self, temp_dataset_dir):
        """Test that use_entry_scoring=True adds secondary scoring."""
        emb_path_a = temp_dataset_dir / "match.embedding.json"
        emb_a = [0.9, 0.1] * 256
        entries_a = [{"id": "rect_0", "image_embedding": [0.92, 0.08] * 256}]
        data_a = {
            "image_path": "/input/match.png",
            "image_embedding": emb_a,
            "tag_embedding": None,
            "entries": entries_a,
        }
        with open(emb_path_a, "w") as f:
            json.dump(data_a, f)

        emb_path_b = temp_dataset_dir / "no_match.embedding.json"
        emb_b = [0.1, 0.9] * 256
        data_b = {
            "image_path": "/input/no_match.png",
            "image_embedding": emb_b,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path_b, "w") as f:
            json.dump(data_b, f)

        winners_pool = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
            "entries": [
                {"image_name": "match", "entry_id": "rect_0"},
            ],
        }
        losers_pool = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
            "entries": [],
        }

        result = find_winner(
            temp_dataset_dir, winners_pool, losers_pool, use_entry_scoring=True
        )

        assert result["winner"] == "match.embedding.json"

    def test_find_winner_entry_scoring_sums_individually(self, temp_dataset_dir):
        """Test that entry scoring sums each entry's contribution separately."""
        emb_path = temp_dataset_dir / "candidate.embedding.json"
        data = {
            "image_path": "/input/candidate.png",
            "image_embedding": [0.85, 0.15] * 256,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        emb_path_win1 = temp_dataset_dir / "win1.embedding.json"
        data_win1 = {
            "image_path": "/input/win1.png",
            "image_embedding": [0.95, 0.05] * 256,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path_win1, "w") as f:
            json.dump(data_win1, f)

        emb_path_win2 = temp_dataset_dir / "win2.embedding.json"
        data_win2 = {
            "image_path": "/input/win2.png",
            "image_embedding": [0.92, 0.08] * 256,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path_win2, "w") as f:
            json.dump(data_win2, f)

        emb_path_lose1 = temp_dataset_dir / "lose1.embedding.json"
        data_lose1 = {
            "image_path": "/input/lose1.png",
            "image_embedding": [0.1, 0.9] * 256,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path_lose1, "w") as f:
            json.dump(data_lose1, f)

        winners_pool = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
            "entries": [
                {"image_name": "win1", "entry_id": None},
                {"image_name": "win2", "entry_id": None},
            ],
        }
        losers_pool = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
            "entries": [
                {"image_name": "lose1", "entry_id": None},
            ],
        }

        result_with_entry = find_winner(
            temp_dataset_dir, winners_pool, losers_pool, use_entry_scoring=True
        )
        result_without_entry = find_winner(
            temp_dataset_dir, winners_pool, losers_pool, use_entry_scoring=False
        )

        assert result_with_entry["score"] != result_without_entry["score"]


class TestFindWinnerEdgeCases:
    """Test edge cases in find_winner."""

    def test_handles_empty_winner_pool(self, temp_dataset_dir):
        """Test handling of empty winner pool."""
        emb_path = temp_dataset_dir / "candidate.embedding.json"
        data = {
            "image_path": "/input/candidate.png",
            "image_embedding": [0.5] * 512,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        winners_pool = {
            "image_embedding_avg": None,
            "tag_embedding_avg": None,
            "entries": [],
        }
        losers_pool = {
            "image_embedding_avg": [0.1, 0.9] * 256,
            "tag_embedding_avg": None,
            "entries": [],
        }

        result = find_winner(temp_dataset_dir, winners_pool, losers_pool)

        assert result["winner"] == "candidate.embedding.json"

    def test_handles_empty_loser_pool(self, temp_dataset_dir):
        """Test handling of empty loser pool."""
        winners_pool = {
            "image_embedding_avg": [0.9, 0.1] * 256,
            "tag_embedding_avg": None,
            "entries": [],
        }
        losers_pool = {
            "image_embedding_avg": None,
            "tag_embedding_avg": None,
            "entries": [],
        }

        for name in ["good", "bad"]:
            emb_path = temp_dataset_dir / f"{name}.embedding.json"
            emb = [0.95, 0.05] * 256 if name == "good" else [0.4, 0.6] * 256
            data = {
                "image_path": f"/input/{name}.png",
                "image_embedding": emb,
                "tag_embedding": None,
                "entries": [],
            }
            with open(emb_path, "w") as f:
                json.dump(data, f)

        result = find_winner(temp_dataset_dir, winners_pool, losers_pool)

        assert result["winner"] == "good.embedding.json"

    def test_skips_candidates_without_embeddings(self, temp_dataset_dir):
        """Test that candidates without embeddings are skipped."""
        winners_pool = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": None,
            "entries": [],
        }
        losers_pool = {
            "image_embedding_avg": [0.5] * 512,
            "tag_embedding_avg": None,
            "entries": [],
        }

        emb_path = temp_dataset_dir / "no_emb.embedding.json"
        data = {
            "image_path": "/input/no_emb.png",
            "image_embedding": None,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        result = find_winner(temp_dataset_dir, winners_pool, losers_pool)

        assert result["winner"] is None
