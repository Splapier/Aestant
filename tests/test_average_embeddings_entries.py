"""Tests for average_embeddings module with entry-based tracking.

Tests storing and using entries (image+entry_id combinations) in
winner/loser pools for attention-based comparison.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


@pytest.fixture
def clean_state_file(tmp_path, monkeypatch):
    """Point state file to temp directory and ensure clean state."""
    state_file = tmp_path / "average_embeddings.json"
    monkeypatch.setattr("chatbot.average_embeddings.STATE_FILE", state_file)
    return state_file


@pytest.fixture
def mock_embedding_access(tmp_path, monkeypatch):
    """Mock embedding file access for testing."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    monkeypatch.setattr("chatbot.tagging_engine.DATASET_DIR", dataset_dir)
    monkeypatch.setattr("chatbot.embedding_engine.DATASET_DIR", dataset_dir)

    def create_emb(name, entry_id=None, emb=None):
        emb_path = dataset_dir / f"{name}.embedding.json"
        data = {
            "image_path": str(tmp_path / f"{name}.png"),
            "image_embedding": emb or [0.1] * 512,
            "tag_embedding": None,
            "entries": [],
        }
        if entry_id:
            data["entries"].append(
                {
                    "id": entry_id,
                    "rect_index": 0,
                    "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
                    "image_embedding": emb or [0.2] * 512,
                }
            )
        with open(emb_path, "w") as f:
            json.dump(data, f)
        return emb_path

    return create_emb


class TestAddWinnerEntry:
    """Test adding entries to winners pool."""

    def test_add_winner_entry_stores_entry(
        self, clean_state_file, mock_embedding_access
    ):
        """Test that add_winner_entry stores the image+entry combo."""
        from chatbot.average_embeddings import add_winner_entry, load_state

        mock_embedding_access("img_a", "rect_0", [0.3] * 512)

        add_winner_entry("img_a", "rect_0", [0.3] * 512, None)

        state = load_state()
        assert len(state["winners"]["entries"]) == 1
        assert state["winners"]["entries"][0] == {
            "image_name": "img_a",
            "entry_id": "rect_0",
        }

    def test_add_winner_entry_with_none_entry_id(
        self, clean_state_file, mock_embedding_access
    ):
        """Test adding entry with None entry_id (full image)."""
        from chatbot.average_embeddings import add_winner_entry, load_state

        mock_embedding_access("img_a", None, [0.3] * 512)

        add_winner_entry("img_a", None, [0.3] * 512, None)

        state = load_state()
        assert len(state["winners"]["entries"]) == 1
        assert state["winners"]["entries"][0] == {
            "image_name": "img_a",
            "entry_id": None,
        }

    def test_add_winner_entry_increments_count(
        self, clean_state_file, mock_embedding_access
    ):
        """Test that add_winner_entry increments image_count."""
        from chatbot.average_embeddings import add_winner_entry, load_state

        mock_embedding_access("img_a", None, [0.1] * 512)
        mock_embedding_access("img_b", None, [0.2] * 512)

        add_winner_entry("img_a", None, [0.1] * 512, None)
        add_winner_entry("img_b", None, [0.2] * 512, None)

        state = load_state()
        assert state["winners"]["image_count"] == 2

    def test_add_winner_entry_updates_running_average(
        self, clean_state_file, mock_embedding_access
    ):
        """Test that running average is updated."""
        from chatbot.average_embeddings import add_winner_entry, load_state

        mock_embedding_access("img_a", None, [1.0, 0.0] * 256)
        mock_embedding_access("img_b", None, [0.0, 1.0] * 256)

        add_winner_entry("img_a", None, [1.0, 0.0] * 256, None)
        add_winner_entry("img_b", None, [0.0, 1.0] * 256, None)

        state = load_state()
        avg = state["winners"]["image_embedding_avg"]
        assert abs(avg[0] - 0.5) < 0.001
        assert abs(avg[1] - 0.5) < 0.001


class TestAddLoserEntry:
    """Test adding entries to losers pool."""

    def test_add_loser_entry_stores_entry(
        self, clean_state_file, mock_embedding_access
    ):
        """Test that add_loser_entry stores the image+entry combo."""
        from chatbot.average_embeddings import add_loser_entry, load_state

        mock_embedding_access("img_b", "rect_0", [0.3] * 512)

        add_loser_entry("img_b", "rect_0", [0.3] * 512, None)

        state = load_state()
        assert len(state["losers"]["entries"]) == 1
        assert state["losers"]["entries"][0] == {
            "image_name": "img_b",
            "entry_id": "rect_0",
        }

    def test_add_loser_entry_with_none_entry_id(
        self, clean_state_file, mock_embedding_access
    ):
        """Test adding entry with None entry_id (full image)."""
        from chatbot.average_embeddings import add_loser_entry, load_state

        mock_embedding_access("img_b", None, [0.3] * 512)

        add_loser_entry("img_b", None, [0.3] * 512, None)

        state = load_state()
        assert len(state["losers"]["entries"]) == 1
        assert state["losers"]["entries"][0] == {
            "image_name": "img_b",
            "entry_id": None,
        }


class TestComputeImageEmbeddingAvg:
    """Test computing average embeddings."""

    def test_compute_avg_returns_precomputed_when_use_rectangles_false(
        self, clean_state_file, mock_embedding_access
    ):
        """Test that pre-computed average is returned when use_rectangles=False."""
        from chatbot.average_embeddings import (
            add_winner_entry,
            compute_image_embedding_avg,
        )

        mock_embedding_access("img_a", None, [0.5] * 512)

        add_winner_entry("img_a", None, [0.5] * 512, None)

        avg = compute_image_embedding_avg("winners", use_rectangles=False)
        assert avg == [0.5] * 512

    def test_compute_avg_dynamically_when_use_rectangles_true(
        self, clean_state_file, mock_embedding_access, monkeypatch
    ):
        """Test that dynamic averaging is used when use_rectangles=True."""
        from chatbot.average_embeddings import (
            add_winner_entry,
            compute_image_embedding_avg,
        )

        mock_embedding_access("img_a", "rect_0", [0.3] * 512)
        mock_embedding_access("img_b", "rect_0", [0.7] * 512)

        add_winner_entry("img_a", "rect_0", [0.3] * 512, None)
        add_winner_entry("img_b", "rect_0", [0.7] * 512, None)

        avg = compute_image_embedding_avg("winners", use_rectangles=True)
        assert abs(avg[0] - 0.5) < 0.001

    def test_compute_avg_raises_for_invalid_pool(self, clean_state_file):
        """Test that ValueError is raised for invalid pool_name."""
        from chatbot.average_embeddings import compute_image_embedding_avg

        with pytest.raises(ValueError, match="Invalid pool_name"):
            compute_image_embedding_avg("invalid", use_rectangles=True)


class TestGetPoolEntries:
    """Test getting pool entries."""

    def test_get_pool_entries_returns_list(
        self, clean_state_file, mock_embedding_access
    ):
        """Test that get_pool_entries returns the entries list."""
        from chatbot.average_embeddings import add_winner_entry, get_pool_entries

        mock_embedding_access("img_a", "rect_0", [0.3] * 512)
        mock_embedding_access("img_b", None, [0.1] * 512)

        add_winner_entry("img_a", "rect_0", [0.3] * 512, None)
        add_winner_entry("img_b", None, [0.1] * 512, None)

        entries = get_pool_entries("winners")
        assert len(entries) == 2
        assert entries[0]["image_name"] == "img_a"
        assert entries[1]["image_name"] == "img_b"

    def test_get_pool_entries_returns_empty_for_empty_pool(self, clean_state_file):
        """Test that empty list is returned for empty pool."""
        from chatbot.average_embeddings import get_pool_entries

        entries = get_pool_entries("winners")
        assert entries == []

    def test_get_pool_entries_raises_for_invalid_pool(self, clean_state_file):
        """Test that ValueError is raised for invalid pool_name."""
        from chatbot.average_embeddings import get_pool_entries

        with pytest.raises(ValueError, match="Invalid pool_name"):
            get_pool_entries("invalid")


class TestGetPoolStatus:
    """Test pool status retrieval."""

    def test_get_pool_status_returns_correct_info(
        self, clean_state_file, mock_embedding_access
    ):
        """Test that get_pool_status returns correct info."""
        from chatbot.average_embeddings import (
            add_winner_entry,
            add_loser_entry,
            get_pool_status,
        )

        mock_embedding_access("img_a", "rect_0", [0.3] * 512)
        mock_embedding_access("img_b", None, [0.1] * 512)
        mock_embedding_access("img_c", "rect_1", [0.2] * 512)

        add_winner_entry("img_a", "rect_0", [0.3] * 512, None)
        add_winner_entry("img_b", None, [0.1] * 512, None)
        add_loser_entry("img_c", "rect_1", [0.2] * 512, None)

        status = get_pool_status()

        assert status["winners"]["entry_count"] == 2
        assert status["losers"]["entry_count"] == 1
        assert status["winners"]["image_count"] == 2
        assert status["losers"]["image_count"] == 1

    def test_get_pool_status_includes_entries_list(
        self, clean_state_file, mock_embedding_access
    ):
        """Test that entries are included in status."""
        from chatbot.average_embeddings import add_winner_entry, get_pool_status

        mock_embedding_access("img_a", "rect_0", [0.3] * 512)

        add_winner_entry("img_a", "rect_0", [0.3] * 512, None)

        status = get_pool_status()
        assert len(status["winners"]["entries"]) == 1
        assert status["winners"]["entries"][0]["image_name"] == "img_a"
        assert status["winners"]["entries"][0]["entry_id"] == "rect_0"


class TestUpdateRunningAverage:
    """Test running average calculation."""

    def test_first_vector_becomes_average(self):
        from chatbot.average_embeddings import update_running_average

        result = update_running_average(None, 0, [1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_subsequent_vectors_average_correctly(self):
        from chatbot.average_embeddings import update_running_average

        current_avg = [1.0, 2.0, 3.0]
        result = update_running_average(current_avg, 1, [3.0, 4.0, 5.0])

        expected = [(1.0 * 1 + 3.0) / 2, (2.0 * 1 + 4.0) / 2, (3.0 * 1 + 5.0) / 2]
        assert result == expected

    def test_average_handles_many_vectors(self):
        from chatbot.average_embeddings import update_running_average

        avg = None
        count = 0
        for i in range(10):
            vec = [float(i)] * 512
            avg = update_running_average(avg, count, vec)
            count += 1

        assert abs(avg[0] - 4.5) < 0.01


class TestStateFileFormat:
    """Test the state file JSON structure."""

    def test_state_file_has_required_keys(self, clean_state_file):
        from chatbot.average_embeddings import load_state

        state = load_state()
        assert "winners" in state
        assert "losers" in state
        assert "last_updated" in state

    def test_pool_structure_has_entries_field(self, clean_state_file):
        from chatbot.average_embeddings import load_state

        state = load_state()

        assert "entries" in state["winners"]
        assert "entries" in state["losers"]
        assert state["winners"]["entries"] == []
        assert state["losers"]["entries"] == []

    def test_migration_adds_entries_field(self, clean_state_file, monkeypatch):
        """Test that old state files are migrated with entries field."""
        from chatbot.average_embeddings import load_state

        state = {
            "winners": {
                "image_embedding_avg": [0.5] * 512,
                "image_count": 1,
            },
            "losers": {
                "image_embedding_avg": None,
                "image_count": 0,
            },
        }
        with open(clean_state_file, "w") as f:
            json.dump(state, f)

        loaded = load_state()
        assert "entries" in loaded["winners"]
        assert "entries" in loaded["losers"]
