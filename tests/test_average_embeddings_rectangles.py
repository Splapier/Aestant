"""Tests for average_embeddings module with rectangle support.

Tests storing and using rectangle metadata for attention-based
comparison in winner/loser pools.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def clean_state_file(tmp_path, monkeypatch):
    """Point state file to temp directory and ensure clean state."""
    state_file = tmp_path / "average_embeddings.json"
    monkeypatch.setattr("chatbot.average_embeddings.STATE_FILE", state_file)
    return state_file


@pytest.fixture
def sample_rectangles():
    """Sample rectangle data for testing."""
    return [{"x1": 100, "y1": 200, "x2": 400, "y2": 500, "color": "#FF0000"}]


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors."""
    return {
        "image": [0.1, 0.2, 0.3, 0.4] * 128,
        "tag": [0.5, 0.6, 0.7, 0.8] * 128,
    }


class TestAddWinnerWithRectangles:
    """Test adding winners with rectangle metadata."""

    def test_add_winner_stores_rectangles(self, clean_state_file, sample_rectangles):
        from chatbot.average_embeddings import add_winner, load_state

        image_emb = [0.1] * 512
        tag_emb = [0.2] * 512

        add_winner(image_emb, tag_emb, rectangles=sample_rectangles)

        state = load_state()
        assert state["winners"]["last_rectangles"] == sample_rectangles

    def test_add_winner_without_rectangles_has_none(self, clean_state_file):
        from chatbot.average_embeddings import add_winner, load_state

        image_emb = [0.1] * 512
        tag_emb = [0.2] * 512

        add_winner(image_emb, tag_emb)

        state = load_state()
        assert state["winners"].get("last_rectangles") is None

    def test_add_winner_increments_image_count(self, clean_state_file):
        from chatbot.average_embeddings import add_winner, load_state

        image_emb = [0.1] * 512
        tag_emb = [0.2] * 512
        rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        add_winner(image_emb, tag_emb, rectangles=rects)

        state = load_state()
        assert state["winners"]["image_count"] == 1

    def test_add_winner_with_empty_rectangles_stores_empty_list(self, clean_state_file):
        from chatbot.average_embeddings import add_winner, load_state

        image_emb = [0.1] * 512
        tag_emb = [0.2] * 512

        add_winner(image_emb, tag_emb, rectangles=[])

        state = load_state()
        assert "last_rectangles" in state["winners"]
        assert state["winners"]["last_rectangles"] == []


class TestAddLoserWithRectangles:
    """Test adding losers with rectangle metadata."""

    def test_add_loser_stores_rectangles(self, clean_state_file, sample_rectangles):
        from chatbot.average_embeddings import add_loser, load_state

        image_emb = [0.1] * 512
        tag_emb = [0.2] * 512

        add_loser(image_emb, tag_emb, rectangles=sample_rectangles)

        state = load_state()
        assert state["losers"]["last_rectangles"] == sample_rectangles

    def test_add_loser_without_rectangles_has_none(self, clean_state_file):
        from chatbot.average_embeddings import add_loser, load_state

        image_emb = [0.1] * 512
        tag_emb = [0.2] * 512

        add_loser(image_emb, tag_emb)

        state = load_state()
        assert state["losers"].get("last_rectangles") is None

    def test_add_loser_increments_image_count(self, clean_state_file):
        from chatbot.average_embeddings import add_loser, load_state

        image_emb = [0.1] * 512
        tag_emb = [0.2] * 512
        rects = [{"x1": 20, "y1": 20, "x2": 60, "y2": 60}]

        add_loser(image_emb, tag_emb, rectangles=rects)

        state = load_state()
        assert state["losers"]["image_count"] == 1


class TestLoadStateWithRectangles:
    """Test loading state that includes rectangle metadata."""

    def test_load_state_preserves_rectangles(self, clean_state_file):
        from chatbot.average_embeddings import load_state, save_state

        state = load_state()
        state["winners"]["last_rectangles"] = [
            {"x1": 100, "y1": 100, "x2": 300, "y2": 300}
        ]
        save_state(state)

        loaded = load_state()
        assert loaded["winners"]["last_rectangles"] == [
            {"x1": 100, "y1": 100, "x2": 300, "y2": 300}
        ]

    def test_load_state_handles_missing_rectangles_key(self, clean_state_file):
        from chatbot.average_embeddings import load_state

        state = load_state()

        assert (
            "last_rectangles" not in state["winners"]
            or state["winners"].get("last_rectangles") is None
        )
        assert (
            "last_rectangles" not in state["losers"]
            or state["losers"].get("last_rectangles") is None
        )


class TestRunningAverageWithRectangles:
    """Test that rectangles persist across multiple adds."""

    def test_rectangles_updated_on_each_add(self, clean_state_file):
        from chatbot.average_embeddings import add_winner, load_state

        rects_1 = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        rects_2 = [{"x1": 20, "y1": 20, "x2": 60, "y2": 60}]

        add_winner([0.1] * 512, None, rectangles=rects_1)
        add_winner([0.2] * 512, None, rectangles=rects_2)

        state = load_state()
        assert state["winners"]["last_rectangles"] == rects_2
        assert state["winners"]["image_count"] == 2

    def test_winner_and_loser_have_independent_rectangles(self, clean_state_file):
        from chatbot.average_embeddings import add_winner, add_loser, load_state

        winner_rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        loser_rects = [{"x1": 100, "y1": 100, "x2": 150, "y2": 150}]

        add_winner([0.1] * 512, None, rectangles=winner_rects)
        add_loser([0.2] * 512, None, rectangles=loser_rects)

        state = load_state()
        assert state["winners"]["last_rectangles"] == winner_rects
        assert state["losers"]["last_rectangles"] == loser_rects


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

    def test_pool_structure_has_all_fields(self, clean_state_file):
        from chatbot.average_embeddings import load_state

        state = load_state()

        for pool_name in ["winners", "losers"]:
            pool = state[pool_name]
            assert "image_embedding_avg" in pool
            assert "tag_embedding_avg" in pool
            assert "image_count" in pool
            assert "tag_count" in pool
