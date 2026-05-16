"""Tests for embedding_engine module with entry support.

Tests rectangle-based entries that enable multiple region embeddings
per image, each with their own rectangle coordinates and cropped embeddings.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from chatbot.embedding_engine import (
    _rects_match,
    _find_matching_entry,
    _get_next_rect_index,
)


class TestRectMatching:
    """Test rectangle matching logic."""

    def test_identical_rects_match(self):
        rects = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        assert _rects_match(rects, rects) is True

    def test_similar_rects_match_within_tolerance(self):
        rects1 = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        rects2 = [{"x1": 12, "y1": 22, "x2": 102, "y2": 202}]
        assert _rects_match(rects1, rects2, tolerance=5) is True

    def test_different_rects_do_not_match(self):
        rects1 = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        rects2 = [{"x1": 50, "y1": 60, "x2": 150, "y2": 250}]
        assert _rects_match(rects1, rects2) is False

    def test_none_rects_match(self):
        assert _rects_match(None, None) is True

    def test_none_vs_rects_do_not_match(self):
        rects = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        assert _rects_match(None, rects) is False
        assert _rects_match(rects, None) is False

    def test_different_length_rects_do_not_match(self):
        rects1 = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        rects2 = [
            {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
            {"x1": 50, "y1": 60, "x2": 150, "y2": 250},
        ]
        assert _rects_match(rects1, rects2) is False


class TestFindMatchingEntry:
    """Test finding matching entries."""

    def test_finds_exact_match(self):
        entries = [
            {
                "id": "rect_0",
                "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
            },
            {
                "id": "rect_1",
                "region_rects": [{"x1": 50, "y1": 60, "x2": 150, "y2": 250}],
            },
        ]
        rects = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        assert _find_matching_entry(entries, rects) == "rect_0"

    def test_finds_match_within_tolerance(self):
        entries = [
            {
                "id": "rect_0",
                "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
            },
        ]
        rects = [{"x1": 12, "y1": 22, "x2": 102, "y2": 202}]
        assert _find_matching_entry(entries, rects) == "rect_0"

    def test_returns_none_for_no_match(self):
        entries = [
            {
                "id": "rect_0",
                "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
            },
        ]
        rects = [{"x1": 500, "y1": 600, "x2": 600, "y2": 700}]
        assert _find_matching_entry(entries, rects) is None

    def test_returns_none_for_empty_entries(self):
        rects = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        assert _find_matching_entry([], rects) is None

    def test_returns_none_for_empty_rects(self):
        entries = [
            {
                "id": "rect_0",
                "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
            },
        ]
        assert _find_matching_entry(entries, []) is None


class TestGetNextRectIndex:
    """Test rect_index calculation."""

    def test_returns_zero_for_empty_list(self):
        assert _get_next_rect_index([]) == 0

    def test_returns_zero_for_none(self):
        assert _get_next_rect_index(None) == 0

    def test_returns_next_index(self):
        entries = [
            {"id": "rect_0", "rect_index": 0},
            {"id": "rect_1", "rect_index": 1},
        ]
        assert _get_next_rect_index(entries) == 2

    def test_handles_missing_rect_index(self):
        entries = [
            {"id": "rect_0"},
            {"id": "rect_1", "rect_index": 1},
        ]
        assert _get_next_rect_index(entries) == 2

    def test_handles_non_sequential_indices(self):
        entries = [
            {"id": "rect_0", "rect_index": 0},
            {"id": "rect_2", "rect_index": 2},
        ]
        assert _get_next_rect_index(entries) == 3


class TestEntryFunctions:
    """Test add_entry_to_embedding and get_entry_embedding."""

    @pytest.fixture
    def mock_embedding_file(self, tmp_path, monkeypatch):
        """Create mock embedding file."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        monkeypatch.setattr("chatbot.tagging_engine.DATASET_DIR", dataset_dir)
        monkeypatch.setattr("chatbot.embedding_engine.DATASET_DIR", dataset_dir)

        image_path = tmp_path / "test_image.png"
        image_path.touch()

        emb_path = dataset_dir / "test.embedding.json"
        data = {
            "image_path": str(image_path),
            "image_embedding": [0.1] * 512,
            "tag_embedding": [0.2] * 512,
            "entries": [],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        return dataset_dir, "test", image_path

    def test_add_entry_creates_new_entry(self, mock_embedding_file, monkeypatch):
        """Test that add_entry_to_embedding creates a new entry."""
        from chatbot.embedding_engine import add_entry_to_embedding, load_embedding_file

        dataset_dir, image_name, image_path = mock_embedding_file

        import numpy as np
        from PIL import Image

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(image_path)

        emb_path = dataset_dir / f"{image_name}.embedding.json"
        with open(emb_path, "r") as f:
            data = json.load(f)
        data["image_path"] = str(image_path)
        with open(emb_path, "w") as f:
            json.dump(data, f)

        rects = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        entry_id = add_entry_to_embedding(image_name, rects)

        assert entry_id == "rect_0"

        data = load_embedding_file(image_name)
        assert len(data["entries"]) == 1
        assert data["entries"][0]["id"] == "rect_0"
        assert data["entries"][0]["rect_index"] == 0
        assert data["entries"][0]["region_rects"] == rects

    def test_add_entry_reuses_existing(self, mock_embedding_file):
        """Test that matching rectangles reuse existing entry."""
        from chatbot.embedding_engine import add_entry_to_embedding, load_embedding_file

        dataset_dir, image_name, _ = mock_embedding_file

        emb_path = dataset_dir / f"{image_name}.embedding.json"
        data = {
            "image_path": str(tmp_path := mock_embedding_file[2]),
            "image_embedding": [0.1] * 512,
            "tag_embedding": [0.2] * 512,
            "entries": [
                {
                    "id": "rect_0",
                    "rect_index": 0,
                    "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
                    "image_embedding": [0.3] * 512,
                }
            ],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        rects = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
        entry_id = add_entry_to_embedding(image_name, rects)

        assert entry_id == "rect_0"

        data = load_embedding_file(image_name)
        assert len(data["entries"]) == 1

    def test_add_entry_increments_index(self, mock_embedding_file, monkeypatch):
        """Test that subsequent entries get correct index."""
        from chatbot.embedding_engine import add_entry_to_embedding, load_embedding_file

        dataset_dir, image_name, image_path = mock_embedding_file

        import numpy as np
        from PIL import Image

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(image_path)

        emb_path = dataset_dir / f"{image_name}.embedding.json"
        with open(emb_path, "r") as f:
            data = json.load(f)
        data["image_path"] = str(image_path)
        data["entries"] = [
            {
                "id": "rect_0",
                "rect_index": 0,
                "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
                "image_embedding": [0.3] * 512,
            }
        ]
        with open(emb_path, "w") as f:
            json.dump(data, f)

        rects = [{"x1": 50, "y1": 60, "x2": 150, "y2": 250}]
        entry_id = add_entry_to_embedding(image_name, rects)

        assert entry_id == "rect_1"

        data = load_embedding_file(image_name)
        assert len(data["entries"]) == 2
        assert data["entries"][1]["id"] == "rect_1"
        assert data["entries"][1]["rect_index"] == 1

    def test_get_entry_embedding_returns_correct_embedding(self, mock_embedding_file):
        """Test get_entry_embedding returns the right embedding."""
        from chatbot.embedding_engine import get_entry_embedding

        dataset_dir, image_name, _ = mock_embedding_file

        emb_path = dataset_dir / f"{image_name}.embedding.json"
        data = {
            "image_path": str(mock_embedding_file[2]),
            "image_embedding": [0.1] * 512,
            "tag_embedding": None,
            "entries": [
                {
                    "id": "rect_0",
                    "rect_index": 0,
                    "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
                    "image_embedding": [0.3] * 512,
                }
            ],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        emb = get_entry_embedding(image_name, "rect_0")
        assert emb == [0.3] * 512

    def test_get_entry_embedding_raises_for_missing(self, mock_embedding_file):
        """Test get_entry_embedding raises ValueError for missing entry."""
        from chatbot.embedding_engine import get_entry_embedding

        dataset_dir, image_name, _ = mock_embedding_file

        emb_path = dataset_dir / f"{image_name}.embedding.json"
        data = {
            "image_path": str(mock_embedding_file[2]),
            "image_embedding": [0.1] * 512,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="rect_99"):
            get_entry_embedding(image_name, "rect_99")

    def test_get_all_entries(self, mock_embedding_file):
        """Test get_all_entries returns all entries."""
        from chatbot.embedding_engine import get_all_entries

        dataset_dir, image_name, _ = mock_embedding_file

        emb_path = dataset_dir / f"{image_name}.embedding.json"
        data = {
            "image_path": str(mock_embedding_file[2]),
            "image_embedding": [0.1] * 512,
            "tag_embedding": None,
            "entries": [
                {
                    "id": "rect_0",
                    "region_rects": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}],
                },
                {
                    "id": "rect_1",
                    "region_rects": [{"x1": 50, "y1": 60, "x2": 150, "y2": 250}],
                },
            ],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        entries = get_all_entries(image_name)
        assert len(entries) == 2
        assert entries[0]["id"] == "rect_0"
        assert entries[1]["id"] == "rect_1"

    def test_get_latest_entry_id(self, mock_embedding_file):
        """Test get_latest_entry_id returns most recent entry."""
        from chatbot.embedding_engine import get_latest_entry_id

        dataset_dir, image_name, _ = mock_embedding_file

        emb_path = dataset_dir / f"{image_name}.embedding.json"
        data = {
            "image_path": str(mock_embedding_file[2]),
            "image_embedding": [0.1] * 512,
            "tag_embedding": None,
            "entries": [
                {"id": "rect_0", "rect_index": 0, "region_rects": []},
                {"id": "rect_1", "rect_index": 1, "region_rects": []},
                {"id": "rect_2", "rect_index": 2, "region_rects": []},
            ],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        assert get_latest_entry_id(image_name) == "rect_2"

    def test_get_latest_entry_id_returns_none_for_empty(self, mock_embedding_file):
        """Test get_latest_entry_id returns None when no entries."""
        from chatbot.embedding_engine import get_latest_entry_id

        dataset_dir, image_name, _ = mock_embedding_file

        emb_path = dataset_dir / f"{image_name}.embedding.json"
        data = {
            "image_path": str(mock_embedding_file[2]),
            "image_embedding": [0.1] * 512,
            "tag_embedding": None,
            "entries": [],
        }
        with open(emb_path, "w") as f:
            json.dump(data, f)

        assert get_latest_entry_id(image_name) is None
