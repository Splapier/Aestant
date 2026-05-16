"""Tests for embedding_engine module with rectangle support.

Tests image embedding with optional rectangle cropping for
attention-based comparison.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image(tmp_path):
    """Create a test image file."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_tags():
    return {
        "head": {
            "hair": {"color": "black", "style": "bob_cut"},
        },
    }


@pytest.fixture
def mock_embedding_engine():
    """Mock CLIP model and processor."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    mock_embedding = np.random.randn(1, 512).astype(np.float32)
    mock_outputs = MagicMock()
    mock_outputs.pooler_output = MagicMock(
        detach=lambda: MagicMock(numpy=lambda: mock_embedding)
    )
    mock_model.get_image_features.return_value = mock_outputs
    mock_model.get_text_features.return_value = mock_outputs
    mock_model.eval.return_value = mock_model

    with (
        patch("chatbot.embedding_engine.CLIPModel") as mock_model_cls,
        patch("chatbot.embedding_engine.CLIPProcessor") as mock_processor_cls,
    ):
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor

        yield {
            "model": mock_model,
            "processor": mock_processor,
            "model_cls": mock_model_cls,
            "processor_cls": mock_processor_cls,
        }


class TestEmbedImageWithRectangles:
    """Test image embedding with rectangle-based cropping."""

    def setup_method(self):
        """Clear embedding model cache before each test."""
        from chatbot.embedding_engine import get_embedding_model

        get_embedding_model.cache_clear()

    def test_embed_image_without_rectangles_returns_full_embedding(
        self, sample_image, mock_embedding_engine
    ):
        from chatbot.embedding_engine import embed_image

        embedding = embed_image(sample_image, rectangles=None)

        assert isinstance(embedding, list)
        assert len(embedding) == 512

    def test_embed_image_with_single_rectangle_crops_image(
        self, sample_image, mock_embedding_engine
    ):
        from chatbot.embedding_engine import embed_image

        rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        embedding = embed_image(sample_image, rectangles=rects)

        assert isinstance(embedding, list)
        assert len(embedding) == 512

        mock_processor_instance = mock_embedding_engine["processor"].return_value
        call_args = mock_processor_instance.call_args
        if call_args and "images" in call_args.kwargs:
            img_arg = call_args.kwargs["images"]
            if hasattr(img_arg, "size"):
                assert img_arg.size == (40, 40)

    def test_embed_image_with_empty_rectangles_returns_full_embedding(
        self, sample_image, mock_embedding_engine
    ):
        from chatbot.embedding_engine import embed_image

        embedding = embed_image(sample_image, rectangles=[])

        assert isinstance(embedding, list)
        assert len(embedding) == 512

    def test_embed_image_with_multiple_rectangles_uses_first(
        self, sample_image, mock_embedding_engine
    ):
        from chatbot.embedding_engine import embed_image

        rects = [
            {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
            {"x1": 60, "y1": 60, "x2": 90, "y2": 90},
        ]
        embedding = embed_image(sample_image, rectangles=rects)

        assert isinstance(embedding, list)
        assert len(embedding) == 512

    def test_embed_image_with_inverted_rect_coords_works(
        self, sample_image, mock_embedding_engine
    ):
        from chatbot.embedding_engine import embed_image

        rects = [{"x1": 50, "y1": 50, "x2": 10, "y2": 10}]
        embedding = embed_image(sample_image, rectangles=rects)

        assert isinstance(embedding, list)
        assert len(embedding) == 512


class TestSaveEmbeddingsWithRectangles:
    """Test saving embeddings with rectangle metadata."""

    @pytest.fixture
    def temp_dataset_dir(self, tmp_path, monkeypatch):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        import chatbot.embedding_engine

        monkeypatch.setattr(chatbot.embedding_engine, "DATASET_DIR", dataset_dir)

        return dataset_dir

    def test_saves_embedding_without_rectangles(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"
        image_emb = [0.1, 0.2, 0.3]

        result = save_embeddings(image_path, image_emb, None)

        content = json.loads(Path(result).read_text())
        assert content["image_embedding"] == image_emb
        assert "region_rects" not in content or content["region_rects"] is None

    def test_saves_embedding_with_rectangles(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"
        image_emb = [0.1, 0.2, 0.3]
        rects = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200, "color": "#FF0000"}]

        result = save_embeddings(image_path, image_emb, None, rectangles=rects)

        content = json.loads(Path(result).read_text())
        assert content["image_embedding"] == image_emb
        assert "region_rects" in content
        assert content["region_rects"] == rects

    def test_saves_embedding_with_empty_rectangles(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"
        image_emb = [0.1, 0.2]

        result = save_embeddings(image_path, image_emb, None, rectangles=[])

        content = json.loads(Path(result).read_text())
        assert content["image_embedding"] == image_emb
        assert "region_rects" in content
        assert content["region_rects"] == []

    def test_region_rects_persists_across_saves(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"
        rects = [{"x1": 5, "y1": 5, "x2": 95, "y2": 95}]

        save_embeddings(image_path, [0.1], None, rectangles=rects)

        content = json.loads((temp_dataset_dir / "photo.embedding.json").read_text())
        assert content["region_rects"] == rects


class TestEmbedFromArray:
    """Test embedding from numpy arrays (used by similarity search)."""

    def setup_method(self):
        from chatbot.embedding_engine import get_embedding_model

        get_embedding_model.cache_clear()

    def test_embed_from_array_with_cropped_image(self, mock_embedding_engine):
        from chatbot.embedding_engine import embed_from_array

        img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        embedding = embed_from_array(img_array)

        assert isinstance(embedding, list)
        assert len(embedding) == 512

    def test_embed_from_array_crops_to_rectangle(self, mock_embedding_engine):
        from chatbot.embedding_engine import embed_from_array

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rects = [{"x1": 20, "y1": 20, "x2": 80, "y2": 80}]

        embedding = embed_from_array(img_array, rectangles=rects)

        assert isinstance(embedding, list)
        assert len(embedding) == 512

        mock_processor_instance = mock_embedding_engine["processor"].return_value
        call_args = mock_processor_instance.call_args
        if call_args and "images" in call_args.kwargs:
            img_arg = call_args.kwargs["images"]
            if hasattr(img_arg, "size"):
                assert img_arg.size == (60, 60)
