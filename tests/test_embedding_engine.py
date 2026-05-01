"""Tests for embedding_engine module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def sample_image(tmp_path):
    from PIL import Image

    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (64, 64), color="red")
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_tags():
    return {
        "head": {
            "hair": {"color": "black", "style": "bob_cut"},
            "eyes": {"color": "blue"},
        },
        "upper_body": {
            "clothing": {"top": "t_shirt"},
        },
    }


@pytest.fixture
def mock_embedding_engine():
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


class TestGetEmbeddingModel:
    """Test model loading and caching."""

    def test_loads_model_once(self, mock_embedding_engine):
        from chatbot.embedding_engine import get_embedding_model

        get_embedding_model.cache_clear()
        model, processor = get_embedding_model()
        mock_embedding_engine["model_cls"].from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32"
        )
        mock_embedding_engine["processor_cls"].from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32"
        )
        assert model is not None

    def test_returns_cached_model(self, mock_embedding_engine):
        from chatbot.embedding_engine import get_embedding_model

        get_embedding_model.cache_clear()
        result1 = get_embedding_model()
        result2 = get_embedding_model()
        assert result1 is result2
        mock_embedding_engine["model_cls"].from_pretrained.assert_called_once()


class TestEmbedImage:
    """Test image embedding."""

    def test_embed_image_returns_list(self, sample_image, mock_embedding_engine):
        from chatbot.embedding_engine import embed_image, get_embedding_model

        get_embedding_model.cache_clear()
        embedding = embed_image(sample_image)

        assert isinstance(embedding, list)
        assert len(embedding) == 512
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_image_uses_correct_model(self, sample_image, mock_embedding_engine):
        from chatbot.embedding_engine import embed_image, get_embedding_model

        get_embedding_model.cache_clear()
        embed_image(sample_image)

        mock_embedding_engine["model"].get_image_features.assert_called_once()

    def test_embed_image_handles_missing_file(self):
        from chatbot.embedding_engine import embed_image, get_embedding_model

        get_embedding_model.cache_clear()
        with pytest.raises(FileNotFoundError):
            embed_image("/nonexistent/image.png")


class TestEmbedTags:
    """Test tag embedding."""

    def test_embed_tags_returns_list(self, sample_tags, mock_embedding_engine):
        from chatbot.embedding_engine import embed_tags, get_embedding_model

        get_embedding_model.cache_clear()
        embedding = embed_tags(sample_tags)

        assert isinstance(embedding, list)
        assert len(embedding) == 512
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_tags_uses_correct_model(self, sample_tags, mock_embedding_engine):
        from chatbot.embedding_engine import embed_tags, get_embedding_model

        get_embedding_model.cache_clear()
        embed_tags(sample_tags)

        mock_embedding_engine["model"].get_text_features.assert_called_once()

    def test_embed_tags_empty_dict(self, mock_embedding_engine):
        from chatbot.embedding_engine import embed_tags, get_embedding_model

        get_embedding_model.cache_clear()
        embedding = embed_tags({})

        assert isinstance(embedding, list)
        assert len(embedding) == 512


class TestSerializeTags:
    """Test tag serialization for embedding."""

    def test_serializes_nested_tags(self, sample_tags):
        from chatbot.embedding_engine import serialize_tags

        result = serialize_tags(sample_tags)
        assert isinstance(result, str)
        assert "head" in result
        assert "hair" in result
        assert "black" in result

    def test_serializes_flat_tags(self):
        from chatbot.embedding_engine import serialize_tags

        tags = {"style": "casual", "color": "blue"}
        result = serialize_tags(tags)
        assert "style: casual" in result
        assert "color: blue" in result

    def test_serializes_empty_tags(self):
        from chatbot.embedding_engine import serialize_tags

        result = serialize_tags({})
        assert result == ""


class TestCreateEmbeddingFilename:
    """Test embedding filename generation."""

    def test_strips_image_extension(self):
        from chatbot.embedding_engine import create_embedding_filename

        result = create_embedding_filename("/path/to/image.jpg")
        assert result == "image.embedding.json"

    def test_handles_png_extension(self):
        from chatbot.embedding_engine import create_embedding_filename

        result = create_embedding_filename("/path/to/photo.png")
        assert result == "photo.embedding.json"

    def test_handles_jpeg_extension(self):
        from chatbot.embedding_engine import create_embedding_filename

        result = create_embedding_filename("/path/to/image.jpeg")
        assert result == "image.embedding.json"

    def test_handles_no_extension(self):
        from chatbot.embedding_engine import create_embedding_filename

        result = create_embedding_filename("/path/to/image")
        assert result == "image.embedding.json"

    def test_handles_stem_with_dots(self):
        from chatbot.embedding_engine import create_embedding_filename

        result = create_embedding_filename("/path/to/my.image.file.png")
        assert result == "my.image.file.embedding.json"


class TestSaveEmbeddings:
    """Test saving embeddings to file."""

    @pytest.fixture
    def temp_dataset_dir(self, tmp_path, monkeypatch):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        import chatbot.embedding_engine

        monkeypatch.setattr(chatbot.embedding_engine, "DATASET_DIR", dataset_dir)

        return dataset_dir

    def test_saves_image_embedding(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"
        image_emb = [0.1, 0.2, 0.3]

        result = save_embeddings(image_path, image_emb, None)

        assert Path(result).exists()
        content = json.loads(Path(result).read_text())
        assert content["image_embedding"] == image_emb
        assert content["tag_embedding"] is None

    def test_saves_tag_embedding(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"
        tag_emb = [0.4, 0.5, 0.6]

        result = save_embeddings(image_path, None, tag_emb)

        assert Path(result).exists()
        content = json.loads(Path(result).read_text())
        assert content["image_embedding"] is None
        assert content["tag_embedding"] == tag_emb

    def test_saves_both_embeddings(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"
        image_emb = [0.1, 0.2]
        tag_emb = [0.3, 0.4]

        result = save_embeddings(image_path, image_emb, tag_emb)

        assert Path(result).exists()
        content = json.loads(Path(result).read_text())
        assert content["image_embedding"] == image_emb
        assert content["tag_embedding"] == tag_emb

    def test_saves_with_metadata(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"
        image_emb = [0.1, 0.2]

        result = save_embeddings(image_path, image_emb, None)

        content = json.loads(Path(result).read_text())
        assert content["image_path"] == image_path
        assert "created_at" in content
        assert content["model"] == "openai/clip-vit-base-patch32"

    def test_overwrites_existing_embedding(self, temp_dataset_dir):
        from chatbot.embedding_engine import save_embeddings

        image_path = "/fake/path/photo.png"

        save_embeddings(image_path, [0.1], None)

        first_content = json.loads(
            Path(temp_dataset_dir / "photo.embedding.json").read_text()
        )
        assert first_content["image_embedding"] == [0.1]

        save_embeddings(image_path, [0.9, 0.8, 0.7], None)

        second_content = json.loads(
            Path(temp_dataset_dir / "photo.embedding.json").read_text()
        )
        assert second_content["image_embedding"] == [0.9, 0.8, 0.7]
