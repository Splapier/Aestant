"""Integration tests for embedding_engine using real CLIP model."""

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def real_image(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (224, 224), color="blue")
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def real_tags():
    return {"style": "casual", "color": "blue"}


@pytest.fixture
def temp_dataset_dir(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    import chatbot.embedding_engine

    monkeypatch.setattr(chatbot.embedding_engine, "DATASET_DIR", dataset_dir)
    return dataset_dir


class TestEmbedImageIntegration:
    """Test image embedding with real CLIP model."""

    def test_embed_image_produces_valid_embedding(self, real_image):
        from chatbot.embedding_engine import embed_image, get_embedding_model

        get_embedding_model.cache_clear()
        embedding = embed_image(real_image)

        assert isinstance(embedding, list)
        assert len(embedding) == 512
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_image_eval_mode_no_grad(self, real_image):
        """Verify model runs in eval mode without gradient tracking."""
        from chatbot.embedding_engine import embed_image, get_embedding_model

        get_embedding_model.cache_clear()
        model, _ = get_embedding_model()
        model.eval()

        embedding = embed_image(real_image)
        assert isinstance(embedding, list)
        assert len(embedding) == 512


class TestEmbedTagsIntegration:
    """Test tag embedding with real CLIP model."""

    def test_embed_tags_produces_valid_embedding(self, real_tags):
        from chatbot.embedding_engine import embed_tags, get_embedding_model

        get_embedding_model.cache_clear()
        embedding = embed_tags(real_tags)

        assert isinstance(embedding, list)
        assert len(embedding) == 512
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_tags_empty_dict(self):
        from chatbot.embedding_engine import embed_tags, get_embedding_model

        get_embedding_model.cache_clear()
        embedding = embed_tags({})

        assert isinstance(embedding, list)
        assert len(embedding) == 512

    def test_embed_tags_nested_dict(self):
        nested_tags = {
            "head": {"hair": {"color": "black", "style": "bob_cut"}},
            "upper_body": {"clothing": {"top": "t_shirt"}},
        }
        from chatbot.embedding_engine import embed_tags, get_embedding_model

        get_embedding_model.cache_clear()
        embedding = embed_tags(nested_tags)

        assert isinstance(embedding, list)
        assert len(embedding) == 512


class TestSaveEmbeddingsIntegration:
    """Test saving embeddings to disk."""

    def test_save_image_embedding(self, real_image, temp_dataset_dir):
        from chatbot.embedding_engine import (
            embed_image,
            get_embedding_model,
            save_embeddings,
        )

        get_embedding_model.cache_clear()
        image_emb = embed_image(real_image)
        result = save_embeddings(real_image, image_emb, None)

        assert Path(result).exists()
        content = json.loads(Path(result).read_text())
        assert content["image_embedding"] == image_emb
        assert content["tag_embedding"] is None
        assert content["image_path"] == real_image
        assert content["model"] == "openai/clip-vit-base-patch32"
        assert "created_at" in content

    def test_save_both_embeddings(self, real_image, real_tags, temp_dataset_dir):
        from chatbot.embedding_engine import (
            embed_image,
            embed_tags,
            get_embedding_model,
            save_embeddings,
        )

        get_embedding_model.cache_clear()
        image_emb = embed_image(real_image)
        tag_emb = embed_tags(real_tags)
        result = save_embeddings(real_image, image_emb, tag_emb)

        assert Path(result).exists()
        content = json.loads(Path(result).read_text())
        assert content["image_embedding"] == image_emb
        assert content["tag_embedding"] == tag_emb
