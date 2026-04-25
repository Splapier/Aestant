"""Tests for tagging_engine module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chatbot.tagging_engine import (
    crop_to_bounding_box,
    dict_to_yaml_string,
    ensure_dataset_dir,
    get_level_0_keys_with_subtrees,
    parse_yaml_response,
    save_tagged_dataset,
    scan_untagged_images,
)
from chatbot.schema_manager import flatten_schema_keys


@pytest.fixture
def sample_image():
    """Create a small test image as numpy array."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_tags():
    """Sample tags dictionary."""
    return {
        "head": {
            "hair": {
                "color": "black",
                "style": "bob_cut",
            },
            "eyes": {
                "color": "blue",
            },
        },
        "upper_body": {
            "clothing": {
                "top": "t_shirt",
            },
        },
    }


@pytest.fixture
def sample_raw_responses():
    """Sample raw YAML responses."""
    return {
        "head": "head:\n  hair:\n    color: black\n    style: bob_cut\n  eyes:\n    color: blue\n",
        "upper_body": "upper_body:\n  clothing:\n    top: t_shirt\n",
    }


@pytest.fixture
def sample_prompts():
    """Sample prompts."""
    return {
        "head": "Describe the head region...",
        "upper_body": "Describe the upper body region...",
    }


class TestCropToBoundingBox:
    """Test bounding box cropping."""

    def test_returns_original_when_no_rectangles(self, sample_image):
        result = crop_to_bounding_box(sample_image, [])
        assert np.array_equal(result, sample_image)

    def test_returns_original_when_multiple_rectangles(self, sample_image):
        rects = [
            {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
            {"x1": 60, "y1": 60, "x2": 90, "y2": 90},
        ]
        result = crop_to_bounding_box(sample_image, rects)
        assert np.array_equal(result, sample_image)

    def test_crops_to_single_rectangle(self, sample_image):
        rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        result = crop_to_bounding_box(sample_image, rects)
        assert result.shape == (40, 40, 3)

    def test_handles_inverted_coordinates(self, sample_image):
        rects = [{"x1": 50, "y1": 50, "x2": 10, "y2": 10}]
        result = crop_to_bounding_box(sample_image, rects)
        assert result.shape == (40, 40, 3)

    def test_handles_out_of_bounds(self, sample_image):
        rects = [{"x1": -10, "y1": -10, "x2": 200, "y2": 200}]
        result = crop_to_bounding_box(sample_image, rects)
        assert result.shape[0] <= 100
        assert result.shape[1] <= 100


class TestGetLevel0KeysWithSubtrees:
    """Test level 0 key extraction."""

    def test_extracts_level_0_keys(self):
        schema = {
            "head": {"hair": {}, "eyes": {}},
            "upper_body": {"clothing": {}},
            "lower_body": {"legs": {}},
        }
        keys = get_level_0_keys_with_subtrees(schema)
        key_names = [k for k, v in keys]
        assert "head" in key_names
        assert "upper_body" in key_names
        assert "lower_body" in key_names
        assert len(keys) == 3

    def test_filters_non_dict_values(self):
        schema = {
            "head": {"hair": {}},
            "version": "1.0",
        }
        keys = get_level_0_keys_with_subtrees(schema)
        key_names = [k for k, v in keys]
        assert "head" in key_names
        assert "version" not in key_names


class TestFlattenSchemaKeys:
    """Test schema key flattening."""

    def test_flattens_nested_keys(self):
        schema = {
            "head": {
                "hair": {
                    "color": {"type": "string"},
                },
                "eyes": {
                    "color": {"type": "string"},
                },
            },
        }
        flat = flatten_schema_keys(schema)
        assert "head.hair.color" in flat
        assert "head.eyes.color" in flat


class TestParseYamlResponse:
    """Test YAML parsing."""

    def test_parses_simple_yaml(self):
        yaml_str = "head:\n  hair:\n    color: black"
        result = parse_yaml_response(yaml_str)
        assert result["head"]["hair"]["color"] == "black"

    def test_parses_yaml_with_code_fences(self):
        yaml_str = "```yaml\nhead:\n  hair:\n    color: black\n```"
        result = parse_yaml_response(yaml_str)
        assert result["head"]["hair"]["color"] == "black"

    def test_handles_empty_response(self):
        result = parse_yaml_response("")
        assert result == {}

    def test_handles_whitespace_only(self):
        result = parse_yaml_response("   \n   ")
        assert result == {}


class TestDictToYamlString:
    """Test dict to YAML conversion."""

    def test_converts_to_yaml_string(self, sample_tags):
        result = dict_to_yaml_string(sample_tags)
        assert isinstance(result, str)
        assert "head:" in result
        assert "hair:" in result

    def test_preserves_hierarchy(self):
        data = {"head": {"hair": {"color": "black"}}}
        result = dict_to_yaml_string(data)
        assert "head:" in result
        assert "hair:" in result
        assert "color: black" in result


class TestScanUntaggedImages:
    """Test scanning for untagged images."""

    @pytest.fixture
    def temp_dirs(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "input"
        dataset_dir = tmp_path / "dataset"
        input_dir.mkdir()
        dataset_dir.mkdir()

        import chatbot.tagging_engine

        monkeypatch.setattr(chatbot.tagging_engine, "INPUT_DIR", input_dir)
        monkeypatch.setattr(chatbot.tagging_engine, "DATASET_DIR", dataset_dir)

        return input_dir, dataset_dir

    def test_returns_empty_when_no_images(self, temp_dirs):
        input_dir, _ = temp_dirs
        result = scan_untagged_images()
        assert result == []

    def test_returns_images_not_in_dataset(self, temp_dirs, monkeypatch):
        from PIL import Image

        input_dir, dataset_dir = temp_dirs

        img = Image.new("RGB", (50, 50), color="red")
        img_path = input_dir / "test_image.png"
        img.save(img_path)

        result = scan_untagged_images()
        assert str(img_path) in result

    def test_excludes_tagged_images(self, temp_dirs, monkeypatch):
        from PIL import Image

        input_dir, dataset_dir = temp_dirs

        img = Image.new("RGB", (50, 50), color="red")
        img_path = input_dir / "test_image.png"
        img.save(img_path)

        dataset_file = dataset_dir / "test_image.yaml"
        dataset_file.write_text("tags: {}")

        result = scan_untagged_images()
        assert str(img_path) not in result


class TestSaveTaggedDataset:
    """Test saving tagged dataset."""

    @pytest.fixture
    def temp_dir(self, tmp_path, monkeypatch):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        import chatbot.tagging_engine

        monkeypatch.setattr(chatbot.tagging_engine, "DATASET_DIR", dataset_dir)

        return dataset_dir

    def test_saves_yaml_file(
        self, temp_dir, sample_tags, sample_raw_responses, sample_prompts
    ):
        image_path = "/fake/path/image.png"

        result = save_tagged_dataset(
            image_path, sample_tags, sample_raw_responses, sample_prompts
        )

        assert Path(result).exists()
        content = Path(result).read_text()
        assert "image_path:" in content
        assert "tags:" in content

    def test_saves_raw_responses(
        self, temp_dir, sample_tags, sample_raw_responses, sample_prompts
    ):
        image_path = "/fake/path/image.png"

        result = save_tagged_dataset(
            image_path, sample_tags, sample_raw_responses, sample_prompts
        )

        content = Path(result).read_text()
        assert "raw_responses:" in content
