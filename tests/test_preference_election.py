"""Tests for preference election functionality.

This module provides unit tests for the preference_election module,
covering JSON loading, image scanning, tournament pairing logic,
and full election workflows.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from chatbot.preference_election import (
    PreferenceData,
    ElectionMatch,
    ElectionResult,
    load_preference_json,
    create_empty_preference_json,
    scan_all_images,
    pair_images_for_round,
    run_single_comparison,
    run_tournament_round,
    run_full_election,
    generate_tags_for_image,
    parse_comparison_response,
    parse_tags_response,
)


class TestPreferenceJSON:
    """Tests for preference JSON loading and creation."""

    @pytest.fixture
    def temp_preference_json(self, tmp_path):
        """Create a temporary preference JSON file."""
        pref_data = {
            "confidence_score": 75,
            "favored_features": ["blue_eyes", "long_hair"],
            "avoided_features": ["dark_theme"],
            "text_hypothesis": "I prefer images with vibrant colors",
        }
        file_path = tmp_path / "preference.json"
        with open(file_path, "w") as f:
            json.dump(pref_data, f)
        return file_path

    @pytest.fixture
    def project_root(self, tmp_path, monkeypatch):
        """Set up temporary project root for testing."""
        monkeypatch.setattr("chatbot.preference_election.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            "chatbot.preference_election.PREFERENCE_JSON_PATH",
            tmp_path / "preference.json",
        )
        monkeypatch.setattr("chatbot.preference_election.INPUT_DIR", tmp_path / "input")
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)
        return tmp_path

    def test_load_valid_preference_json(self, temp_preference_json, monkeypatch):
        """Test loading a valid preference JSON file."""
        monkeypatch.setattr(
            "chatbot.preference_election.PREFERENCE_JSON_PATH", temp_preference_json
        )
        result = load_preference_json()
        assert result.confidence_score == 75
        assert result.favored_features == ["blue_eyes", "long_hair"]
        assert result.avoided_features == ["dark_theme"]
        assert result.text_hypothesis == "I prefer images with vibrant colors"

    def test_load_missing_json_raises(self, project_root):
        """Test that loading missing JSON raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_preference_json()

    def test_create_empty_preference_json(self, project_root):
        """Test creating empty preference JSON template."""
        result = create_empty_preference_json()
        assert result.confidence_score == 0
        assert result.favored_features == []
        assert result.avoided_features == []
        assert result.text_hypothesis == ""
        assert project_root.joinpath("preference.json").exists()


class TestScanAllImages:
    """Tests for scanning all images from input directory."""

    @pytest.fixture
    def image_dir(self, tmp_path):
        """Create test image directory with various images."""
        for i, color in enumerate(["red", "blue", "green", "yellow", "purple"]):
            img = Image.new("RGB", (10, 10), color)
            img.save(tmp_path / f"test_{i}.png")
        return tmp_path

    @pytest.fixture
    def project_root(self, image_dir, monkeypatch):
        """Set up temporary project root pointing to image dir."""
        monkeypatch.setattr("chatbot.preference_election.INPUT_DIR", image_dir)
        return image_dir

    def test_scan_all_images_returns_all(self, project_root):
        """Test that scan_all_images returns all images, not limited to 2."""
        images = scan_all_images()
        assert len(images) == 5

    def test_scan_all_images_empty_dir(self, tmp_path, monkeypatch):
        """Test scanning empty directory returns empty list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr("chatbot.preference_election.INPUT_DIR", empty_dir)
        result = scan_all_images()
        assert result == []


class TestPairingLogic:
    """Tests for tournament pairing logic."""

    def test_pairing_even_count(self):
        """Test correct pairing for even number of images."""
        images = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)]
        images[0][:] = [255, 0, 0]
        images[1][:] = [0, 255, 0]
        images[2][:] = [0, 0, 255]
        images[3][:] = [255, 255, 0]
        pairs, byes = pair_images_for_round(images)

        assert len(pairs) == 2
        assert byes == []

        all_items = [item for pair in pairs for item in pair]
        assert len(all_items) == 4

    def test_pairing_odd_count(self):
        """Test correct bye handling for odd number of images."""
        images = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
        pairs, byes = pair_images_for_round(images)

        assert len(pairs) == 2
        assert len(byes) == 1

        paired = [item for pair in pairs for item in pair]
        assert len(paired) + len(byes) == len(images)

    def test_pairing_single_image(self):
        """Test single image goes to bye."""
        images = [np.zeros((10, 10, 3), dtype=np.uint8)]
        pairs, byes = pair_images_for_round(images)

        assert pairs == []
        assert len(byes) == 1

    def test_pairing_two_images(self):
        """Test two images form single pair."""
        images = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]
        pairs, byes = pair_images_for_round(images)

        assert len(pairs) == 1
        assert byes == []


class TestSingleComparison:
    """Tests for single image comparison via VLM."""

    def test_parse_comparison_response_valid(self):
        """Test parsing valid JSON response from VLM."""
        response = '{"winner": "A", "reason": "Image A has blue eyes"}'
        result = parse_comparison_response(response)

        assert result["winner"] == "A"
        assert "blue eyes" in result["reason"]

    def test_parse_comparison_response_case_insensitive(self):
        """Test that winner parsing is case insensitive."""
        response = '{"winner": "a", "reason": "test"}'
        result = parse_comparison_response(response)

        assert result["winner"] == "A"

    def test_parse_comparison_response_invalid(self):
        """Test handling of invalid response."""
        response = "not valid json"
        result = parse_comparison_response(response)

        assert result["winner"] is None
        assert "error" in result["reason"].lower()

    @patch("chatbot.preference_election.get_provider")
    def test_run_single_comparison(self, mock_get_provider):
        """Test running single comparison with mock provider."""
        mock_provider = MagicMock()
        mock_provider.stream_chat.return_value = iter(
            ['{"winner": "A", "reason": "test reason"}']
        )
        mock_get_provider.return_value = mock_provider

        pref = PreferenceData(50, [], [], "")
        img_a = np.zeros((10, 10, 3), dtype=np.uint8)
        img_b = np.zeros((10, 10, 3), dtype=np.uint8) + 128

        result = run_single_comparison(
            img_a, img_b, pref, "lmstudio", {"endpoint_url": "http://localhost:1234"}
        )

        assert result.winner == "A"
        assert result.image_a is img_a
        assert result.image_b is img_b


class TestTournamentRound:
    """Tests for tournament round execution."""

    @patch("chatbot.preference_election.run_single_comparison")
    def test_run_tournament_round_basic(self, mock_compare):
        """Test basic tournament round execution."""
        mock_compare.return_value = ElectionMatch(
            image_a=np.zeros((10, 10, 3), dtype=np.uint8),
            image_b=np.zeros((10, 10, 3), dtype=np.uint8) + 128,
            winner="A",
        )

        pref = PreferenceData(50, [], [], "")
        images = [
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8) + 10,
            np.zeros((10, 10, 3), dtype=np.uint8) + 20,
            np.zeros((10, 10, 3), dtype=np.uint8) + 30,
        ]
        byes = []

        winners, remaining_byes = run_tournament_round(
            images, byes, pref, "lmstudio", {}
        )

        assert len(winners) == 2
        assert remaining_byes == []

    @patch("chatbot.preference_election.run_single_comparison")
    def test_run_tournament_round_with_bye(self, mock_compare):
        """Test tournament round with bye (odd image)."""
        mock_compare.return_value = ElectionMatch(
            image_a=np.zeros((10, 10, 3), dtype=np.uint8),
            image_b=np.zeros((10, 10, 3), dtype=np.uint8) + 128,
            winner="A",
        )

        pref = PreferenceData(50, [], [], "")
        images = [
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8) + 10,
        ]
        byes = [np.zeros((10, 10, 3), dtype=np.uint8) + 20]

        winners, remaining_byes = run_tournament_round(
            images, byes, pref, "lmstudio", {}
        )

        assert len(winners) == 1


class TestTagGeneration:
    """Tests for tag generation functionality."""

    def test_parse_tags_response_valid(self):
        """Test parsing valid tags response."""
        response = '{"tags": ["blue_eyes", "long_hair", "vibrant_colors"]}'
        result = parse_tags_response(response)

        assert result["tags"] == ["blue_eyes", "long_hair", "vibrant_colors"]

    def test_parse_tags_response_invalid(self):
        """Test handling invalid tags response."""
        response = "not json"
        result = parse_tags_response(response)

        assert result["tags"] == []

    def test_parse_tags_response_missing_tags(self):
        """Test handling response without tags key."""
        response = '{"other": "data"}'
        result = parse_tags_response(response)

        assert result["tags"] == []

    @patch("chatbot.preference_election.get_provider")
    def test_generate_tags_for_image(self, mock_get_provider):
        """Test generating tags for an image."""
        mock_provider = MagicMock()
        mock_provider.stream_chat.return_value = iter(
            ['{"tags": ["tag1", "tag2", "tag3"]}']
        )
        mock_get_provider.return_value = mock_provider

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = generate_tags_for_image(img, "lmstudio", {})

        assert "tag1" in result
        assert "tag2" in result
        assert "tag3" in result


class TestFullElection:
    """Tests for full election workflow."""

    @patch("chatbot.preference_election.generate_tags_for_image")
    @patch("chatbot.preference_election.run_tournament_round")
    @patch("chatbot.preference_election.run_single_comparison")
    def test_full_election_4_images(self, mock_compare, mock_round, mock_tags):
        """Test complete election with 4 images."""
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.zeros((10, 10, 3), dtype=np.uint8) + 10
        img3 = np.zeros((10, 10, 3), dtype=np.uint8) + 20
        img4 = np.zeros((10, 10, 3), dtype=np.uint8) + 30

        mock_compare.return_value = ElectionMatch(img1, img2, "A")
        mock_round.side_effect = [
            ([img1, img3], []),
            ([img1], []),
        ]
        mock_tags.return_value = ["tag1", "tag2"]

        pref = PreferenceData(50, [], [], "")
        result = run_full_election(pref, "lmstudio", {})

        assert result.winner_image is not None
        assert result.runner_up_image is not None

    @patch("chatbot.preference_election.generate_tags_for_image")
    @patch("chatbot.preference_election.run_tournament_round")
    @patch("chatbot.preference_election.run_single_comparison")
    def test_full_election_5_images_with_bye(self, mock_compare, mock_round, mock_tags):
        """Test complete election with 5 images (has bye handling)."""
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.zeros((10, 10, 3), dtype=np.uint8) + 10
        img3 = np.zeros((10, 10, 3), dtype=np.uint8) + 20
        img4 = np.zeros((10, 10, 3), dtype=np.uint8) + 30
        img5 = np.zeros((10, 10, 3), dtype=np.uint8) + 40

        mock_compare.return_value = ElectionMatch(img1, img2, "A")

        def round_side_effect(images, byes, pref, provider_type, config):
            if len(images) == 4:
                return [img1, img3], []
            elif len(images) == 3:
                return [img1, img5], []
            elif len(images) == 2:
                return [img1], []
            return [images[0]], []

        mock_round.side_effect = round_side_effect
        mock_tags.return_value = ["tag1", "tag2"]

        pref = PreferenceData(50, [], [], "")
        result = run_full_election(pref, "lmstudio", {})

        assert result.winner_image is not None
        assert result.runner_up_image is not None
