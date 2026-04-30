"""Tests for new scanning and navigation functions in tagging_engine."""

from pathlib import Path

import pytest
from PIL import Image

from chatbot.tagging_engine import scan_untagged_images


class TestScanAllImages:
    """Test scanning for all images regardless of tag status."""

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

    def test_returns_empty_when_no_images(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import scan_all_images

        result = scan_all_images()
        assert result == []

    def test_returns_all_images_including_tagged(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import scan_all_images

        input_dir, dataset_dir = temp_dirs

        img1 = Image.new("RGB", (50, 50), color="red")
        img1_path = input_dir / "image1.png"
        img1.save(img1_path)

        img2 = Image.new("RGB", (50, 50), color="blue")
        img2_path = input_dir / "image2.png"
        img2.save(img2_path)

        dataset_file = dataset_dir / "image1.yaml"
        dataset_file.write_text("tags: {}")

        result = scan_all_images()
        assert str(img1_path) in result
        assert str(img2_path) in result
        assert len(result) == 2

    def test_returns_sorted_paths(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import scan_all_images

        input_dir, _ = temp_dirs

        for name in ["c.png", "a.png", "b.png"]:
            img = Image.new("RGB", (50, 50), color="white")
            img.save(input_dir / name)

        result = scan_all_images()
        stems = [Path(p).stem for p in result]
        assert stems == ["a", "b", "c"]

    def test_filters_non_image_files(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import scan_all_images

        input_dir, _ = temp_dirs

        img = Image.new("RGB", (50, 50), color="white")
        img.save(input_dir / "photo.png")
        (input_dir / "notes.txt").write_text("not an image")

        result = scan_all_images()
        assert len(result) == 1
        assert str(input_dir / "photo.png") in result


class TestFindFirstUntaggedIndex:
    """Test finding the first untagged image index."""

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

    def test_returns_zero_when_no_images(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import find_first_untagged_index

        result = find_first_untagged_index([])
        assert result == 0

    def test_returns_zero_when_first_untagged(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import find_first_untagged_index

        all_paths = ["/fake/image1.png", "/fake/image2.png"]
        result = find_first_untagged_index(all_paths)
        assert result == 0

    def test_returns_first_untagged_when_some_tagged(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import find_first_untagged_index

        input_dir, dataset_dir = temp_dirs

        img1 = Image.new("RGB", (50, 50), color="red")
        img1.save(input_dir / "image1.png")
        img2 = Image.new("RGB", (50, 50), color="blue")
        img2.save(input_dir / "image2.png")
        img3 = Image.new("RGB", (50, 50), color="green")
        img3.save(input_dir / "image3.png")

        (dataset_dir / "image1.yaml").write_text("tags: {}")
        (dataset_dir / "image2.yaml").write_text("tags: {}")

        all_paths = [
            str(input_dir / "image1.png"),
            str(input_dir / "image2.png"),
            str(input_dir / "image3.png"),
        ]

        result = find_first_untagged_index(all_paths)
        assert result == 2

    def test_returns_zero_when_all_tagged(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import find_first_untagged_index

        input_dir, dataset_dir = temp_dirs

        for i in range(3):
            img = Image.new("RGB", (50, 50), color="white")
            name = f"img{i}.png"
            img.save(input_dir / name)
            (dataset_dir / f"{name}.yaml").write_text("tags: {}")

        all_paths = sorted([str(input_dir / f"img{i}.png") for i in range(3)])

        result = find_first_untagged_index(all_paths)
        assert result == 0

    def test_returns_zero_when_mixed_but_first_available(self, temp_dirs, monkeypatch):
        from chatbot.tagging_engine import find_first_untagged_index

        input_dir, dataset_dir = temp_dirs

        for i in range(3):
            img = Image.new("RGB", (50, 50), color="white")
            name = f"img{i}.png"
            img.save(input_dir / name)

        (dataset_dir / "img1.yaml").write_text("tags: {}")

        all_paths = sorted([str(input_dir / f"img{i}.png") for i in range(3)])

        result = find_first_untagged_index(all_paths)
        assert result == 0
