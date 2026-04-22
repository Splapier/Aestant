"""Tests for schema_viewer module and schema viewer UI functionality."""

import json
import pytest
from pathlib import Path

from chatbot.schema_manager import (
    load_master_schema,
    save_master_schema,
    get_protected_keys_at_level,
    parse_key_path,
    get_key_at_path,
    set_key_at_path,
    delete_key_at_path,
    validate_schema_change,
    flatten_schema_keys,
    find_key_path,
    get_field_descriptions,
    init_schema_schema,
)


@pytest.fixture
def sample_schema():
    return {
        "head": {
            "hair": {
                "color": {
                    "type": "string",
                    "default": None,
                    "confidence_score": 0.0,
                    "description": "Color of the hair.",
                },
            },
            "eyes": {
                "color": {
                    "type": "string",
                    "default": None,
                    "confidence_score": 0.0,
                    "description": "Color of the eyes.",
                },
            },
        },
        "upper_body": {
            "clothing": {
                "top": {
                    "type": "string",
                    "default": None,
                    "confidence_score": 0.0,
                    "description": "Type of top worn.",
                },
            },
        },
        "lower_body": {
            "legs": {
                "pants": {
                    "type": "string",
                    "default": None,
                    "confidence_score": 0.0,
                    "description": "Type of pants.",
                },
            },
        },
    }


@pytest.fixture
def temp_schema_file(tmp_path):
    schema_file = tmp_path / "test_master_schema.json"
    return schema_file


class TestProtectedKeys:
    """Test protected key identification (now dynamic)."""

    def test_level_0_protected_from_schema(self, sample_schema):
        protected = get_protected_keys_at_level(sample_schema, 0)
        assert "head" in protected
        assert "upper_body" in protected
        assert "lower_body" in protected

    def test_level_1_protected_from_schema(self, sample_schema):
        protected = get_protected_keys_at_level(sample_schema, 1)
        assert "hair" in protected
        assert "eyes" in protected
        assert "clothing" in protected
        assert "legs" in protected

    def test_validate_blocks_level_0_delete(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "delete", "head"
        )
        assert not allowed

    def test_validate_blocks_level_1_delete(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "delete", "head.hair"
        )
        assert not allowed

    def test_validate_allows_level_2_delete(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "delete", "head.hair.color"
        )
        assert allowed

    def test_validate_blocks_level_0_add(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "add", "new_top_level"
        )
        assert not allowed

    def test_validate_blocks_level_1_add(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "add", "head.new_key"
        )
        assert not allowed

    def test_validate_allows_level_2_add(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "add", "head.hair.new_color"
        )
        assert allowed


class TestSchemaLoading:
    """Test loading and initialization of master schema."""

    def test_load_returns_dict(self, tmp_path, sample_schema):
        schema_path = tmp_path / "master_schema.json"
        with open(schema_path, "w") as f:
            json.dump(sample_schema, f)

        import chatbot.schema_manager

        original = chatbot.schema_manager.MASTER_SCHEMA_PATH
        chatbot.schema_manager.MASTER_SCHEMA_PATH = schema_path
        try:
            schema = load_master_schema()
            assert isinstance(schema, dict)
            assert "head" in schema
        finally:
            chatbot.schema_manager.MASTER_SCHEMA_PATH = original

    def test_load_creates_default_if_missing(self, tmp_path):
        schema_path = tmp_path / "master_schema.json"
        assert not schema_path.exists()

        import chatbot.schema_manager

        original = chatbot.schema_manager.MASTER_SCHEMA_PATH
        chatbot.schema_manager.MASTER_SCHEMA_PATH = schema_path
        try:
            schema = load_master_schema()
            assert schema_path.exists()
            assert isinstance(schema, dict)
        finally:
            chatbot.schema_manager.MASTER_SCHEMA_PATH = original


class TestKeyOperations:
    """Test getting, setting, and deleting keys at paths."""

    def test_get_existing_key_returns_field(self, sample_schema):
        result = get_key_at_path(sample_schema, "head.hair.color")
        assert result is not None
        assert result["type"] == "string"
        assert result["description"] == "Color of the hair."

    def test_get_nonexistent_key_returns_none(self, sample_schema):
        result = get_key_at_path(sample_schema, "head.nonexistent")
        assert result is None

    def test_set_new_key_under_hair(self, sample_schema):
        new_field = {
            "type": "string",
            "default": None,
            "confidence_score": 0.0,
            "description": "Length of the hair.",
        }
        set_key_at_path(sample_schema, "head.hair.length", new_field)
        result = get_key_at_path(sample_schema, "head.hair.length")
        assert result is not None
        assert result["description"] == "Length of the hair."

    def test_delete_leaf_key(self, sample_schema):
        result = delete_key_at_path(sample_schema, "head.hair.color")
        assert result is True
        assert get_key_at_path(sample_schema, "head.hair.color") is None


class TestValidation:
    """Test schema change validation."""

    def test_cannot_delete_level_0_key(self, sample_schema):
        new_schema = dict(sample_schema)
        del new_schema["head"]
        allowed, msg = validate_schema_change(
            sample_schema, new_schema, "delete", "head"
        )
        assert allowed is False
        assert "Cannot delete level 0" in msg

    def test_cannot_delete_level_1_key(self, sample_schema):
        new_schema = dict(sample_schema)
        del new_schema["head"]["hair"]
        allowed, msg = validate_schema_change(
            sample_schema, new_schema, "delete", "head.hair"
        )
        assert allowed is False
        assert "Cannot delete level 1" in msg

    def test_can_delete_deeper_key(self, sample_schema):
        new_schema = dict(sample_schema)
        del new_schema["head"]["hair"]["color"]
        allowed, msg = validate_schema_change(
            sample_schema, new_schema, "delete", "head.hair.color"
        )
        assert allowed is True

    def test_cannot_add_level_0_key(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "add", "new_top_level"
        )
        assert allowed is False

    def test_cannot_add_level_1_key(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "add", "upper_body.new_section"
        )
        assert allowed is False

    def test_can_add_level_2_key(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "add", "head.hair.new_field"
        )
        assert allowed is True


class TestFlattenSchema:
    """Test flattening schema to leaf keys with paths."""

    def test_flattens_leaf_keys_with_paths(self, sample_schema):
        flat = flatten_schema_keys(sample_schema)
        assert "head.hair.color" in flat
        assert "head.eyes.color" in flat
        assert "upper_body.clothing.top" in flat
        assert "lower_body.legs.pantyhose" in flat

    def test_flatten_excludes_branch_nodes(self, sample_schema):
        flat = flatten_schema_keys(sample_schema)
        assert "head" not in flat
        assert "head.hair" not in flat
        assert "upper_body.clothing" not in flat


class TestHierarchicalGrouping:
    """Test grouping keys by their parent hierarchy."""

    def test_group_by_immediate_parent(self, sample_schema):
        flat = flatten_schema_keys(sample_schema)
        grouped = {}
        for path, field_def in flat.items():
            path_parts = parse_key_path(path)
            if len(path_parts) >= 2:
                parent_key = path_parts[-2]
                if parent_key not in grouped:
                    grouped[parent_key] = []
                grouped[parent_key].append((path, field_def))

        assert "hair" in grouped
        assert "eyes" in grouped
        assert "clothing" in grouped

    def test_path_shows_full_hierarchy(self, sample_schema):
        flat = flatten_schema_keys(sample_schema)
        path = "head.hair.color"
        assert path in flat
        path_parts = parse_key_path(path)
        parent_path = ".".join(path_parts[:-1])
        assert parent_path == "head.hair"


class TestFindKeyPath:
    """Test finding key paths in schema."""

    def test_finds_nested_key(self, sample_schema):
        path = find_key_path(sample_schema, "color")
        assert path is not None
        assert isinstance(path, list)
        assert len(path) >= 2

    def test_returns_none_for_nonexistent(self, sample_schema):
        result = find_key_path(sample_schema, "nonexistent_field")
        assert result is None


class TestFieldDescriptions:
    """Test generating field descriptions."""

    def test_generates_all_descriptions(self, sample_schema):
        desc = get_field_descriptions(sample_schema)
        assert "head.hair.color" in desc
        assert "Color of the hair" in desc


class TestCanAddUnderLowerKeys:
    """Test that users can add keys under lower level keys (hair, eyes, etc.)."""

    def test_can_add_under_hair(self, sample_schema):
        allowed, _ = validate_schema_change(
            sample_schema, sample_schema, "add", "head.hair.new_trait"
        )
        assert allowed is True

    def test_can_add_under_eyes(self, sample_schema):
        allowed, _ = validate_schema_change(
            sample_schema, sample_schema, "add", "head.eyes.new_trait"
        )
        assert allowed is True

    def test_can_add_under_clothing(self, sample_schema):
        allowed, _ = validate_schema_change(
            sample_schema, sample_schema, "add", "upper_body.clothing.new_trait"
        )
        assert allowed is True

    def test_can_add_under_legs(self, sample_schema):
        allowed, _ = validate_schema_change(
            sample_schema, sample_schema, "add", "lower_body.legs.new_trait"
        )
        assert allowed is True


class TestCanDeleteUserAddedKeys:
    """Test that user-added keys can be deleted."""

    def test_can_delete_user_added_key(self, sample_schema):
        set_key_at_path(
            sample_schema,
            "head.hair.user_trait",
            {
                "type": "string",
                "default": None,
                "confidence_score": 0.0,
                "description": "User added trait",
            },
        )
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "delete", "head.hair.user_trait"
        )
        assert allowed is True


class TestSchemaViewerDisplay:
    """Test helper functions for schema viewer display."""

    def test_builds_display_path_format(self):
        path_parts = ["head", "hair", "color"]
        display = " → ".join(path_parts)
        assert display == "head → hair → color"

    def test_identifies_leaf_vs_branch_keys(self, sample_schema):
        flat = flatten_schema_keys(sample_schema)
        for path in flat:
            path_parts = parse_key_path(path)
            if len(path_parts) >= 3:
                assert (
                    "color" in path_parts
                    or "top" in path_parts
                    or "pants" in path_parts
                )
