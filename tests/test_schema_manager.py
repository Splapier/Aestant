"""Tests for schema_manager module."""

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
                "pantyhose": {
                    "type": "string",
                    "default": None,
                    "confidence_score": 0.0,
                    "description": "Type of pantyhose.",
                },
            },
        },
    }


class TestProtectedKeys:
    """Test protected key identification (now dynamic from schema structure)."""

    def test_level_0_protected_from_schema(self, sample_schema):
        protected = get_protected_keys_at_level(sample_schema, 0)
        assert "head" in protected
        assert "upper_body" in protected
        assert "lower_body" in protected

    def test_level_1_protected_includes_from_schema(self, sample_schema):
        protected = get_protected_keys_at_level(sample_schema, 1)
        assert "hair" in protected
        assert "eyes" in protected

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


class TestParseKeyPath:
    """Test key path parsing."""

    def test_simple_path(self):
        assert parse_key_path("body_regions") == ["body_regions"]

    def test_nested_path(self):
        assert parse_key_path("body_regions.head.hair.color") == [
            "body_regions",
            "head",
            "hair",
            "color",
        ]

    def test_empty_path(self):
        assert parse_key_path("") == []


class TestGetSetKeyAtPath:
    """Test getting and setting keys at paths."""

    def test_get_existing_key(self, sample_schema):
        result = get_key_at_path(sample_schema, "head.hair.color")
        assert result is not None
        assert result["type"] == "string"

    def test_get_nonexistent_key(self, sample_schema):
        result = get_key_at_path(sample_schema, "head.nonexistent")
        assert result is None

    def test_set_new_key(self, sample_schema):
        new_field = {
            "type": "string",
            "default": None,
            "confidence_score": 0.0,
            "description": "New field",
        }
        set_key_at_path(sample_schema, "head.hair.length", new_field)
        result = get_key_at_path(sample_schema, "head.hair.length")
        assert result is not None
        assert result["description"] == "New field"


class TestDeleteKeyAtPath:
    """Test key deletion."""

    def test_delete_leaf_key(self, sample_schema):
        result = delete_key_at_path(sample_schema, "head.hair.color")
        assert result is True

    def test_delete_nonexistent(self, sample_schema):
        result = delete_key_at_path(sample_schema, "head.nothing")
        assert result is False

    def test_delete_nested_key(self, sample_schema):
        delete_key_at_path(sample_schema, "head.hair")
        result = get_key_at_path(sample_schema, "head.hair")
        assert result is None


class TestValidateSchemaChange:
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

    def test_can_delete_inner_key(self, sample_schema):
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
            sample_schema, sample_schema, "add", "head.new_section"
        )
        assert allowed is False

    def test_can_add_inner_key(self, sample_schema):
        allowed, msg = validate_schema_change(
            sample_schema, sample_schema, "add", "head.hair.new_field"
        )
        assert allowed is True


class TestFlattenSchemaKeys:
    """Test schema flattening."""

    def test_flattens_all_leaf_keys(self, sample_schema):
        flat = flatten_schema_keys(sample_schema)
        assert "head.hair.color" in flat
        assert "upper_body.clothing.top" in flat
        assert "lower_body.legs.pantyhose" in flat


class TestFindKeyPath:
    """Test finding key paths."""

    def test_finds_nested_key(self, sample_schema):
        path = find_key_path(sample_schema, "color")
        assert path is not None


class TestFieldDescriptions:
    """Test field description generation."""

    def test_generates_descriptions(self, sample_schema):
        desc = get_field_descriptions(sample_schema)
        assert "head.hair.color" in desc
        assert "Color of the hair" in desc


class TestLoadSaveSchema:
    """Test loading and saving master schema."""

    def test_load_returns_dict(self):
        schema = load_master_schema()
        assert isinstance(schema, dict)
        assert "head" in schema

    def test_save_and_reload(self, tmp_path, sample_schema):
        temp_path = tmp_path / "test_schema.json"
        import chatbot.schema_manager

        original = chatbot.schema_manager.MASTER_SCHEMA_PATH
        chatbot.schema_manager.MASTER_SCHEMA_PATH = temp_path

        try:
            save_master_schema(sample_schema)
            loaded = load_master_schema()
            assert loaded == sample_schema
        finally:
            chatbot.schema_manager.MASTER_SCHEMA_PATH = original
