"""Tests for the structured output file-based extraction."""

import json
import os
import tempfile

from pydantic import BaseModel

from daz_agent_sdk.structured import (
    ensure_cwd,
    extract_result,
    schema_filename,
    schema_instructions,
)


class SampleSchema(BaseModel):
    name: str
    value: int


def test_schema_filename_unique() -> None:
    f1 = schema_filename()
    f2 = schema_filename()
    assert f1 != f2
    assert f1.startswith("_structured_output_")
    assert f1.endswith(".json")


def test_schema_instructions_contains_schema() -> None:
    instructions = schema_instructions(SampleSchema, "output.json")
    assert "output.json" in instructions
    assert "name" in instructions
    assert "value" in instructions
    assert "MUST produce" in instructions


def test_ensure_cwd_with_existing() -> None:
    cwd, created = ensure_cwd("/tmp/existing")
    assert cwd == "/tmp/existing"
    assert created is False


def test_ensure_cwd_creates_temp() -> None:
    cwd, created = ensure_cwd(None)
    assert created is True
    assert os.path.isdir(cwd)
    os.rmdir(cwd)


def test_extract_result_from_file() -> None:
    """When the file exists, reads from file."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = "output.json"
        filepath = os.path.join(tmp, filename)
        with open(filepath, "w") as f:
            json.dump({"name": "test", "value": 42}, f)

        result = extract_result(SampleSchema, filename, tmp, "")
        assert isinstance(result, SampleSchema)
        assert result.name == "test"
        assert result.value == 42
        # File should be cleaned up
        assert not os.path.exists(filepath)


def test_extract_result_from_response_text() -> None:
    """When file doesn't exist, falls back to response text."""
    with tempfile.TemporaryDirectory() as tmp:
        json_text = json.dumps({"name": "fallback", "value": 99})
        result = extract_result(SampleSchema, "nonexistent.json", tmp, json_text)
        assert isinstance(result, SampleSchema)
        assert result.name == "fallback"
        assert result.value == 99


def test_extract_result_from_markdown_fenced_response() -> None:
    """Handles markdown-fenced JSON in response text."""
    with tempfile.TemporaryDirectory() as tmp:
        json_text = '```json\n{"name": "fenced", "value": 7}\n```'
        result = extract_result(SampleSchema, "nonexistent.json", tmp, json_text)
        assert result.name == "fenced"
        assert result.value == 7


def test_extract_result_empty_raises() -> None:
    """Raises when both file and response text are empty."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            extract_result(SampleSchema, "nonexistent.json", tmp, "")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "empty" in str(e).lower()


def test_extract_result_from_prose_with_json() -> None:
    """Extracts JSON embedded at the end of a prose response."""
    with tempfile.TemporaryDirectory() as tmp:
        prose = (
            'The analysis has been written to the file.\n\n'
            'Key findings:\n- thing one\n- thing two\n\n'
            '{"name": "embedded", "value": 55}'
        )
        result = extract_result(SampleSchema, "nonexistent.json", tmp, prose)
        assert result.name == "embedded"
        assert result.value == 55


def test_extract_result_from_prose_with_fenced_json() -> None:
    """Extracts JSON from markdown fences in middle of prose."""
    with tempfile.TemporaryDirectory() as tmp:
        prose = (
            'Here is the analysis:\n\n'
            '```json\n{"name": "fenced_mid", "value": 77}\n```\n\n'
            'Let me know if you need anything else.'
        )
        result = extract_result(SampleSchema, "nonexistent.json", tmp, prose)
        assert result.name == "fenced_mid"
        assert result.value == 77


def test_extract_result_invalid_json_raises() -> None:
    """Raises when JSON is invalid."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            extract_result(SampleSchema, "nonexistent.json", tmp, "not json at all")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "parse failed" in str(e).lower()
