from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import BaseModel

from agent_sdk.types import (
    AgentError,
    AudioResult,
    Capability,
    ErrorKind,
    ImageResult,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    Tier,
    parse_json_from_llm,
)
from pathlib import Path


# ##################################################################
# tier values
# verify all five tiers have the expected string values
def test_tier_values() -> None:
    assert Tier.HIGH.value == "high"
    assert Tier.MEDIUM.value == "medium"
    assert Tier.LOW.value == "low"
    assert Tier.FREE_FAST.value == "free_fast"
    assert Tier.FREE_THINKING.value == "free_thinking"


# ##################################################################
# capability values
# verify all six capabilities exist
def test_capability_values() -> None:
    assert len(Capability) == 6
    assert Capability.TEXT.value == "text"
    assert Capability.IMAGE.value == "image"


# ##################################################################
# error kind values
# verify all error classifications exist
def test_error_kind_values() -> None:
    assert len(ErrorKind) == 6
    assert ErrorKind.RATE_LIMIT.value == "rate_limit"
    assert ErrorKind.AUTH.value == "auth"


# ##################################################################
# model info creation and qualified name
# verify frozen dataclass and property
def test_model_info_qualified_name() -> None:
    info = ModelInfo(
        provider="claude",
        model_id="claude-opus-4-6",
        display_name="Claude Opus",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.HIGH,
    )
    assert info.qualified_name == "claude:claude-opus-4-6"
    assert info.supports_streaming is True
    assert info.supports_tools is False


# ##################################################################
# model info is frozen
# confirm immutability
def test_model_info_frozen() -> None:
    info = ModelInfo(
        provider="ollama",
        model_id="qwen3-8b",
        display_name="Qwen3 8B",
        capabilities=frozenset({Capability.TEXT}),
        tier=Tier.FREE_FAST,
    )
    with pytest.raises(AttributeError):
        info.provider = "changed"  # type: ignore[misc]


# ##################################################################
# message to dict
# verify serialisation with and without metadata
def test_message_to_dict_without_metadata() -> None:
    msg = Message(role="user", content="hello")
    d = msg.to_dict()
    assert d == {"role": "user", "content": "hello"}
    assert "metadata" not in d


def test_message_to_dict_with_metadata() -> None:
    msg = Message(role="assistant", content="hi", metadata={"tokens": 5})
    d = msg.to_dict()
    assert d["metadata"] == {"tokens": 5}


# ##################################################################
# response fields
# verify all fields are accessible
def test_response_fields() -> None:
    model = ModelInfo(
        provider="test",
        model_id="test-1",
        display_name="Test",
        capabilities=frozenset({Capability.TEXT}),
        tier=Tier.LOW,
    )
    cid = uuid4()
    tid = uuid4()
    resp = Response(text="answer", model_used=model, conversation_id=cid, turn_id=tid)
    assert resp.text == "answer"
    assert resp.model_used.provider == "test"
    assert resp.usage == {}


# ##################################################################
# structured response
# verify parsed field carries pydantic instance
def test_structured_response() -> None:
    class Score(BaseModel):
        value: int

    model = ModelInfo(
        provider="test",
        model_id="test-1",
        display_name="Test",
        capabilities=frozenset({Capability.STRUCTURED}),
        tier=Tier.LOW,
    )
    parsed = Score(value=42)
    resp = StructuredResponse(
        text='{"value": 42}',
        model_used=model,
        conversation_id=uuid4(),
        turn_id=uuid4(),
        parsed=parsed,
    )
    assert resp.parsed.value == 42
    assert isinstance(resp, Response)


# ##################################################################
# image result fields
# verify path and dimensions
def test_image_result() -> None:
    model = ModelInfo(
        provider="local",
        model_id="z-image-turbo",
        display_name="Z-Image Turbo",
        capabilities=frozenset({Capability.IMAGE}),
        tier=Tier.HIGH,
    )
    result = ImageResult(
        path=Path("/tmp/test.jpg"),
        model_used=model,
        conversation_id=uuid4(),
        prompt="a cat",
        width=512,
        height=512,
    )
    assert result.width == 512
    assert result.prompt == "a cat"


# ##################################################################
# audio result fields
# verify voice and optional duration
def test_audio_result() -> None:
    model = ModelInfo(
        provider="local",
        model_id="qwen3-tts",
        display_name="Qwen3 TTS",
        capabilities=frozenset({Capability.TTS}),
        tier=Tier.HIGH,
    )
    result = AudioResult(
        path=Path("/tmp/test.mp3"),
        model_used=model,
        conversation_id=uuid4(),
        text="hello world",
        voice="gary",
    )
    assert result.duration_seconds is None
    assert result.voice == "gary"


# ##################################################################
# agent error
# verify structured error with attempts
def test_agent_error_to_dict() -> None:
    err = AgentError(
        "rate limited on all providers",
        kind=ErrorKind.RATE_LIMIT,
        attempts=[
            {"provider": "claude", "status": 429},
            {"provider": "codex", "status": 429},
        ],
    )
    d = err.to_dict()
    assert d["kind"] == "rate_limit"
    assert len(d["attempts"]) == 2
    assert "rate limited" in d["error"]


def test_agent_error_is_exception() -> None:
    err = AgentError("test", kind=ErrorKind.INTERNAL)
    with pytest.raises(AgentError):
        raise err


# ##################################################################
# parse json from llm
# verify markdown code block stripping and plain json
def test_parse_json_plain() -> None:
    result = parse_json_from_llm('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_with_code_block() -> None:
    text = '```json\n{"key": "value"}\n```'
    result = parse_json_from_llm(text)
    assert result == {"key": "value"}


def test_parse_json_with_bare_code_block() -> None:
    text = '```\n{"items": [1, 2, 3]}\n```'
    result = parse_json_from_llm(text)
    assert result == {"items": [1, 2, 3]}


def test_parse_json_invalid_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_from_llm("not json at all")


def test_parse_json_with_whitespace() -> None:
    text = '  \n  {"a": 1}  \n  '
    result = parse_json_from_llm(text)
    assert result == {"a": 1}
