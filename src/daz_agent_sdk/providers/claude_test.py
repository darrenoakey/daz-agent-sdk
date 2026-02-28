from __future__ import annotations

import pytest

from daz_agent_sdk.types import (
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    Tier,
)
from daz_agent_sdk.providers.claude import ClaudeProvider, _classify_error, _build_prompt


# ##################################################################
# check if claude sdk is available
# skip all integration tests if not installed
def _sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_SDK = _sdk_available()


# ##################################################################
# test classify error
# verify error classification from exception messages
def test_classify_rate_limit() -> None:
    assert _classify_error(Exception("rate_limit_event")) == ErrorKind.RATE_LIMIT
    assert _classify_error(Exception("HTTP 429")) == ErrorKind.RATE_LIMIT
    assert _classify_error(Exception("overloaded")) == ErrorKind.RATE_LIMIT


def test_classify_auth() -> None:
    assert _classify_error(Exception("401 Unauthorized")) == ErrorKind.AUTH
    assert _classify_error(Exception("403 Forbidden")) == ErrorKind.AUTH


def test_classify_timeout() -> None:
    assert _classify_error(Exception("request timed out")) == ErrorKind.TIMEOUT


def test_classify_invalid() -> None:
    assert _classify_error(Exception("400 bad request invalid")) == ErrorKind.INVALID_REQUEST


def test_classify_internal() -> None:
    assert _classify_error(Exception("something weird happened")) == ErrorKind.INTERNAL


# ##################################################################
# test build prompt
# verify message history is concatenated correctly
def test_build_prompt_simple() -> None:
    messages = [Message(role="user", content="hello")]
    result = _build_prompt(messages, schema=None)
    assert result == "hello"


def test_build_prompt_with_system() -> None:
    messages = [
        Message(role="system", content="you are helpful"),
        Message(role="user", content="hi"),
    ]
    result = _build_prompt(messages, schema=None)
    assert "[System]" in result
    assert "you are helpful" in result
    assert "hi" in result


def test_build_prompt_with_history() -> None:
    messages = [
        Message(role="user", content="first"),
        Message(role="assistant", content="response"),
        Message(role="user", content="second"),
    ]
    result = _build_prompt(messages, schema=None)
    assert "first" in result
    assert "[Previous assistant response]" in result
    assert "second" in result


def test_build_prompt_with_schema() -> None:
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        answer: str

    messages = [Message(role="user", content="test")]
    result = _build_prompt(messages, schema=TestSchema)
    assert "JSON" in result
    assert "answer" in result


# ##################################################################
# test provider basics
# verify class attributes and availability detection
def test_provider_name() -> None:
    provider = ClaudeProvider()
    assert provider.name == "claude"


@pytest.mark.asyncio
async def test_available_returns_bool() -> None:
    provider = ClaudeProvider()
    result = await provider.available()
    assert isinstance(result, bool)
    assert result == _HAS_SDK


@pytest.mark.asyncio
async def test_list_models_when_available() -> None:
    provider = ClaudeProvider()
    models = await provider.list_models()
    if _HAS_SDK:
        assert len(models) >= 3
        assert all(isinstance(m, ModelInfo) for m in models)
        names = [m.model_id for m in models]
        assert "claude-opus-4-6" in names
        assert "claude-sonnet-4-6" in names
        assert "claude-haiku-4-5-20251001" in names
    else:
        assert models == []


# ##################################################################
# test generate image raises
# claude does not support image generation
@pytest.mark.asyncio
async def test_generate_image_raises() -> None:
    from pathlib import Path
    provider = ClaudeProvider()
    with pytest.raises(NotImplementedError):
        await provider.generate_image("test", width=512, height=512, output=Path("/tmp/test.jpg"))


# ##################################################################
# integration tests
# these talk to the real claude sdk — skip if not installed
@pytest.mark.skipif(not _HAS_SDK, reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_complete_simple() -> None:
    provider = ClaudeProvider()
    models = await provider.list_models()
    haiku = next(m for m in models if m.tier == Tier.LOW)
    messages = [Message(role="user", content="What is 2+2? Reply with just the number.")]
    result = await provider.complete(messages, haiku, timeout=30.0)
    assert isinstance(result, Response)
    assert "4" in result.text


@pytest.mark.skipif(not _HAS_SDK, reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_complete_structured() -> None:
    from pydantic import BaseModel

    class MathResult(BaseModel):
        answer: int

    provider = ClaudeProvider()
    models = await provider.list_models()
    haiku = next(m for m in models if m.tier == Tier.LOW)
    messages = [Message(role="user", content="What is 10 + 5?")]
    result = await provider.complete(messages, haiku, schema=MathResult, timeout=30.0)
    assert isinstance(result, StructuredResponse)
    assert result.parsed.answer == 15


@pytest.mark.skipif(not _HAS_SDK, reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_stream_simple() -> None:
    provider = ClaudeProvider()
    models = await provider.list_models()
    haiku = next(m for m in models if m.tier == Tier.LOW)
    messages = [Message(role="user", content="Say hello in exactly one word.")]
    chunks: list[str] = []
    async for chunk in provider.stream(messages, haiku, timeout=30.0):
        chunks.append(chunk)
    full_text = "".join(chunks)
    assert len(full_text) > 0
