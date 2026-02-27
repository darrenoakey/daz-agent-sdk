from __future__ import annotations

import pytest

from agent_sdk.types import (
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    Tier,
)
from agent_sdk.providers.gemini import GeminiProvider, _classify_error, _build_prompt


# ##################################################################
# check if google-genai sdk and api key are available
# skip integration tests if not installed or configured
def _sdk_available() -> bool:
    try:
        from google import genai  # type: ignore[attr-defined]  # noqa: F401
        return True
    except ImportError:
        return False


def _api_key_available() -> bool:
    import os
    return bool(os.environ.get("GEMINI_API_KEY"))


_HAS_SDK = _sdk_available()
_HAS_API_KEY = _api_key_available()
_INTEGRATION_AVAILABLE = _HAS_SDK and _HAS_API_KEY


# ##################################################################
# test classify error
# verify error classification from exception messages
def test_classify_rate_limit() -> None:
    assert _classify_error(Exception("HTTP 429 Too Many Requests")) == ErrorKind.RATE_LIMIT
    assert _classify_error(Exception("quota exceeded")) == ErrorKind.RATE_LIMIT
    assert _classify_error(Exception("resource_exhausted")) == ErrorKind.RATE_LIMIT
    assert _classify_error(Exception("rate limit hit")) == ErrorKind.RATE_LIMIT


def test_classify_auth() -> None:
    assert _classify_error(Exception("401 Unauthorized")) == ErrorKind.AUTH
    assert _classify_error(Exception("403 Forbidden")) == ErrorKind.AUTH
    assert _classify_error(Exception("api_key invalid")) == ErrorKind.AUTH
    assert _classify_error(Exception("UNAUTHENTICATED")) == ErrorKind.AUTH


def test_classify_timeout() -> None:
    assert _classify_error(Exception("request timed out")) == ErrorKind.TIMEOUT
    assert _classify_error(Exception("deadline exceeded")) == ErrorKind.TIMEOUT


def test_classify_invalid() -> None:
    assert _classify_error(Exception("400 bad request invalid input")) == ErrorKind.INVALID_REQUEST
    assert _classify_error(Exception("invalid argument")) == ErrorKind.INVALID_REQUEST


def test_classify_internal() -> None:
    assert _classify_error(Exception("something weird happened")) == ErrorKind.INTERNAL
    assert _classify_error(Exception("500 Internal Server Error")) == ErrorKind.INTERNAL


# ##################################################################
# test build prompt
# verify message history is assembled correctly
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
    provider = GeminiProvider()
    assert provider.name == "gemini"


@pytest.mark.asyncio
async def test_available_returns_bool() -> None:
    provider = GeminiProvider()
    result = await provider.available()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_list_models_when_available() -> None:
    import os
    if not (_HAS_SDK and os.environ.get("GEMINI_API_KEY")):
        provider = GeminiProvider()
        models = await provider.list_models()
        assert models == []
        return

    provider = GeminiProvider()
    models = await provider.list_models()
    assert len(models) >= 3
    assert all(isinstance(m, ModelInfo) for m in models)
    ids = [m.model_id for m in models]
    assert "gemini-2.5-pro" in ids
    assert "gemini-2.5-flash" in ids
    assert "gemini-2.5-flash-lite" in ids


# ##################################################################
# test generate image raises
# gemini does not support image generation via this provider
@pytest.mark.asyncio
async def test_generate_image_raises() -> None:
    from pathlib import Path
    provider = GeminiProvider()
    with pytest.raises(NotImplementedError):
        await provider.generate_image("test", width=512, height=512, output=Path("/tmp/test.jpg"))


# ##################################################################
# integration tests
# these talk to the real Gemini API — skip if SDK not installed or
# GEMINI_API_KEY is not set in the environment
@pytest.mark.skipif(not _INTEGRATION_AVAILABLE, reason="google-genai not installed or GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_complete_simple() -> None:
    provider = GeminiProvider()
    models = await provider.list_models()
    flash_lite = next(m for m in models if m.tier == Tier.LOW)
    messages = [Message(role="user", content="What is 2+2? Reply with just the number.")]
    result = await provider.complete(messages, flash_lite, timeout=30.0)
    assert isinstance(result, Response)
    assert "4" in result.text


@pytest.mark.skipif(not _INTEGRATION_AVAILABLE, reason="google-genai not installed or GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_stream_simple() -> None:
    provider = GeminiProvider()
    models = await provider.list_models()
    flash_lite = next(m for m in models if m.tier == Tier.LOW)
    messages = [Message(role="user", content="Say hello in exactly one word.")]
    chunks: list[str] = []
    async for chunk in provider.stream(messages, flash_lite, timeout=30.0):
        chunks.append(chunk)
    full_text = "".join(chunks)
    assert len(full_text) > 0
