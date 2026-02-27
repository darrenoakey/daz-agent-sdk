from __future__ import annotations

import os

import pytest
import pytest_asyncio  # noqa: F401 — registers asyncio mode

from agent_sdk.providers.codex import (
    CodexProvider,
    _build_messages,
    _classify_error,
    _import_sdk,
)
from agent_sdk.types import (
    Capability,
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    Tier,
)


# ##################################################################
# sdk availability
# check once at module level to gate integration tests
_HAS_SDK = _import_sdk() is not None
_HAS_API_KEY = bool(os.environ.get("OPENAI_API_KEY"))

skip_if_no_sdk = pytest.mark.skipif(not _HAS_SDK, reason="openai not installed")
skip_if_no_api = pytest.mark.skipif(
    not (_HAS_SDK and _HAS_API_KEY),
    reason="openai not installed or OPENAI_API_KEY not set",
)


# ##################################################################
# helper — make model info
# build a CodexProvider ModelInfo for use in integration tests
def _make_model(model_id: str = "gpt-4.1-mini") -> ModelInfo:
    return ModelInfo(
        provider="codex",
        model_id=model_id,
        display_name="GPT-4.1 Mini",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.MEDIUM,
    )


# ##################################################################
# classify error — unit tests (no SDK required)
def test_classify_rate_limit() -> None:
    err = Exception("HTTP 429 rate_limit exceeded")
    assert _classify_error(err) == ErrorKind.RATE_LIMIT


def test_classify_auth() -> None:
    err = Exception("401 Unauthorized auth failed")
    assert _classify_error(err) == ErrorKind.AUTH


def test_classify_timeout() -> None:
    err = Exception("Request timed out after 30s")
    assert _classify_error(err) == ErrorKind.TIMEOUT


def test_classify_invalid() -> None:
    err = Exception("400 invalid request body")
    assert _classify_error(err) == ErrorKind.INVALID_REQUEST


def test_classify_internal() -> None:
    err = Exception("Something completely unexpected happened")
    assert _classify_error(err) == ErrorKind.INTERNAL


# ##################################################################
# build messages — unit tests (no SDK required)
def test_build_messages_simple() -> None:
    messages = [Message(role="user", content="Hello")]
    result = _build_messages(messages)
    assert result == [{"role": "user", "content": "Hello"}]


def test_build_messages_with_system() -> None:
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hi"),
    ]
    result = _build_messages(messages)
    assert result == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]


def test_build_messages_with_history() -> None:
    messages = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="4"),
        Message(role="user", content="And 3+3?"),
    ]
    result = _build_messages(messages)
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "What is 2+2?"}
    assert result[1] == {"role": "assistant", "content": "4"}
    assert result[2] == {"role": "user", "content": "And 3+3?"}


# ##################################################################
# provider basics — unit tests (no SDK required)
def test_provider_name() -> None:
    provider = CodexProvider()
    assert provider.name == "codex"


@pytest.mark.asyncio
async def test_available_returns_bool() -> None:
    provider = CodexProvider()
    result = await provider.available()
    assert isinstance(result, bool)


@skip_if_no_sdk
@pytest.mark.asyncio
async def test_list_models_when_available() -> None:
    if not _HAS_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")
    provider = CodexProvider()
    models = await provider.list_models()
    assert len(models) == 3
    for m in models:
        assert isinstance(m, ModelInfo)
        assert m.provider == "codex"
        assert Capability.TEXT in m.capabilities


@pytest.mark.asyncio
async def test_generate_image_raises() -> None:
    from pathlib import Path

    provider = CodexProvider()
    with pytest.raises(NotImplementedError):
        await provider.generate_image("a cat", width=512, height=512, output=Path("/tmp/test.png"))


# ##################################################################
# integration tests — require openai SDK and OPENAI_API_KEY
@skip_if_no_api
@pytest.mark.asyncio
async def test_complete_simple() -> None:
    provider = CodexProvider()
    messages = [Message(role="user", content="What is 2+2? Reply with just the number.")]
    model = _make_model()
    resp = await provider.complete(messages, model, timeout=60.0)
    assert isinstance(resp, Response)
    assert "4" in resp.text
    assert resp.model_used is model
    assert resp.conversation_id is not None
    assert resp.turn_id is not None


@skip_if_no_api
@pytest.mark.asyncio
async def test_stream_simple() -> None:
    provider = CodexProvider()
    messages = [Message(role="user", content="Count from 1 to 3, one number per line.")]
    model = _make_model()
    chunks: list[str] = []
    async for chunk in provider.stream(messages, model, timeout=60.0):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    assert len(chunks) > 0
    full = "".join(chunks)
    assert full.strip() != ""
