from __future__ import annotations

import pytest
import pytest_asyncio  # noqa: F401  # pyright: ignore[reportUnusedImport]

from daz_agent_sdk.providers.codex import (
    CodexProvider,
    _build_prompt,
    _classify_error,
)
from daz_agent_sdk.types import (
    Capability,
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    Tier,
)


# ##################################################################
# helper — make model info
def _make_model(model_id: str = "gpt-5.3-codex") -> ModelInfo:
    return ModelInfo(
        provider="codex",
        model_id=model_id,
        display_name="GPT-5.3 Codex",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.HIGH,
    )


# ##################################################################
# classify error — unit tests
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
# build prompt — unit tests
def test_build_prompt_simple() -> None:
    messages = [Message(role="user", content="Hello")]
    result = _build_prompt(messages)
    assert result == "Hello"


def test_build_prompt_with_system() -> None:
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hi"),
    ]
    result = _build_prompt(messages)
    assert "[System]" in result
    assert "You are helpful." in result
    assert "Hi" in result


def test_build_prompt_with_history() -> None:
    messages = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="4"),
        Message(role="user", content="And 3+3?"),
    ]
    result = _build_prompt(messages)
    assert "What is 2+2?" in result
    assert "[Previous assistant response]" in result
    assert "And 3+3?" in result


def test_build_prompt_with_schema() -> None:
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        answer: str

    messages = [Message(role="user", content="test")]
    result = _build_prompt(messages, schema=TestSchema)
    assert "JSON" in result
    assert "answer" in result


# ##################################################################
# provider basics — unit tests
def test_provider_name() -> None:
    provider = CodexProvider()
    assert provider.name == "codex"


@pytest.mark.asyncio
async def test_available_returns_bool() -> None:
    provider = CodexProvider()
    result = await provider.available()
    assert isinstance(result, bool)
    assert result is True  # codex CLI must be installed


@pytest.mark.asyncio
async def test_list_models() -> None:
    provider = CodexProvider()
    models = await provider.list_models()
    assert len(models) == 2
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
# integration tests — call real codex CLI
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


@pytest.mark.asyncio
async def test_stream_simple() -> None:
    provider = CodexProvider()
    messages = [Message(role="user", content="What is 2+2? Reply with just the number.")]
    model = _make_model()
    chunks: list[str] = []
    async for chunk in provider.stream(messages, model, timeout=60.0):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    assert len(chunks) > 0
    full = "".join(chunks)
    assert "4" in full
