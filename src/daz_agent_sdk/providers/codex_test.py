from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from daz_agent_sdk.providers.codex import (
    CodexProvider,
    _build_prompt,
    _classify_error,
)
from daz_agent_sdk.types import (
    AgentError,
    Capability,
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    Tier,
)


# ##################################################################
# helper — make model info
def _make_model(model_id: str = "gpt-5.6-sol") -> ModelInfo:
    return ModelInfo(
        provider="codex",
        model_id=model_id,
        display_name="GPT-5.6 Sol",
        capabilities=frozenset(
            {Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}
        ),
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


def test_build_prompt_ignores_schema() -> None:
    """Schema is now handled separately via schema_instructions, not in _build_prompt."""
    messages = [Message(role="user", content="test")]
    result = _build_prompt(messages)
    assert result == "test"


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
    assert len(models) == 1
    for m in models:
        assert isinstance(m, ModelInfo)
        assert m.provider == "codex"
        assert Capability.TEXT in m.capabilities
    assert models[0].model_id == "gpt-5.6-sol"
    assert models[0].display_name == "GPT-5.6 Sol"


# ##################################################################
# integration tests — call real codex CLI
@pytest.mark.asyncio
async def test_complete_simple() -> None:
    provider = CodexProvider()
    messages = [
        Message(role="user", content="What is 2+2? Reply with just the number.")
    ]
    model = _make_model()
    resp = await provider.complete(messages, model, timeout=60.0)
    assert isinstance(resp, Response)
    assert "4" in resp.text
    assert resp.model_used is model
    assert resp.conversation_id is not None
    assert resp.turn_id is not None


@pytest.mark.asyncio
async def test_complete_structured() -> None:
    """Structured output via file-based extraction — codex returns text, SDK parses it."""
    from pydantic import BaseModel

    class MathResult(BaseModel):
        answer: int

    provider = CodexProvider()
    messages = [Message(role="user", content="What is 10 + 5?")]
    model = _make_model()
    result = await provider.complete(messages, model, schema=MathResult, timeout=60.0)
    assert isinstance(result, StructuredResponse)
    assert result.parsed.answer == 15


@pytest.mark.asyncio
async def test_stream_simple() -> None:
    provider = CodexProvider()
    messages = [
        Message(role="user", content="What is 2+2? Reply with just the number.")
    ]
    model = _make_model()
    chunks: list[str] = []
    async for chunk in provider.stream(messages, model, timeout=60.0):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    assert len(chunks) > 0
    full = "".join(chunks)
    assert "4" in full


@pytest.mark.asyncio
async def test_stream_timeout_reaps_real_codex_process() -> None:
    provider = CodexProvider()
    messages = [
        Message(
            role="user",
            content="Write a detailed analysis that cannot complete instantly.",
        )
    ]
    model = _make_model()

    async def consume() -> None:
        async for _ in provider.stream(messages, model, timeout=0.75):
            pass

    task = asyncio.create_task(consume())
    child_pid = 0
    for _ in range(50):
        process_list = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,comm="],
            capture_output=True,
            text=True,
            check=True,
        )
        codex_rows = []
        for row in process_list.stdout.splitlines():
            fields = row.strip().split(maxsplit=2)
            if (
                len(fields) == 3
                and int(fields[1]) == os.getpid()
                and Path(fields[2]).name == "codex"
            ):
                codex_rows.append(fields)
        if codex_rows:
            child_pid = int(codex_rows[0][0])
            break
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=0.01)
        except TimeoutError:
            pass

    assert child_pid > 0
    with pytest.raises(AgentError) as exc_info:
        await task
    assert exc_info.value.kind == ErrorKind.TIMEOUT
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
