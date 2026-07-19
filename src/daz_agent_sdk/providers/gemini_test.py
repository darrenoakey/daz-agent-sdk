from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from daz_agent_sdk.providers.gemini import (
    GeminiProvider,
    _classify_error,
    _build_prompt,
    _find_gemini_cli,
)
from daz_agent_sdk.types import (
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    Tier,
)


# ##################################################################
# classify error — unit tests
def test_classify_rate_limit() -> None:
    assert (
        _classify_error(Exception("HTTP 429 Too Many Requests")) == ErrorKind.RATE_LIMIT
    )
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
    assert (
        _classify_error(Exception("400 bad request invalid input"))
        == ErrorKind.INVALID_REQUEST
    )
    assert _classify_error(Exception("invalid argument")) == ErrorKind.INVALID_REQUEST


def test_classify_internal() -> None:
    assert _classify_error(Exception("something weird happened")) == ErrorKind.INTERNAL
    assert _classify_error(Exception("500 Internal Server Error")) == ErrorKind.INTERNAL


# ##################################################################
# build prompt — unit tests
def test_build_prompt_simple() -> None:
    messages = [Message(role="user", content="hello")]
    result = _build_prompt(messages)
    assert result == "hello"


def test_build_prompt_with_system() -> None:
    messages = [
        Message(role="system", content="you are helpful"),
        Message(role="user", content="hi"),
    ]
    result = _build_prompt(messages)
    assert "[System]" in result
    assert "you are helpful" in result
    assert "hi" in result


def test_build_prompt_with_history() -> None:
    messages = [
        Message(role="user", content="first"),
        Message(role="assistant", content="response"),
        Message(role="user", content="second"),
    ]
    result = _build_prompt(messages)
    assert "first" in result
    assert "[Previous assistant response]" in result
    assert "second" in result


def test_build_prompt_ignores_schema() -> None:
    """Schema is now handled separately via schema_instructions, not in _build_prompt."""
    messages = [Message(role="user", content="test")]
    result = _build_prompt(messages)
    assert result == "test"


# ##################################################################
# provider basics — unit tests
def test_provider_name() -> None:
    provider = GeminiProvider()
    assert provider.name == "gemini"


def test_find_gemini_cli_outside_path() -> None:
    restricted_path = "/usr/bin:/bin:/usr/sbin:/sbin"
    assert shutil.which("gemini", path=restricted_path) is None
    assert _find_gemini_cli(restricted_path) == "/opt/homebrew/bin/gemini"


@pytest.mark.asyncio
async def test_available_returns_bool() -> None:
    provider = GeminiProvider()
    result = await provider.available()
    assert isinstance(result, bool)
    assert result is True  # gemini CLI must be installed


@pytest.mark.asyncio
async def test_list_models() -> None:
    provider = GeminiProvider()
    models = await provider.list_models()
    assert len(models) == 3
    assert all(isinstance(m, ModelInfo) for m in models)
    ids = [m.model_id for m in models]
    assert "gemini-2.5-pro" in ids
    assert "gemini-2.5-flash" in ids
    assert "gemini-2.5-flash-lite" in ids


# ##################################################################
# integration tests — execute the real subprocess protocol through a local CLI.
@pytest.fixture
def gemini_cli(tmp_path: Path) -> Iterator[Path]:
    executable = tmp_path / "gemini"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys

prompt = sys.stdin.read()
output_format = sys.argv[sys.argv.index("-o") + 1]
if output_format == "stream-json":
    print(json.dumps({"type": "message", "role": "assistant", "content": "4"}))
    print(json.dumps({"type": "result", "status": "success"}))
else:
    response = '{"answer":15}' if '10 + 5' in prompt else "4"
    print(json.dumps({"response": response, "stats": {}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}{os.pathsep}{original_path}"
    try:
        yield executable
    finally:
        os.environ["PATH"] = original_path


@pytest.mark.asyncio
async def test_complete_simple(gemini_cli: Path) -> None:
    assert gemini_cli.is_file()
    provider = GeminiProvider()
    models = await provider.list_models()
    flash_lite = next(m for m in models if m.tier == Tier.LOW)
    messages = [
        Message(role="user", content="What is 2+2? Reply with just the number.")
    ]
    result = await provider.complete(messages, flash_lite, timeout=30.0)
    assert isinstance(result, Response)
    assert "4" in result.text


@pytest.mark.asyncio
async def test_complete_structured(gemini_cli: Path) -> None:
    """Structured output via file-based extraction — gemini returns text, SDK parses it."""
    assert gemini_cli.is_file()
    from pydantic import BaseModel

    class MathResult(BaseModel):
        answer: int

    provider = GeminiProvider()
    models = await provider.list_models()
    flash_lite = next(m for m in models if m.tier == Tier.LOW)
    messages = [Message(role="user", content="What is 10 + 5?")]
    result = await provider.complete(
        messages, flash_lite, schema=MathResult, timeout=60.0
    )
    assert isinstance(result, StructuredResponse)
    assert result.parsed.answer == 15


@pytest.mark.asyncio
async def test_stream_simple(gemini_cli: Path) -> None:
    assert gemini_cli.is_file()
    provider = GeminiProvider()
    models = await provider.list_models()
    flash_lite = next(m for m in models if m.tier == Tier.LOW)
    messages = [
        Message(role="user", content="What is 2+2? Reply with just the number.")
    ]
    chunks: list[str] = []
    async for chunk in provider.stream(messages, flash_lite, timeout=30.0):
        chunks.append(chunk)
    full_text = "".join(chunks)
    assert len(full_text) > 0
    assert "4" in full_text
