from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest
import pytest_asyncio  # noqa: F401 — registers asyncio mode
from pydantic import BaseModel

from daz_agent_sdk.providers.arbiter import ArbiterProvider, _KNOWN_TIERS
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
# arbiter reachability check
# probe the default arbiter endpoint at module load so online tests
# skip quickly when the spark machine is offline. avoids per-test
# try/except noise and long timeouts in CI.
_DEFAULT_URL = "http://10.0.0.254:8400"


def _arbiter_reachable() -> bool:
    parsed = urlparse(_DEFAULT_URL)
    host = parsed.hostname or "10.0.0.254"
    port = parsed.port or 8400
    try:
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        return True
    except OSError:
        return False


ARBITER_RUNNING = _arbiter_reachable()
skip_if_no_arbiter = pytest.mark.skipif(
    not ARBITER_RUNNING,
    reason="Arbiter not reachable at 10.0.0.254:8400",
)

# ##################################################################
# test model selection
# use qwen3.6-27b as the test model.
TEST_MODEL_ID = "qwen3.6-27b"


# ##################################################################
# helper — make model info
# build a ModelInfo for the named arbiter LLM. used when passing a
# model into provider methods without first calling list_models().
def _make_model(model_id: str = TEST_MODEL_ID) -> ModelInfo:
    return ModelInfo(
        provider="arbiter",
        model_id=model_id,
        display_name="Test Model",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.FREE_THINKING,
    )


# ##################################################################
# known tiers — unit tests (no arbiter required)
# verify the hardcoded tier mapping covers the registered llm_names.
def test_known_tiers_contains_qwen() -> None:
    assert _KNOWN_TIERS["qwen3.6-27b"] == Tier.SUMMARIES


# ##################################################################
# constructor — unit tests
def test_constructor_defaults_to_spark_arbiter() -> None:
    provider = ArbiterProvider()
    assert provider._base_url == "http://10.0.0.254:8400"


def test_constructor_strips_trailing_slash() -> None:
    provider = ArbiterProvider(base_url="http://example:9000/")
    assert provider._base_url == "http://example:9000"


# ##################################################################
# available — online tests
@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_available_returns_true() -> None:
    provider = ArbiterProvider()
    result = await provider.available()
    assert result is True


@pytest.mark.asyncio
async def test_available_wrong_port_returns_false() -> None:
    provider = ArbiterProvider(base_url="http://localhost:19999")
    result = await provider.available()
    assert result is False


# ##################################################################
# list models — online tests
@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_list_models_returns_model_info_instances() -> None:
    provider = ArbiterProvider()
    models = await provider.list_models()
    assert len(models) > 0
    for m in models:
        assert isinstance(m, ModelInfo)
        assert m.provider == "arbiter"
        assert m.model_id != ""
        assert Capability.TEXT in m.capabilities
        assert Capability.STRUCTURED in m.capabilities
        assert isinstance(m.tier, Tier)


@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_list_models_includes_qwen() -> None:
    provider = ArbiterProvider()
    models = await provider.list_models()
    names = {m.model_id for m in models}
    assert "qwen3.6-27b" in names


@pytest.mark.asyncio
async def test_list_models_wrong_port_returns_empty() -> None:
    provider = ArbiterProvider(base_url="http://localhost:19999")
    models = await provider.list_models()
    assert models == []


# ##################################################################
# complete — online tests
# generous timeouts so the first call can warm a cold model.
@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_complete_simple_prompt() -> None:
    provider = ArbiterProvider()
    messages = [Message(role="user", content="What is 2+2? Reply with just the number.")]
    model = _make_model()
    resp = await provider.complete(messages, model, timeout=180.0)
    assert isinstance(resp, Response)
    assert "4" in resp.text
    assert resp.model_used is model
    assert resp.conversation_id is not None
    assert resp.turn_id is not None


@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_complete_with_system_message() -> None:
    provider = ArbiterProvider()
    messages = [
        Message(role="system", content="You are a terse assistant. Be brief."),
        Message(role="user", content="What is the capital of France?"),
    ]
    model = _make_model()
    resp = await provider.complete(messages, model, timeout=180.0)
    assert isinstance(resp, Response)
    assert "Paris" in resp.text


# ##################################################################
# structured output — online tests
@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_complete_structured_output() -> None:
    class MathAnswer(BaseModel):
        result: int
        explanation: str

    provider = ArbiterProvider()
    messages = [Message(role="user", content="What is 3 multiplied by 7?")]
    model = _make_model()
    resp = await provider.complete(messages, model, schema=MathAnswer, timeout=300.0)
    assert isinstance(resp, StructuredResponse)
    assert isinstance(resp.parsed, MathAnswer)
    assert resp.parsed.result == 21


# ##################################################################
# stream — online tests
@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_stream_yields_string_chunks() -> None:
    provider = ArbiterProvider()
    messages = [Message(role="user", content="Count from 1 to 5, one number per line.")]
    model = _make_model()
    chunks: list[str] = []
    async for chunk in provider.stream(messages, model, timeout=180.0):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    assert len(chunks) > 0
    full = "".join(chunks)
    assert full.strip() != ""


@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_stream_produces_complete_response() -> None:
    provider = ArbiterProvider()
    messages = [Message(role="user", content="What is 10 divided by 2? Reply with just the number.")]
    model = _make_model()
    full = ""
    async for chunk in provider.stream(messages, model, timeout=180.0):
        full += chunk
    assert "5" in full


# ##################################################################
# connection error handling
@pytest.mark.asyncio
async def test_complete_wrong_port_raises_agent_error() -> None:
    provider = ArbiterProvider(base_url="http://localhost:19999")
    messages = [Message(role="user", content="hello")]
    model = _make_model()
    with pytest.raises(AgentError) as exc_info:
        await provider.complete(messages, model, timeout=5.0)
    assert exc_info.value.kind == ErrorKind.NOT_AVAILABLE


@pytest.mark.asyncio
async def test_stream_wrong_port_raises_agent_error() -> None:
    provider = ArbiterProvider(base_url="http://localhost:19999")
    messages = [Message(role="user", content="hello")]
    model = _make_model()
    with pytest.raises(AgentError) as exc_info:
        async for _ in provider.stream(messages, model, timeout=5.0):
            pass
    assert exc_info.value.kind == ErrorKind.NOT_AVAILABLE


# ##################################################################
# qwen3.6-27b — dedicated reasoning-model test
# verifies the reasoning→content fallback path with the default
# low/free_fast/free_thinking tier model. marked slow because a cold
# qwen load is ~10 minutes.
@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_complete_qwen_reasoning_model() -> None:
    provider = ArbiterProvider()
    messages = [Message(role="user", content="What is 2+2? Answer with just the number.")]
    model = ModelInfo(
        provider="arbiter",
        model_id="qwen3.6-27b",
        display_name="Qwen3.6 27B",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.FREE_THINKING,
    )
    resp = await provider.complete(messages, model, timeout=900.0)
    assert isinstance(resp, Response)
    # content may be null when max_tokens runs out mid-reasoning; our
    # provider falls back to the reasoning field so text is never empty
    assert resp.text.strip() != ""
