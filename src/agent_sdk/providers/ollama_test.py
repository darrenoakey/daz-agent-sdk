from __future__ import annotations

import socket

import pytest
import pytest_asyncio  # noqa: F401 — registers asyncio mode

from agent_sdk.providers.ollama import OllamaProvider, _tier_from_param_size
from agent_sdk.types import (
    Capability,
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    Tier,
    AgentError,
)
from pydantic import BaseModel


# ##################################################################
# ollama reachability check
# probe at module level so tests skip quickly when ollama is offline.
# avoids per-test try/except noise.
def _ollama_reachable() -> bool:
    try:
        s = socket.create_connection(("localhost", 11434), timeout=2)
        s.close()
        return True
    except OSError:
        return False


OLLAMA_RUNNING = _ollama_reachable()
skip_if_no_ollama = pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running on localhost:11434")

# ##################################################################
# test model selection
# prefer a model already loaded in Ollama VRAM (instant response).
# falls back to the first listed model if nothing is warm.
# raises a clear error if no models are available at all.
# ##################################################################
# vision-only models
# models that cannot do text chat — skip them when picking a test model
_VISION_ONLY = {"moondream", "llava", "bakllava"}


# ##################################################################
# is text capable
# returns True if the model name is not a known vision-only model
def _is_text_capable(name: str) -> bool:
    base = name.split(":")[0].lower()
    return base not in _VISION_ONLY


def _pick_test_model() -> str:
    if not OLLAMA_RUNNING:
        return "phi3:latest"
    import urllib.request
    import json as _json

    # prefer a warm text model
    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=3) as r:
            ps = _json.loads(r.read())
            warm = ps.get("models", [])
            for m in warm:
                if _is_text_capable(m["name"]):
                    return m["name"]
    except Exception:
        pass

    # fall back to any installed text model
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            tags = _json.loads(r.read())
            models = tags.get("models", [])
            for m in models:
                if _is_text_capable(m.get("name", "")):
                    return m["name"]
    except Exception:
        pass

    return "phi3:latest"


TEST_MODEL_ID = _pick_test_model()


# ##################################################################
# helper — make model info
# build a ModelInfo for TEST_MODEL_ID to pass into provider methods.
def _make_model(model_id: str = TEST_MODEL_ID) -> ModelInfo:
    return ModelInfo(
        provider="ollama",
        model_id=model_id,
        display_name="Test Model",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.FREE_FAST,
    )


# ##################################################################
# tier from param size — unit tests (no ollama required)
# verify the parameter-size-to-tier mapping logic independently.
def test_tier_from_param_size_small() -> None:
    assert _tier_from_param_size("3.8B") == Tier.FREE_FAST


def test_tier_from_param_size_large() -> None:
    assert _tier_from_param_size("36.0B") == Tier.FREE_THINKING


def test_tier_from_param_size_medium_boundary() -> None:
    assert _tier_from_param_size("20B") == Tier.FREE_FAST
    assert _tier_from_param_size("20.1B") == Tier.FREE_THINKING


def test_tier_from_param_size_megabytes() -> None:
    assert _tier_from_param_size("137M") == Tier.FREE_FAST


def test_tier_from_param_size_empty() -> None:
    assert _tier_from_param_size("") == Tier.FREE_FAST


def test_tier_from_param_size_unknown() -> None:
    assert _tier_from_param_size("unknown") == Tier.FREE_FAST


# ##################################################################
# available — online tests
@skip_if_no_ollama
@pytest.mark.asyncio
async def test_available_returns_true() -> None:
    provider = OllamaProvider()
    result = await provider.available()
    assert result is True


@pytest.mark.asyncio
async def test_available_wrong_port_returns_false() -> None:
    provider = OllamaProvider(base_url="http://localhost:19999")
    result = await provider.available()
    assert result is False


# ##################################################################
# list models — online tests
@skip_if_no_ollama
@pytest.mark.asyncio
async def test_list_models_returns_model_info_instances() -> None:
    provider = OllamaProvider()
    models = await provider.list_models()
    assert len(models) > 0
    for m in models:
        assert isinstance(m, ModelInfo)
        assert m.provider == "ollama"
        assert m.model_id != ""
        assert Capability.TEXT in m.capabilities
        assert Capability.STRUCTURED in m.capabilities
        assert isinstance(m.tier, Tier)


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_list_models_tiers_assigned() -> None:
    provider = OllamaProvider()
    models = await provider.list_models()
    tiers = {m.tier for m in models}
    # should have at least FREE_FAST since we have small models
    assert Tier.FREE_FAST in tiers or Tier.FREE_THINKING in tiers


@pytest.mark.asyncio
async def test_list_models_wrong_port_returns_empty() -> None:
    provider = OllamaProvider(base_url="http://localhost:19999")
    models = await provider.list_models()
    assert models == []


# ##################################################################
# complete — online tests
# timeouts are generous to allow for model loading from disk/VRAM.
# on slow machines the first request can take 30+ seconds.
@skip_if_no_ollama
@pytest.mark.asyncio
async def test_complete_simple_prompt() -> None:
    provider = OllamaProvider()
    messages = [Message(role="user", content="What is 2+2? Reply with just the number.")]
    model = _make_model()
    resp = await provider.complete(messages, model, timeout=180.0)
    assert isinstance(resp, Response)
    assert "4" in resp.text
    assert resp.model_used is model
    assert resp.conversation_id is not None
    assert resp.turn_id is not None


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_complete_returns_response_type() -> None:
    provider = OllamaProvider()
    messages = [Message(role="user", content="Say 'hello' and nothing else.")]
    model = _make_model()
    resp = await provider.complete(messages, model, timeout=180.0)
    assert isinstance(resp, Response)
    assert not isinstance(resp, StructuredResponse)
    assert resp.text.strip() != ""


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_complete_with_system_message() -> None:
    provider = OllamaProvider()
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
@skip_if_no_ollama
@pytest.mark.asyncio
async def test_complete_structured_output() -> None:
    class MathAnswer(BaseModel):
        result: int
        explanation: str

    provider = OllamaProvider()
    messages = [Message(role="user", content="What is 3 multiplied by 7?")]
    model = _make_model()
    resp = await provider.complete(messages, model, schema=MathAnswer, timeout=180.0)
    assert isinstance(resp, StructuredResponse)
    assert isinstance(resp.parsed, MathAnswer)
    assert resp.parsed.result == 21


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_complete_structured_with_system_message() -> None:
    class Sentiment(BaseModel):
        label: str
        confidence: float

    provider = OllamaProvider()
    messages = [
        Message(role="system", content="You are a sentiment analyser."),
        Message(role="user", content="Classify: 'I love this product!'"),
    ]
    model = _make_model()
    resp = await provider.complete(messages, model, schema=Sentiment, timeout=180.0)
    assert isinstance(resp, StructuredResponse)
    assert isinstance(resp.parsed, Sentiment)
    assert resp.parsed.label in {"positive", "negative", "neutral", "Positive", "Negative", "Neutral"}
    assert 0.0 <= resp.parsed.confidence <= 1.0


# ##################################################################
# stream — online tests
@skip_if_no_ollama
@pytest.mark.asyncio
async def test_stream_yields_string_chunks() -> None:
    provider = OllamaProvider()
    messages = [Message(role="user", content="Count from 1 to 5, one number per line.")]
    model = _make_model()
    chunks: list[str] = []
    async for chunk in provider.stream(messages, model, timeout=180.0):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    assert len(chunks) > 0
    full = "".join(chunks)
    assert full.strip() != ""


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_stream_produces_complete_response() -> None:
    provider = OllamaProvider()
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
    provider = OllamaProvider(base_url="http://localhost:19999")
    messages = [Message(role="user", content="hello")]
    model = _make_model()
    with pytest.raises(AgentError) as exc_info:
        await provider.complete(messages, model, timeout=5.0)
    assert exc_info.value.kind == ErrorKind.NOT_AVAILABLE


@pytest.mark.asyncio
async def test_stream_wrong_port_raises_agent_error() -> None:
    provider = OllamaProvider(base_url="http://localhost:19999")
    messages = [Message(role="user", content="hello")]
    model = _make_model()
    with pytest.raises(AgentError) as exc_info:
        async for _ in provider.stream(messages, model, timeout=5.0):
            pass
    assert exc_info.value.kind == ErrorKind.NOT_AVAILABLE
