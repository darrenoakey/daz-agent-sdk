from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator, Type
from uuid import uuid4

import pytest
from pydantic import BaseModel

from daz_agent_sdk.conversation import Conversation
from daz_agent_sdk.providers.base import Provider
from daz_agent_sdk.registry import _provider_cache, _model_cache
from daz_agent_sdk.types import (
    Capability,
    ImageResult,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    T,
    Tier,
)


# ##################################################################
# in-process provider
# a real Provider implementation that returns canned data without
# any network calls. used as a concrete plugin for all conversation tests.
class InProcessProvider(Provider):
    name = "inprocess"

    # ##################################################################
    # init
    # configure the provider with a fixed response string and optional
    # per-call response sequence for multi-turn tests
    def __init__(self, response: str = "ok", responses: list[str] | None = None) -> None:
        self._response = response
        self._responses = responses or []
        self._call_count = 0

    # ##################################################################
    # available
    # always reports as available — no external dependency
    async def available(self) -> bool:
        return True

    # ##################################################################
    # list models
    # returns a single ModelInfo representing this test provider
    async def list_models(self) -> list[ModelInfo]:
        return [_inprocess_model()]

    # ##################################################################
    # complete
    # returns a Response or StructuredResponse with the configured text.
    # if schema is provided, attempts to parse the response text as json.
    async def complete(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        schema: Type[T] | None = None,
        tools=None,
        cwd=None,
        max_turns: int = 1,
        timeout: float = 120.0,
    ) -> Response | StructuredResponse:
        text = self._next_response()
        conv_id = uuid4()
        turn_id = uuid4()
        if schema is not None:
            try:
                data = json.loads(text)
                parsed = schema(**data)
            except Exception:
                parsed = None
            return StructuredResponse(
                text=text,
                model_used=model,
                conversation_id=conv_id,
                turn_id=turn_id,
                parsed=parsed,
            )
        return Response(
            text=text,
            model_used=model,
            conversation_id=conv_id,
            turn_id=turn_id,
        )

    # ##################################################################
    # stream
    # yields the response as individual word chunks to simulate streaming
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        text = self._next_response()
        for word in text.split():
            yield word + " "

    # ##################################################################
    # generate image
    # not supported — raises NotImplementedError
    async def generate_image(self, prompt: str, *, width: int, height: int, output: Path, **kwargs) -> ImageResult:
        raise NotImplementedError("InProcessProvider does not support image generation")

    # ##################################################################
    # next response
    # returns the next canned response string, cycling through the list
    def _next_response(self) -> str:
        if self._responses:
            idx = min(self._call_count, len(self._responses) - 1)
            text = self._responses[idx]
        else:
            text = self._response
        self._call_count += 1
        return text


# ##################################################################
# inprocess model
# factory for the ModelInfo used by InProcessProvider
def _inprocess_model() -> ModelInfo:
    return ModelInfo(
        provider="inprocess",
        model_id="v1",
        display_name="InProcess/v1",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.HIGH,
        supports_streaming=True,
        supports_structured=True,
        supports_conversation=True,
    )


# ##################################################################
# conversation fixture helper
# builds a Conversation wired to the InProcessProvider without touching
# any external service. registers the provider and model in the registry.
def _make_conversation(
    name: str | None = "test",
    response: str = "hello",
    responses: list[str] | None = None,
    tier: Tier = Tier.HIGH,
    system: str | None = None,
) -> Conversation:
    provider = InProcessProvider(response=response, responses=responses)
    _provider_cache["inprocess"] = provider
    _model_cache["inprocess:v1"] = _inprocess_model()
    return Conversation(
        name=name,
        tier=tier,
        system=system,
        provider="inprocess",
        model="v1",
    )


# ##################################################################
# cleanup fixture
# removes the inprocess provider from caches after each test
@pytest.fixture(autouse=True)
def cleanup_registry():
    yield
    _provider_cache.pop("inprocess", None)
    _model_cache.pop("inprocess:v1", None)


# ##################################################################
# test init sets name
# verifies that the name passed to __init__ is stored correctly
def test_init_sets_name():
    conv = Conversation(name="my-chat")
    assert conv.name == "my-chat"


# ##################################################################
# test init default tier
# verifies that Tier.HIGH is the default when no tier is specified
def test_init_default_tier():
    conv = Conversation()
    assert conv.tier == Tier.HIGH


# ##################################################################
# test context manager
# verifies __aenter__ and __aexit__ work without raising
@pytest.mark.asyncio
async def test_context_manager():
    conv = _make_conversation()
    async with conv as chat:
        assert chat is conv


# ##################################################################
# test say adds to history
# verifies that both the user message and the assistant reply are
# appended to history after a say() call
@pytest.mark.asyncio
async def test_say_adds_to_history():
    conv = _make_conversation(response="world")
    async with conv as chat:
        await chat.say("hello")
    hist = conv.history
    assert len(hist) == 2
    assert hist[0].role == "user"
    assert hist[0].content == "hello"
    assert hist[1].role == "assistant"
    assert hist[1].content == "world"


# ##################################################################
# test say returns response
# verifies that say() returns a Response object
@pytest.mark.asyncio
async def test_say_returns_response():
    conv = _make_conversation(response="pong")
    async with conv as chat:
        result = await chat.say("ping")
    assert isinstance(result, Response)
    assert result.text == "pong"


# ##################################################################
# schema for structured response test
class _BookOutline(BaseModel):
    title: str
    chapters: int


# ##################################################################
# test say with schema
# verifies that say() with schema returns a StructuredResponse with
# a parsed pydantic instance in .parsed
@pytest.mark.asyncio
async def test_say_with_schema():
    json_response = '{"title": "My Book", "chapters": 3}'
    conv = _make_conversation(response=json_response)
    async with conv as chat:
        result = await chat.say("outline please", schema=_BookOutline)
    assert isinstance(result, StructuredResponse)
    assert isinstance(result.parsed, _BookOutline)
    assert result.parsed.title == "My Book"
    assert result.parsed.chapters == 3


# ##################################################################
# test stream adds to history
# verifies that stream() appends user message and full assistant text
@pytest.mark.asyncio
async def test_stream_adds_to_history():
    conv = _make_conversation(response="hello world")
    async with conv as chat:
        chunks = []
        async for chunk in chat.stream("go"):
            chunks.append(chunk)
    hist = conv.history
    assert len(hist) == 2
    assert hist[0].role == "user"
    assert hist[1].role == "assistant"
    # reassembled text should match original words
    assert "hello" in hist[1].content
    assert "world" in hist[1].content


# ##################################################################
# test stream yields chunks
# verifies that stream() actually yields individual text chunks
@pytest.mark.asyncio
async def test_stream_yields_chunks():
    conv = _make_conversation(response="one two three")
    async with conv as chat:
        chunks = []
        async for chunk in chat.stream("go"):
            chunks.append(chunk)
    # the InProcessProvider splits on whitespace — expect 3 chunks
    assert len(chunks) == 3


# ##################################################################
# test history property
# verifies the history property returns a list of Message objects
@pytest.mark.asyncio
async def test_history_property():
    conv = _make_conversation(response="response1")
    async with conv as chat:
        await chat.say("question1")
    hist = conv.history
    assert isinstance(hist, list)
    assert all(isinstance(m, Message) for m in hist)


# ##################################################################
# test system prompt
# verifies that a system message is prepended as the first message
@pytest.mark.asyncio
async def test_system_prompt():
    conv = _make_conversation(response="hi", system="You are a pirate")
    async with conv as chat:
        await chat.say("hello")
    hist = conv.history
    assert hist[0].role == "system"
    assert hist[0].content == "You are a pirate"
    assert hist[1].role == "user"
    assert hist[2].role == "assistant"


# ##################################################################
# test fork copies history
# verifies that the forked conversation starts with the same history
@pytest.mark.asyncio
async def test_fork_copies_history():
    conv = _make_conversation(responses=["first", "second"])
    async with conv as chat:
        await chat.say("turn 1")
    forked = conv.fork(name="fork-1")
    assert len(forked.history) == len(conv.history)
    for orig, fork_msg in zip(conv.history, forked.history):
        assert orig.role == fork_msg.role
        assert orig.content == fork_msg.content


# ##################################################################
# test fork independent
# verifies that changes to the forked conversation do not affect original
@pytest.mark.asyncio
async def test_fork_independent():
    conv = _make_conversation(responses=["first", "second", "third"])
    async with conv as chat:
        await chat.say("turn 1")
    original_len = len(conv.history)
    forked = conv.fork()
    # add new exchange directly to forked history to simulate a turn
    forked._history.append(Message(role="user", content="fork question"))
    assert len(conv.history) == original_len
    assert len(forked.history) == original_len + 1


# ##################################################################
# test tier override in say
# verifies that passing a tier to say() uses that tier rather than default
@pytest.mark.asyncio
async def test_tier_override_in_say():
    # we test that the call does not raise when the tier override is used.
    # InProcessProvider ignores tier at execution time, but we confirm
    # the method accepts and passes the override without error.
    conv = _make_conversation(response="ok", tier=Tier.HIGH)
    async with conv as chat:
        result = await chat.say("hello", tier=Tier.LOW)
    assert result.text == "ok"


# ##################################################################
# test multiple turns
# verifies that a multi-turn conversation builds correct sequential history
@pytest.mark.asyncio
async def test_multiple_turns():
    conv = _make_conversation(responses=["answer1", "answer2", "answer3"])
    async with conv as chat:
        await chat.say("question1")
        await chat.say("question2")
        await chat.say("question3")
    hist = conv.history
    # 3 user + 3 assistant = 6 messages total
    assert len(hist) == 6
    assert hist[0].role == "user"
    assert hist[0].content == "question1"
    assert hist[1].role == "assistant"
    assert hist[1].content == "answer1"
    assert hist[2].role == "user"
    assert hist[2].content == "question2"
    assert hist[3].role == "assistant"
    assert hist[3].content == "answer2"
    assert hist[4].role == "user"
    assert hist[4].content == "question3"
    assert hist[5].role == "assistant"
    assert hist[5].content == "answer3"
