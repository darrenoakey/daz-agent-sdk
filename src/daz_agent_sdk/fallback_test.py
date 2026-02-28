from __future__ import annotations

import asyncio
import time

import pytest

from daz_agent_sdk.fallback import classify_error, execute_with_fallback
from daz_agent_sdk.types import AgentError, ErrorKind


# ##################################################################
# error classification tests


def test_classify_rate_limit_message():
    assert classify_error(Exception("rate limit exceeded")) == ErrorKind.RATE_LIMIT


def test_classify_rate_limit_429():
    assert classify_error(Exception("HTTP 429 Too Many Requests")) == ErrorKind.RATE_LIMIT


def test_classify_rate_limit_overloaded():
    assert classify_error(Exception("Provider is overloaded, please try again")) == ErrorKind.RATE_LIMIT


def test_classify_rate_limit_capacity():
    assert classify_error(Exception("No capacity available")) == ErrorKind.RATE_LIMIT


def test_classify_auth_401():
    assert classify_error(Exception("HTTP 401 Unauthorized")) == ErrorKind.AUTH


def test_classify_auth_403():
    assert classify_error(Exception("403 Forbidden")) == ErrorKind.AUTH


def test_classify_auth_api_key():
    assert classify_error(Exception("invalid api key provided")) == ErrorKind.AUTH


def test_classify_timeout_asyncio():
    assert classify_error(asyncio.TimeoutError()) == ErrorKind.TIMEOUT


def test_classify_timeout_builtin():
    assert classify_error(TimeoutError("read timeout")) == ErrorKind.TIMEOUT


def test_classify_timeout_message():
    assert classify_error(Exception("deadline exceeded for request")) == ErrorKind.TIMEOUT


def test_classify_invalid_request_400():
    assert classify_error(Exception("400 bad request")) == ErrorKind.INVALID_REQUEST


def test_classify_invalid_request_schema():
    assert classify_error(Exception("schema validation error in field x")) == ErrorKind.INVALID_REQUEST


def test_classify_not_available_connection_refused():
    assert classify_error(ConnectionRefusedError("connection refused")) == ErrorKind.NOT_AVAILABLE


def test_classify_not_available_service_unavailable():
    assert classify_error(Exception("503 service unavailable")) == ErrorKind.NOT_AVAILABLE


def test_classify_not_available_offline():
    assert classify_error(Exception("provider is offline")) == ErrorKind.NOT_AVAILABLE


def test_classify_internal_default():
    assert classify_error(Exception("something unexpected went wrong")) == ErrorKind.INTERNAL


def test_classify_internal_500():
    # "500" alone does not match any known fragment — falls through to INTERNAL
    assert classify_error(Exception("internal server error 500")) == ErrorKind.INTERNAL


# ##################################################################
# single-shot cascade tests


@pytest.mark.asyncio
async def test_single_shot_first_provider_succeeds():
    calls: list[str] = []

    async def execute(provider: str):
        calls.append(provider)
        return f"result from {provider}"

    result = await execute_with_fallback(
        "high",
        ["p1", "p2"],
        execute,
        is_conversation=False,
    )
    assert result == "result from p1"
    assert calls == ["p1"]


@pytest.mark.asyncio
async def test_single_shot_first_fails_second_succeeds():
    calls: list[str] = []

    async def execute(provider: str):
        calls.append(provider)
        if provider == "p1":
            raise Exception("rate limit exceeded")
        return f"result from {provider}"

    result = await execute_with_fallback(
        "high",
        ["p1", "p2"],
        execute,
        is_conversation=False,
    )
    assert result == "result from p2"
    assert calls == ["p1", "p2"]


@pytest.mark.asyncio
async def test_single_shot_all_fail_raises_agent_error():
    async def execute(provider: str):
        raise Exception("503 service unavailable")

    with pytest.raises(AgentError) as exc_info:
        await execute_with_fallback(
            "high",
            ["p1", "p2", "p3"],
            execute,
            is_conversation=False,
        )

    err = exc_info.value
    assert len(err.attempts) == 3
    assert all(not a["success"] for a in err.attempts)


@pytest.mark.asyncio
async def test_single_shot_all_fail_agent_error_has_attempts():
    errors = ["rate limit exceeded", "503 service unavailable", "provider offline"]

    async def execute(provider: str):
        idx = ["p1", "p2", "p3"].index(provider)
        raise Exception(errors[idx])

    with pytest.raises(AgentError) as exc_info:
        await execute_with_fallback(
            "medium",
            ["p1", "p2", "p3"],
            execute,
            is_conversation=False,
        )

    err = exc_info.value
    assert len(err.attempts) == 3
    providers_tried = [a["provider"] for a in err.attempts]
    assert providers_tried == ["p1", "p2", "p3"]


@pytest.mark.asyncio
async def test_single_shot_empty_chain_raises_agent_error():
    async def execute(provider: str):
        return "result"

    with pytest.raises(AgentError):
        await execute_with_fallback("high", [], execute, is_conversation=False)


@pytest.mark.asyncio
async def test_single_shot_timeout_cascades():
    calls: list[str] = []

    async def execute(provider: str):
        calls.append(provider)
        if provider == "p1":
            raise asyncio.TimeoutError()
        return "ok"

    result = await execute_with_fallback("high", ["p1", "p2"], execute, is_conversation=False)
    assert result == "ok"
    assert calls == ["p1", "p2"]


@pytest.mark.asyncio
async def test_single_shot_internal_error_cascades():
    calls: list[str] = []

    async def execute(provider: str):
        calls.append(provider)
        if provider == "p1":
            raise Exception("unexpected internal failure")
        return "ok"

    result = await execute_with_fallback("high", ["p1", "p2"], execute, is_conversation=False)
    assert result == "ok"
    assert calls == ["p1", "p2"]


# ##################################################################
# auth and invalid request raise immediately tests


@pytest.mark.asyncio
async def test_auth_error_raises_immediately():
    calls: list[str] = []

    async def execute(provider: str):
        calls.append(provider)
        raise Exception("401 Unauthorized — invalid api key")

    with pytest.raises(AgentError) as exc_info:
        await execute_with_fallback("high", ["p1", "p2", "p3"], execute, is_conversation=False)

    # should have stopped at p1 without trying p2 or p3
    assert calls == ["p1"]
    assert exc_info.value.kind == ErrorKind.AUTH


@pytest.mark.asyncio
async def test_invalid_request_raises_immediately():
    calls: list[str] = []

    async def execute(provider: str):
        calls.append(provider)
        raise Exception("400 bad request: schema validation failed")

    with pytest.raises(AgentError) as exc_info:
        await execute_with_fallback("high", ["p1", "p2", "p3"], execute, is_conversation=False)

    assert calls == ["p1"]
    assert exc_info.value.kind == ErrorKind.INVALID_REQUEST


# ##################################################################
# conversation backoff timing tests


@pytest.mark.asyncio
async def test_conversation_backoff_first_attempt_no_delay():
    call_times: list[float] = []

    async def execute(provider: str):
        call_times.append(time.monotonic())
        if provider == "p1":
            raise Exception("rate limit exceeded")
        return "ok"

    start = time.monotonic()
    await execute_with_fallback("medium", ["p1", "p2"], execute, is_conversation=True)
    elapsed = time.monotonic() - start

    # first attempt is immediate; second has 1s backoff (2^0=1)
    assert elapsed >= 0.9


@pytest.mark.asyncio
async def test_conversation_backoff_respects_max():
    from daz_agent_sdk.config import Config, FallbackConfig, FallbackConversationConfig, FallbackSingleShotConfig

    cfg = Config(
        fallback=FallbackConfig(
            single_shot=FallbackSingleShotConfig(),
            conversation=FallbackConversationConfig(max_backoff_seconds=2),
        )
    )

    delays: list[float] = []
    last_time: list[float] = [time.monotonic()]

    async def execute(provider: str):
        now = time.monotonic()
        if last_time:
            delays.append(now - last_time[0])
        last_time[0] = now
        if provider != "p4":
            raise Exception("rate limit exceeded")
        return "ok"

    await execute_with_fallback(
        "medium",
        ["p1", "p2", "p3", "p4"],
        execute,
        is_conversation=True,
        config=cfg,
    )

    # delays: p1→p2 = min(2^0=1, 2) = 1s; p2→p3 = min(2^1=2, 2) = 2s; p3→p4 = min(2^2=4, 2) = 2s
    # all delays should be <= 2s + small tolerance
    for d in delays[1:]:  # skip first entry (no prior provider)
        assert d <= 2.5


@pytest.mark.asyncio
async def test_conversation_no_backoff_for_single_shot():
    call_times: list[float] = []

    async def execute(provider: str):
        call_times.append(time.monotonic())
        if provider == "p1":
            raise Exception("rate limit exceeded")
        return "ok"

    start = time.monotonic()
    await execute_with_fallback("high", ["p1", "p2"], execute, is_conversation=False)
    elapsed = time.monotonic() - start

    # single-shot has NO backoff — should be well under 1 second
    assert elapsed < 0.5


# ##################################################################
# logging tests


@pytest.mark.asyncio
async def test_logging_records_attempt_and_success():
    events: list[dict] = []

    class _FakeLogger:
        def log_event(self, event_type: str, **kwargs):
            events.append({"event": event_type, **kwargs})

    async def execute(provider: str):
        return "ok"

    await execute_with_fallback(
        "high",
        ["p1"],
        execute,
        is_conversation=False,
        logger=_FakeLogger(),
    )

    event_types = [e["event"] for e in events]
    assert "attempt_start" in event_types
    assert "attempt_success" in event_types


@pytest.mark.asyncio
async def test_logging_records_cascade():
    events: list[dict] = []

    class _FakeLogger:
        def log_event(self, event_type: str, **kwargs):
            events.append({"event": event_type, **kwargs})

    async def execute(provider: str):
        if provider == "p1":
            raise Exception("rate limit exceeded")
        return "ok"

    await execute_with_fallback(
        "high",
        ["p1", "p2"],
        execute,
        is_conversation=False,
        logger=_FakeLogger(),
    )

    cascade_events = [e for e in events if e["event"] == "cascade"]
    assert len(cascade_events) == 1
    assert cascade_events[0]["from_provider"] == "p1"
    assert cascade_events[0]["to_provider"] == "p2"


@pytest.mark.asyncio
async def test_logging_records_all_failed():
    events: list[dict] = []

    class _FakeLogger:
        def log_event(self, event_type: str, **kwargs):
            events.append({"event": event_type, **kwargs})

    async def execute(provider: str):
        raise Exception("503 service unavailable")

    with pytest.raises(AgentError):
        await execute_with_fallback(
            "high",
            ["p1", "p2"],
            execute,
            is_conversation=False,
            logger=_FakeLogger(),
        )

    event_types = [e["event"] for e in events]
    assert "all_failed" in event_types
