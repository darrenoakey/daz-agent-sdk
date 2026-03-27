from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Protocol

from daz_agent_sdk.config import Config
from daz_agent_sdk.types import AgentError, ErrorKind


# ##################################################################
# event logger protocol
# structural typing for any object that has a log_event method —
# allows ConversationLogger and test fakes to satisfy the same interface.
class EventLogger(Protocol):
    def log_event(self, event_type: str, **kwargs: Any) -> None: ...


# ##################################################################
# error message fragments used to classify provider exceptions
# ordered from most specific to least specific within each category
_RATE_LIMIT_FRAGMENTS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "too many requests",
    "capacity",
    "overloaded",
    "quota",
)

_AUTH_FRAGMENTS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "authentication",
    "api key",
    "api_key",
    "invalid key",
    "invalid_api_key",
    "permission denied",
)

_TIMEOUT_FRAGMENTS = (
    "timeout",
    "timed out",
    "deadline exceeded",
    "read timeout",
    "connect timeout",
)

_INVALID_REQUEST_FRAGMENTS = (
    "400",
    "invalid request",
    "invalid_request",
    "bad request",
    "validation error",
    "schema",
    "malformed",
)

_NOT_AVAILABLE_FRAGMENTS = (
    "connection refused",
    "not available",
    "not_available",
    "service unavailable",
    "503",
    "offline",
    "unreachable",
    "name or service not known",
    "cannot connect",
    "no route to host",
)


# ##################################################################
# classify error
# maps any exception to an ErrorKind for fallback decision making.
# checks the exception message and type name against known fragments.
# returns INTERNAL for anything that does not match a known category.
def classify_error(err: Exception) -> ErrorKind:
    message = str(err).lower()
    err_type = type(err).__name__.lower()
    combined = f"{err_type} {message}"

    if isinstance(err, (asyncio.TimeoutError, TimeoutError)):
        return ErrorKind.TIMEOUT

    if isinstance(err, ConnectionRefusedError):
        return ErrorKind.NOT_AVAILABLE

    for fragment in _NOT_AVAILABLE_FRAGMENTS:
        if fragment in combined:
            return ErrorKind.NOT_AVAILABLE

    for fragment in _RATE_LIMIT_FRAGMENTS:
        if fragment in combined:
            return ErrorKind.RATE_LIMIT

    for fragment in _AUTH_FRAGMENTS:
        if fragment in combined:
            return ErrorKind.AUTH

    for fragment in _TIMEOUT_FRAGMENTS:
        if fragment in combined:
            return ErrorKind.TIMEOUT

    for fragment in _INVALID_REQUEST_FRAGMENTS:
        if fragment in combined:
            return ErrorKind.INVALID_REQUEST

    return ErrorKind.INTERNAL


# ##################################################################
# execute with fallback
# runs execute_fn against each provider in providers_chain in order.
#
# single-shot mode (is_conversation=False):
#   cascades immediately on RATE_LIMIT, TIMEOUT, NOT_AVAILABLE, INTERNAL.
#   raises immediately on AUTH or INVALID_REQUEST.
#
# conversation mode (is_conversation=True):
#   applies exponential backoff (1s, 2s, 4s... up to max_backoff_seconds)
#   before cascading to the next provider.
#
# returns the first successful result, or raises AgentError with all
# attempt records if every provider in the chain fails.
async def execute_with_fallback(
    tier: str,
    providers_chain: list[str],
    execute_fn: Callable[[str], Coroutine[Any, Any, Any]],
    *,
    is_conversation: bool = False,
    config: Config | None = None,
    logger: EventLogger | None = None,
) -> Any:
    from daz_agent_sdk.config import load_config

    cfg = config or load_config()
    max_backoff = cfg.fallback.conversation.max_backoff_seconds
    ss_max_retries = cfg.fallback.single_shot.max_retries
    ss_retry_base = cfg.fallback.single_shot.retry_base_seconds

    attempts: list[dict[str, Any]] = []

    for index, provider_entry in enumerate(providers_chain):

        if logger is not None:
            logger.log_event("attempt_start", provider=provider_entry, tier=tier, attempt_index=index)

        # conversation mode: exponential backoff before each cascade
        if is_conversation and index > 0:
            delay = min(2 ** (index - 1), max_backoff)
            if logger is not None:
                logger.log_event("backoff", attempt=index, delay_ms=int(delay * 1000))
            await asyncio.sleep(delay)

        # determine retry count: single-shot retries per provider, conversation gets 1
        retries = ss_max_retries if not is_conversation else 1
        last_kind: ErrorKind = ErrorKind.INTERNAL

        for retry in range(retries):
            attempt: dict[str, Any] = {
                "provider": provider_entry,
                "tier": tier,
                "retry": retry,
            }

            # single-shot per-provider backoff on retry (not first attempt)
            if not is_conversation and retry > 0:
                delay = min(ss_retry_base * (2 ** (retry - 1)), 30.0)
                attempt["backoff_seconds"] = delay
                if logger is not None:
                    logger.log_event(
                        "retry_backoff",
                        provider=provider_entry,
                        retry=retry,
                        delay_ms=int(delay * 1000),
                    )
                await asyncio.sleep(delay)

            try:
                result = await execute_fn(provider_entry)
                if logger is not None:
                    logger.log_event(
                        "attempt_success",
                        provider=provider_entry,
                        tier=tier,
                        attempt_index=index,
                        retry=retry,
                    )
                attempt["success"] = True
                attempts.append(attempt)
                return result

            except Exception as err:
                kind = classify_error(err)
                attempt["error"] = str(err)
                attempt["kind"] = kind.value
                attempt["success"] = False
                attempts.append(attempt)
                last_kind = kind

                if logger is not None:
                    logger.log_event(
                        "attempt_failed",
                        provider=provider_entry,
                        tier=tier,
                        attempt_index=index,
                        retry=retry,
                        error=str(err),
                        kind=kind.value,
                    )

                # AUTH and INVALID_REQUEST are caller bugs — raise immediately
                if kind in (ErrorKind.AUTH, ErrorKind.INVALID_REQUEST):
                    if logger is not None:
                        logger.log_event(
                            "raise_immediate",
                            provider=provider_entry,
                            kind=kind.value,
                            error=str(err),
                        )
                    raise AgentError(
                        f"Non-retryable error from {provider_entry}: {err}",
                        kind=kind,
                        attempts=attempts,
                    ) from err

                # TIMEOUT and NOT_AVAILABLE: no point retrying same provider
                if kind in (ErrorKind.TIMEOUT, ErrorKind.NOT_AVAILABLE):
                    break

                # RATE_LIMIT and INTERNAL: worth retrying with backoff
                # (loop continues to next retry)

        # all retries for this provider exhausted — cascade to next
        if logger is not None and index < len(providers_chain) - 1:
            logger.log_event(
                "cascade",
                from_provider=provider_entry,
                to_provider=providers_chain[index + 1],
                reason=last_kind.value if last_kind else "unknown",
            )

    # all providers exhausted — build detailed error message
    if logger is not None:
        logger.log_event("all_failed", tier=tier, attempts=len(attempts))

    lines = [f"All providers in chain failed for tier '{tier}' after {len(attempts)} attempt(s):"]
    for a in attempts:
        status = "ok" if a.get("success") else a.get("kind", "unknown")
        retry_info = f" retry={a['retry']}" if a.get("retry", 0) > 0 else ""
        err_msg = a.get("error", "")
        # truncate long error messages to first line
        first_line = err_msg.split("\n")[0][:200] if err_msg else ""
        lines.append(f"  [{a['provider']}{retry_info}] {status}: {first_line}")

    raise AgentError(
        "\n".join(lines),
        kind=ErrorKind.NOT_AVAILABLE,
        attempts=attempts,
    )
