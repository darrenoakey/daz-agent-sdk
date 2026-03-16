from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

import claude_agent_sdk as _sdk

from daz_agent_sdk.structured import ensure_cwd, extract_result, schema_filename, schema_instructions
from daz_agent_sdk.types import (
    AgentError,
    Capability,
    ErrorKind,
    ImageResult,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    Tier,
)

_logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_CLAUDE_MODELS = [
    ModelInfo(
        provider="claude",
        model_id="claude-opus-4-6",
        display_name="Claude Opus 4.6",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.HIGH,
        supports_tools=True,
    ),
    ModelInfo(
        provider="claude",
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.MEDIUM,
        supports_tools=True,
    ),
    ModelInfo(
        provider="claude",
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.LOW,
        supports_tools=True,
    ),
]


# ##################################################################
# strip claudecode env
# the claude agent sdk cannot be invoked from within a claude code
# session without first removing the CLAUDECODE env var
def _strip_claudecode() -> str | None:
    return os.environ.pop("CLAUDECODE", None)


# ##################################################################
# restore claudecode env
# put the CLAUDECODE var back after sdk call completes
def _restore_claudecode(saved: str | None) -> None:
    if saved is not None:
        os.environ["CLAUDECODE"] = saved


# ##################################################################
# classify claude error
# map claude sdk exceptions to our error kinds for fallback decisions
def _classify_error(err: Exception) -> ErrorKind:
    msg = str(err).lower()
    if "rate_limit" in msg or "429" in msg or "overloaded" in msg or "capacity" in msg:
        return ErrorKind.RATE_LIMIT
    if "401" in msg or "403" in msg or "auth" in msg or "permission" in msg:
        return ErrorKind.AUTH
    if "timeout" in msg or "timed out" in msg:
        return ErrorKind.TIMEOUT
    if "400" in msg or "invalid" in msg:
        return ErrorKind.INVALID_REQUEST
    return ErrorKind.INTERNAL



# ##################################################################
# extract text from messages
# pull text content from claude sdk AssistantMessage objects
def _extract_text(message: Any) -> str:
    # ResultMessage carries the final text after agentic tool use
    if isinstance(message, _sdk.ResultMessage):
        result = getattr(message, "result", None)
        if result:
            return str(result)
        return ""
    if isinstance(message, _sdk.AssistantMessage):
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, _sdk.TextBlock):
                parts.append(block.text)
        return "".join(parts)
    return ""


# ##################################################################
# claude provider
# wraps the claude agent sdk for text generation, streaming, and
# structured output with automatic CLAUDECODE env handling
class ClaudeProvider:
    name = "claude"

    # ##################################################################
    # init
    # store permission mode and any extra options
    def __init__(self, permission_mode: str = "bypassPermissions") -> None:
        self._permission_mode = permission_mode

    # ##################################################################
    # available
    # check if claude agent sdk is installed and importable
    async def available(self) -> bool:
        return hasattr(_sdk, "query")

    # ##################################################################
    # list models
    # return the known claude model catalogue
    async def list_models(self) -> list[ModelInfo]:
        return list(_CLAUDE_MODELS)

    # ##################################################################
    # complete
    # send messages to claude and collect the full response
    async def complete(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        schema: Type[T] | None = None,
        tools: list[str] | None = None,
        cwd: str | Path | None = None,
        max_turns: int = 1,
        timeout: float = 300.0,
        mcp_servers: dict[str, Any] | None = None,
    ) -> Response | StructuredResponse:
        # For structured output, use file-based extraction
        out_filename: str | None = None
        effective_cwd = cwd
        created_temp_cwd = False
        if schema is not None:
            out_filename = schema_filename()
            effective_cwd, created_temp_cwd = ensure_cwd(cwd)

        prompt = _build_prompt(messages)
        if schema is not None and out_filename is not None:
            prompt += schema_instructions(schema, out_filename)

        options = _build_options(_sdk, model, tools, effective_cwd, max_turns, self._permission_mode, mcp_servers)

        saved = _strip_claudecode()
        try:
            response_text = await asyncio.wait_for(
                _collect_response(prompt, options),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise AgentError(
                f"claude request timed out after {timeout}s",
                kind=ErrorKind.TIMEOUT,
            )
        except Exception as err:
            kind = _classify_error(err)
            raise AgentError(str(err), kind=kind) from err
        finally:
            _restore_claudecode(saved)

        try:
            if schema is not None and out_filename is not None and effective_cwd is not None:
                try:
                    parsed_obj = extract_result(schema, out_filename, str(effective_cwd), response_text)
                except Exception as e:
                    raise AgentError(str(e), kind=ErrorKind.INTERNAL) from e
                return StructuredResponse(
                    text=response_text,
                    model_used=model,
                    conversation_id=_placeholder_uuid(),
                    turn_id=_placeholder_uuid(),
                    parsed=parsed_obj,
                )

            return Response(
                text=response_text,
                model_used=model,
                conversation_id=_placeholder_uuid(),
                turn_id=_placeholder_uuid(),
            )
        finally:
            if created_temp_cwd and effective_cwd:
                import shutil
                shutil.rmtree(effective_cwd, ignore_errors=True)

    # ##################################################################
    # stream
    # send messages to claude and yield response chunks as they arrive
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 300.0,
        mcp_servers: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        prompt = _build_prompt(messages)
        options = _build_options(_sdk, model, None, None, 1, self._permission_mode, mcp_servers)

        saved = _strip_claudecode()
        try:
            async for chunk in _stream_response(prompt, options):
                yield chunk
        except Exception as err:
            kind = _classify_error(err)
            raise AgentError(str(err), kind=kind) from err
        finally:
            _restore_claudecode(saved)

    # ##################################################################
    # generate image
    # claude does not support image generation
    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output: Path,
        **kwargs: Any,
    ) -> ImageResult:
        raise NotImplementedError("claude does not support image generation")


# ##################################################################
# build prompt
# combine message history into a single prompt string for the sdk,
# appending json schema instructions when structured output needed
def _build_prompt(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"[System]\n{msg.content}")
        elif msg.role == "user":
            parts.append(msg.content)
        elif msg.role == "assistant":
            parts.append(f"[Previous assistant response]\n{msg.content}")
    return "\n\n".join(parts)


# ##################################################################
# build options
# create ClaudeAgentOptions with the right settings
def _build_options(
    sdk: Any,
    model: ModelInfo,
    tools: list[str] | None,
    cwd: str | Path | None,
    max_turns: int,
    permission_mode: str,
    mcp_servers: dict[str, Any] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "permission_mode": permission_mode,
        "allowed_tools": tools or [],
        "max_turns": max_turns,
    }
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if model.model_id != "claude-opus-4-6":
        kwargs["model"] = model.model_id
    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers
    return sdk.ClaudeAgentOptions(**kwargs)


# ##################################################################
# collect response
# iterate through the sdk query and collect all text blocks.
# monkey-patches parse_message to skip unknown message types
# (e.g. rate_limit_event) instead of raising MessageParseError
# which kills the async generator and loses all collected text.
async def _collect_response(prompt: str, options: Any) -> str:
    import importlib
    _client = importlib.import_module("claude_agent_sdk._internal.client")

    class _SkipMessage:
        """Sentinel returned for unparseable messages."""

    _orig_parse = getattr(_client, "parse_message")

    def _safe_parse(data: Any) -> Any:
        try:
            return _orig_parse(data)
        except Exception:
            return _SkipMessage()

    setattr(_client, "parse_message", _safe_parse)
    parts: list[str] = []
    result_text: str | None = None
    msg_count = 0
    skip_count = 0
    msg_types: list[str] = []
    try:
        async for message in _sdk.query(prompt=prompt, options=options):
            msg_count += 1
            if isinstance(message, _SkipMessage):
                skip_count += 1
                continue
            msg_types.append(type(message).__name__)
            # ResultMessage is the definitive final answer in agentic mode
            if isinstance(message, _sdk.ResultMessage):
                r = getattr(message, "result", None)
                if r:
                    result_text = str(r)
                continue
            text = _extract_text(message)
            if text:
                parts.append(text)
    except Exception as err:
        err_str = str(err).lower()
        if "messageparse" in err_str or "unknown" in err_str:
            _logger.warning("claude: swallowed parse error after %d msgs (%d skipped): %s", msg_count, skip_count, err)
        else:
            raise
    finally:
        setattr(_client, "parse_message", _orig_parse)
    # prefer ResultMessage.result (final answer) over intermediate text
    if result_text is not None:
        return result_text.strip()
    response = "".join(parts).strip()
    if not response:
        _logger.warning(
            "claude: empty response — %d messages received, %d skipped, types: %s, parts: %d",
            msg_count, skip_count, msg_types, len(parts),
        )
    return response


# ##################################################################
# stream response
# iterate through the sdk query and yield text chunks as they arrive
async def _stream_response(prompt: str, options: Any) -> AsyncIterator[str]:
    import importlib
    _client = importlib.import_module("claude_agent_sdk._internal.client")

    class _SkipMessage:
        pass

    _orig_parse = getattr(_client, "parse_message")

    def _safe_parse(data: Any) -> Any:
        try:
            return _orig_parse(data)
        except Exception:
            return _SkipMessage()

    setattr(_client, "parse_message", _safe_parse)
    try:
        async for message in _sdk.query(prompt=prompt, options=options):
            if isinstance(message, _SkipMessage):
                continue
            text = _extract_text(message)
            if text:
                yield text
    except Exception as err:
        err_str = str(err).lower()
        if "messageparse" in err_str or "unknown" in err_str:
            pass
        else:
            raise
    finally:
        setattr(_client, "parse_message", _orig_parse)


# ##################################################################
# placeholder uuid
# generate a uuid — will be replaced by conversation-level uuid
# management in the conversation module
def _placeholder_uuid() -> Any:
    from uuid import uuid4
    return uuid4()
