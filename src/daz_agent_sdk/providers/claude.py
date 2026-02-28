from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

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
    parse_json_from_llm,
)

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
# import sdk
# lazy import to avoid hard dependency — provider reports unavailable
# if not installed
def _import_sdk() -> Any:
    try:
        import claude_agent_sdk
        return claude_agent_sdk
    except ImportError:
        return None


# ##################################################################
# extract text from messages
# pull text content from claude sdk AssistantMessage objects
def _extract_text(sdk: Any, message: Any) -> str:
    if not hasattr(sdk, "AssistantMessage") or not hasattr(sdk, "TextBlock"):
        text_attr = getattr(message, "text", None)
        if text_attr is not None:
            return str(text_attr)
        return ""
    if isinstance(message, sdk.AssistantMessage):
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, sdk.TextBlock):
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
        sdk = _import_sdk()
        if sdk is None:
            return False
        # check that the query function exists
        return hasattr(sdk, "query")

    # ##################################################################
    # list models
    # return the known claude model catalogue
    async def list_models(self) -> list[ModelInfo]:
        if not await self.available():
            return []
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
        timeout: float = 120.0,
        mcp_servers: dict[str, Any] | None = None,
    ) -> Response | StructuredResponse:
        sdk = _import_sdk()
        if sdk is None:
            raise AgentError("claude_agent_sdk not installed", kind=ErrorKind.NOT_AVAILABLE)

        prompt = _build_prompt(messages, schema)
        options = _build_options(sdk, model, tools, cwd, max_turns, self._permission_mode, mcp_servers)

        saved = _strip_claudecode()
        try:
            response_text = await asyncio.wait_for(
                _collect_response(sdk, prompt, options),
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

        if schema is not None:
            parsed_json = parse_json_from_llm(response_text)
            parsed_obj = schema.model_validate(parsed_json)
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

    # ##################################################################
    # stream
    # send messages to claude and yield response chunks as they arrive
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 120.0,
        mcp_servers: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        sdk = _import_sdk()
        if sdk is None:
            raise AgentError("claude_agent_sdk not installed", kind=ErrorKind.NOT_AVAILABLE)

        prompt = _build_prompt(messages, schema=None)
        options = _build_options(sdk, model, None, None, 1, self._permission_mode, mcp_servers)

        saved = _strip_claudecode()
        try:
            async for chunk in _stream_response(sdk, prompt, options):
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
def _build_prompt(messages: list[Message], schema: Type[T] | None) -> str:
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"[System]\n{msg.content}")
        elif msg.role == "user":
            parts.append(msg.content)
        elif msg.role == "assistant":
            parts.append(f"[Previous assistant response]\n{msg.content}")
    prompt = "\n\n".join(parts)
    if schema is not None:
        schema_json = schema.model_json_schema()
        import json
        prompt += (
            "\n\nReturn ONLY valid JSON matching this schema, no other text:\n"
            f"```json\n{json.dumps(schema_json, indent=2)}\n```"
        )
    return prompt


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
# iterate through the sdk query and collect all text blocks
async def _collect_response(sdk: Any, prompt: str, options: Any) -> str:
    parts: list[str] = []
    try:
        async for message in sdk.query(prompt=prompt, options=options):
            text = _extract_text(sdk, message)
            if text:
                parts.append(text)
    except Exception as err:
        err_str = str(err).lower()
        if "messageparse" in err_str or "unknown" in err_str:
            pass  # skip unknown message types (rate_limit_event etc)
        else:
            raise
    return "".join(parts).strip()


# ##################################################################
# stream response
# iterate through the sdk query and yield text chunks as they arrive
async def _stream_response(sdk: Any, prompt: str, options: Any) -> AsyncIterator[str]:
    try:
        async for message in sdk.query(prompt=prompt, options=options):
            text = _extract_text(sdk, message)
            if text:
                yield text
    except Exception as err:
        err_str = str(err).lower()
        if "messageparse" in err_str or "unknown" in err_str:
            pass
        else:
            raise


# ##################################################################
# placeholder uuid
# generate a uuid — will be replaced by conversation-level uuid
# management in the conversation module
def _placeholder_uuid() -> Any:
    from uuid import uuid4
    return uuid4()
