from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Type
from uuid import uuid4

from agent_sdk.providers.base import Provider
from agent_sdk.types import (
    AgentError,
    Capability,
    ErrorKind,
    ImageResult,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    T,
    Tier,
    parse_json_from_llm,
)


# ##################################################################
# codex models
# static list of OpenAI GPT-4.1 model variants with capability and tier info
_CODEX_MODELS = [
    ModelInfo(
        provider="codex",
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.HIGH,
        supports_tools=True,
    ),
    ModelInfo(
        provider="codex",
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.MEDIUM,
        supports_tools=True,
    ),
    ModelInfo(
        provider="codex",
        model_id="gpt-4.1-nano",
        display_name="GPT-4.1 Nano",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.LOW,
        supports_tools=False,
    ),
]


# ##################################################################
# has sdk
# module-level flag set once on first import attempt
_HAS_SDK: bool | None = None


# ##################################################################
# import sdk
# lazy import openai — provider reports unavailable if not installed.
# returns the openai module or None.
def _import_sdk() -> Any:
    global _HAS_SDK
    try:
        import openai  # type: ignore[import]

        _HAS_SDK = True
        return openai
    except ImportError:
        _HAS_SDK = False
        return None


# ##################################################################
# classify openai error
# map openai SDK exceptions to agent-sdk ErrorKind for fallback decisions.
# inspects the error message and type name for key signal words.
def _classify_error(err: Exception) -> ErrorKind:
    msg = str(err).lower()
    type_name = type(err).__name__.lower()
    if "429" in msg or "rate_limit" in msg or "ratelimit" in type_name:
        return ErrorKind.RATE_LIMIT
    if "401" in msg or "403" in msg or "auth" in msg or "authentication" in type_name or "permission" in msg:
        return ErrorKind.AUTH
    if "timeout" in msg or "timed out" in msg or "timeout" in type_name:
        return ErrorKind.TIMEOUT
    if "400" in msg or "invalid" in msg or "badrequest" in type_name:
        return ErrorKind.INVALID_REQUEST
    return ErrorKind.INTERNAL


# ##################################################################
# build messages
# convert agent_sdk Message objects to the dict format OpenAI expects.
# passes role and content for each message as-is.
def _build_messages(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


# ##################################################################
# codex provider
# wraps the OpenAI async client for text completion, streaming, and
# structured output. uses OPENAI_API_KEY from the environment.
class CodexProvider(Provider):
    name = "codex"

    # ##################################################################
    # available
    # returns True if openai SDK is importable and OPENAI_API_KEY is set
    async def available(self) -> bool:
        sdk = _import_sdk()
        if sdk is None:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    # ##################################################################
    # list models
    # return the static codex model catalogue when the provider is available
    async def list_models(self) -> list[ModelInfo]:
        if not await self.available():
            return []
        return list(_CODEX_MODELS)

    # ##################################################################
    # complete
    # send messages to OpenAI and return a full response. when schema is
    # provided the response text is parsed and validated as pydantic.
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
    ) -> Response | StructuredResponse:
        sdk = _import_sdk()
        if sdk is None:
            raise AgentError("openai not installed", kind=ErrorKind.NOT_AVAILABLE)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AgentError("OPENAI_API_KEY not set", kind=ErrorKind.AUTH)

        msg_list = _build_messages(messages)

        if schema is not None:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            instruction = (
                f"\n\nRespond ONLY with valid JSON that matches this schema:\n{schema_json}\n"
                "Do not include any explanation or markdown. Output raw JSON only."
            )
            system_indices = [i for i, m in enumerate(msg_list) if m["role"] == "system"]
            if system_indices:
                last_sys = system_indices[-1]
                msg_list[last_sys] = {
                    "role": "system",
                    "content": msg_list[last_sys]["content"] + instruction,
                }
            else:
                msg_list.insert(0, {"role": "system", "content": instruction.strip()})

        conversation_id = uuid4()
        turn_id = uuid4()

        try:
            client = sdk.AsyncOpenAI(api_key=api_key, timeout=timeout)
            completion = await client.chat.completions.create(
                model=model.model_id,
                messages=msg_list,
            )
        except Exception as err:
            kind = _classify_error(err)
            raise AgentError(str(err), kind=kind) from err

        text = (completion.choices[0].message.content or "") if completion.choices else ""
        usage_data: dict[str, Any] = {}
        if completion.usage:
            usage_data = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        if schema is not None:
            try:
                parsed_json = parse_json_from_llm(text)
                parsed_instance = schema.model_validate(parsed_json)
            except Exception as exc:
                raise AgentError(
                    f"Failed to parse structured response: {exc}",
                    kind=ErrorKind.INVALID_REQUEST,
                ) from exc
            return StructuredResponse(
                text=text,
                model_used=model,
                conversation_id=conversation_id,
                turn_id=turn_id,
                usage=usage_data,
                parsed=parsed_instance,
            )

        return Response(
            text=text,
            model_used=model,
            conversation_id=conversation_id,
            turn_id=turn_id,
            usage=usage_data,
        )

    # ##################################################################
    # stream
    # send messages to OpenAI with streaming enabled and yield text chunks
    # as they arrive. each non-empty delta content string is yielded.
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        sdk = _import_sdk()
        if sdk is None:
            raise AgentError("openai not installed", kind=ErrorKind.NOT_AVAILABLE)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AgentError("OPENAI_API_KEY not set", kind=ErrorKind.AUTH)

        try:
            client = sdk.AsyncOpenAI(api_key=api_key, timeout=timeout)
            stream = await client.chat.completions.create(
                model=model.model_id,
                messages=_build_messages(messages),
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except AgentError:
            raise
        except Exception as err:
            kind = _classify_error(err)
            raise AgentError(str(err), kind=kind) from err

    # ##################################################################
    # generate image
    # codex does not support image generation
    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output: Path,
        **kwargs: Any,
    ) -> ImageResult:
        raise NotImplementedError("codex does not support image generation")
