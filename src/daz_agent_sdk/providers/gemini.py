from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Type
from uuid import uuid4

from daz_agent_sdk.providers.base import Provider
from daz_agent_sdk.types import (
    AgentError,
    Capability,
    ErrorKind,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    T,
    Tier,
    parse_json_from_llm,
)


# ##################################################################
# gemini models
# static catalogue of known Gemini models with their capabilities
_GEMINI_MODELS = [
    ModelInfo(
        provider="gemini",
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.HIGH,
        supports_streaming=True,
        supports_structured=True,
        supports_conversation=True,
        supports_tools=False,
    ),
    ModelInfo(
        provider="gemini",
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.MEDIUM,
        supports_streaming=True,
        supports_structured=True,
        supports_conversation=True,
        supports_tools=False,
    ),
    ModelInfo(
        provider="gemini",
        model_id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash Lite",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.LOW,
        supports_streaming=True,
        supports_structured=True,
        supports_conversation=True,
        supports_tools=False,
    ),
]


# ##################################################################
# import sdk
# lazy import to avoid hard dependency — provider reports unavailable
# if google-genai is not installed
def _import_sdk() -> Any:
    try:
        from google import genai  # type: ignore[attr-defined]
        return genai
    except ImportError:
        return None


# ##################################################################
# classify error
# map google-genai exceptions and HTTP status codes to our error kinds
# for fallback decision making
def _classify_error(err: Exception) -> ErrorKind:
    msg = str(err).lower()
    if "429" in msg or "quota" in msg or "rate" in msg or "resource_exhausted" in msg:
        return ErrorKind.RATE_LIMIT
    if "401" in msg or "403" in msg or "api_key" in msg or "permission" in msg or "unauthenticated" in msg:
        return ErrorKind.AUTH
    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        return ErrorKind.TIMEOUT
    if "400" in msg or "invalid" in msg or "bad request" in msg:
        return ErrorKind.INVALID_REQUEST
    return ErrorKind.INTERNAL


# ##################################################################
# build prompt
# combine message history into a single prompt string for the genai
# SDK, appending JSON schema instructions when structured output needed
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
        prompt += (
            "\n\nReturn ONLY valid JSON matching this schema, no other text:\n"
            f"```json\n{json.dumps(schema_json, indent=2)}\n```"
        )
    return prompt


# ##################################################################
# gemini provider
# wraps the google-genai SDK for text generation, streaming, and
# structured output. requires GEMINI_API_KEY in the environment.
class GeminiProvider(Provider):
    name = "gemini"

    # ##################################################################
    # init
    # store optional api key override; falls back to GEMINI_API_KEY env var
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    # ##################################################################
    # get client
    # build and return a configured genai Client instance
    def _get_client(self) -> Any:
        genai = _import_sdk()
        if genai is None:
            raise AgentError("google-genai not installed", kind=ErrorKind.NOT_AVAILABLE)
        api_key = self._api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise AgentError("GEMINI_API_KEY not set", kind=ErrorKind.AUTH)
        return genai.Client(api_key=api_key)

    # ##################################################################
    # available
    # check if google-genai is installed and GEMINI_API_KEY is configured
    async def available(self) -> bool:
        if _import_sdk() is None:
            return False
        api_key = self._api_key or os.environ.get("GEMINI_API_KEY")
        return bool(api_key)

    # ##################################################################
    # list models
    # return the known Gemini model catalogue
    async def list_models(self) -> list[ModelInfo]:
        if not await self.available():
            return []
        return list(_GEMINI_MODELS)

    # ##################################################################
    # complete
    # send messages to Gemini and collect the full response text.
    # when schema is provided, appends schema instructions and parses
    # the JSON response into the given pydantic model.
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
        import asyncio

        prompt = _build_prompt(messages, schema)
        conversation_id = uuid4()
        turn_id = uuid4()

        try:
            client = self._get_client()

            def _call() -> str:
                response = client.models.generate_content(
                    model=model.model_id,
                    contents=prompt,
                )
                return response.text or ""

            loop = asyncio.get_event_loop()
            response_text = await asyncio.wait_for(
                loop.run_in_executor(None, _call),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise AgentError(
                f"gemini request timed out after {timeout}s",
                kind=ErrorKind.TIMEOUT,
            )
        except AgentError:
            raise
        except Exception as err:
            kind = _classify_error(err)
            raise AgentError(str(err), kind=kind) from err

        if schema is not None:
            try:
                parsed_json = parse_json_from_llm(response_text)
                parsed_obj = schema.model_validate(parsed_json)
            except Exception as exc:
                raise AgentError(
                    f"Failed to parse structured response: {exc}",
                    kind=ErrorKind.INVALID_REQUEST,
                ) from exc
            return StructuredResponse(
                text=response_text,
                model_used=model,
                conversation_id=conversation_id,
                turn_id=turn_id,
                parsed=parsed_obj,
            )

        return Response(
            text=response_text,
            model_used=model,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )

    # ##################################################################
    # stream
    # send messages to Gemini and yield response chunks as they arrive
    # using the generate_content_stream API
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        import asyncio

        prompt = _build_prompt(messages, schema=None)

        try:
            client = self._get_client()

            def _iter_chunks() -> list[str]:
                chunks: list[str] = []
                for chunk in client.models.generate_content_stream(
                    model=model.model_id,
                    contents=prompt,
                ):
                    text = chunk.text or ""
                    if text:
                        chunks.append(text)
                return chunks

            loop = asyncio.get_event_loop()
            chunks = await asyncio.wait_for(
                loop.run_in_executor(None, _iter_chunks),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise AgentError(
                f"gemini stream timed out after {timeout}s",
                kind=ErrorKind.TIMEOUT,
            )
        except AgentError:
            raise
        except Exception as err:
            kind = _classify_error(err)
            raise AgentError(str(err), kind=kind) from err

        for chunk in chunks:
            yield chunk

    # ##################################################################
    # generate image
    # gemini does not support image generation via this provider
    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output: Path,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("gemini does not support image generation")
