from __future__ import annotations

import asyncio
import json
import shutil
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
# classify error
# map gemini CLI error messages to our error kinds for fallback decisions
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
# combine message history into a single prompt string for the gemini CLI
def _build_prompt(messages: list[Message], schema: Type[T] | None = None) -> str:
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
# run gemini subprocess
# pipe prompt via stdin, return (stdout, stderr, returncode)
async def _run_gemini(prompt: str, model_id: str, output_format: str, timeout: float) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        "gemini", "-p", "", "-m", model_id, "-o", output_format,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(prompt.encode()),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise AgentError(f"gemini request timed out after {timeout}s", kind=ErrorKind.TIMEOUT)
    return stdout_bytes.decode(), stderr_bytes.decode(), proc.returncode or 0


# ##################################################################
# gemini provider
# wraps the gemini CLI for text generation, streaming, and structured
# output. auth is handled by the gemini CLI itself (Google account).
class GeminiProvider(Provider):
    name = "gemini"

    # ##################################################################
    # available
    # check if gemini CLI is on PATH
    async def available(self) -> bool:
        return shutil.which("gemini") is not None

    # ##################################################################
    # list models
    # return the known Gemini model catalogue
    async def list_models(self) -> list[ModelInfo]:
        if not await self.available():
            return []
        return list(_GEMINI_MODELS)

    # ##################################################################
    # complete
    # send messages to gemini CLI and collect the full response text
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
        if shutil.which("gemini") is None:
            raise AgentError("gemini CLI not found", kind=ErrorKind.NOT_AVAILABLE)

        prompt = _build_prompt(messages, schema)
        conversation_id = uuid4()
        turn_id = uuid4()

        try:
            stdout, stderr, returncode = await _run_gemini(prompt, model.model_id, "json", timeout)
        except AgentError:
            raise
        except Exception as err:
            raise AgentError(str(err), kind=_classify_error(err)) from err

        if returncode != 0 and not stdout.strip():
            raise AgentError(f"gemini exited with code {returncode}: {stderr}", kind=ErrorKind.INTERNAL)

        # parse JSON response: {"response": "text", "stats": {...}}
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise AgentError(f"Failed to parse gemini JSON output: {exc}", kind=ErrorKind.INTERNAL) from exc

        response_text = data.get("response", "")
        usage: dict[str, Any] = {}
        stats = data.get("stats", {})
        if stats:
            models_stats = stats.get("models", {})
            for model_stats in models_stats.values():
                tokens = model_stats.get("tokens", {})
                if tokens:
                    usage = {
                        "input_tokens": tokens.get("input", 0),
                        "output_tokens": tokens.get("candidates", 0),
                        "total_tokens": tokens.get("total", 0),
                    }
                    break

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
                usage=usage,
                parsed=parsed_obj,
            )

        return Response(
            text=response_text,
            model_used=model,
            conversation_id=conversation_id,
            turn_id=turn_id,
            usage=usage,
        )

    # ##################################################################
    # stream
    # send messages to gemini CLI with stream-json format and yield chunks
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        if shutil.which("gemini") is None:
            raise AgentError("gemini CLI not found", kind=ErrorKind.NOT_AVAILABLE)

        prompt = _build_prompt(messages, schema=None)
        proc = await asyncio.create_subprocess_exec(
            "gemini", "-p", "", "-m", model.model_id, "-o", "stream-json",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(prompt.encode())
        proc.stdin.close()

        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type", "")
                if event_type == "message" and event.get("role") == "assistant":
                    content = event.get("content", "")
                    if content:
                        yield content
                elif event_type == "result":
                    status = event.get("status", "")
                    if status != "success":
                        raise AgentError(f"gemini stream failed: {status}", kind=ErrorKind.INTERNAL)
        except AgentError:
            raise
        except Exception as err:
            raise AgentError(str(err), kind=_classify_error(err)) from err
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

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
