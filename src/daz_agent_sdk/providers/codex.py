from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Type
from uuid import uuid4

from daz_agent_sdk.providers.base import Provider
from daz_agent_sdk.structured import extract_result, schema_filename, schema_instructions
from daz_agent_sdk.types import (
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
)


# ##################################################################
# codex models
# static list of codex model variants — uses ChatGPT auth via codex CLI
_CODEX_MODELS = [
    ModelInfo(
        provider="codex",
        model_id="gpt-5.3-codex",
        display_name="GPT-5.3 Codex",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.HIGH,
        supports_tools=True,
    ),
    ModelInfo(
        provider="codex",
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED, Capability.AGENTIC}),
        tier=Tier.MEDIUM,
        supports_tools=True,
    ),
]


# ##################################################################
# classify error
# map codex CLI error messages to agent-sdk ErrorKind for fallback decisions
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
# build prompt
# combine message history into a single prompt string for the codex CLI
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
# run codex subprocess
# pipe prompt to stdin, return (stdout, stderr, returncode)
async def _run_codex(prompt: str, model_id: str, timeout: float) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        "codex", "exec", "-", "--json", "-m", model_id, "-s", "read-only", "--ephemeral",
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
        raise AgentError(f"codex request timed out after {timeout}s", kind=ErrorKind.TIMEOUT)
    return stdout_bytes.decode(), stderr_bytes.decode(), proc.returncode or 0


# ##################################################################
# parse jsonl events
# extract agent_message text from codex JSONL event stream
def _parse_jsonl_response(stdout: str) -> tuple[str, dict[str, Any]]:
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    for line in stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type", "")
        if event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text:
                    text_parts.append(text)
            elif item.get("type") == "error":
                error_msg = item.get("message", "unknown error")
                raise AgentError(error_msg, kind=_classify_error(Exception(error_msg)))
        elif event_type == "turn.completed":
            usage = event.get("usage", {})
        elif event_type == "turn.failed":
            error = event.get("error", {})
            error_msg = error.get("message", "codex turn failed")
            raise AgentError(error_msg, kind=_classify_error(Exception(error_msg)))
        elif event_type == "error":
            error_msg = event.get("message", "codex error")
            raise AgentError(error_msg, kind=_classify_error(Exception(error_msg)))
    return "".join(text_parts), usage


# ##################################################################
# codex provider
# wraps the codex CLI for text completion, streaming, and structured
# output. uses ChatGPT auth managed by the codex CLI itself.
class CodexProvider(Provider):
    name = "codex"

    # ##################################################################
    # available
    # returns True if the codex CLI is on PATH
    async def available(self) -> bool:
        return shutil.which("codex") is not None

    # ##################################################################
    # list models
    # return the static codex model catalogue when the CLI is available
    async def list_models(self) -> list[ModelInfo]:
        if not await self.available():
            return []
        return list(_CODEX_MODELS)

    # ##################################################################
    # complete
    # send messages to codex CLI and return a full response
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
    ) -> Response | StructuredResponse:
        if shutil.which("codex") is None:
            raise AgentError("codex CLI not found", kind=ErrorKind.NOT_AVAILABLE)

        prompt = _build_prompt(messages)
        # For structured output, append file-writing instructions
        out_filename: str | None = None
        if schema is not None:
            out_filename = schema_filename()
            prompt += schema_instructions(schema, out_filename)

        conversation_id = uuid4()
        turn_id = uuid4()

        try:
            stdout, stderr, returncode = await _run_codex(prompt, model.model_id, timeout)
        except AgentError:
            raise
        except Exception as err:
            raise AgentError(str(err), kind=_classify_error(err)) from err

        if returncode != 0 and not stdout.strip():
            raise AgentError(f"codex exited with code {returncode}: {stderr}", kind=ErrorKind.INTERNAL)

        text, usage = _parse_jsonl_response(stdout)

        if schema is not None:
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix="codex-structured-")
            try:
                parsed_instance = extract_result(schema, out_filename or "", tmp_dir, text)
            except Exception as exc:
                raise AgentError(str(exc), kind=ErrorKind.INTERNAL) from exc
            finally:
                import shutil as _shutil
                _shutil.rmtree(tmp_dir, ignore_errors=True)
            return StructuredResponse(
                text=text,
                model_used=model,
                conversation_id=conversation_id,
                turn_id=turn_id,
                usage=usage,
                parsed=parsed_instance,
            )

        return Response(
            text=text,
            model_used=model,
            conversation_id=conversation_id,
            turn_id=turn_id,
            usage=usage,
        )

    # ##################################################################
    # stream
    # send messages to codex CLI and yield text chunks as JSONL events arrive
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 300.0,
    ) -> AsyncIterator[str]:
        if shutil.which("codex") is None:
            raise AgentError("codex CLI not found", kind=ErrorKind.NOT_AVAILABLE)

        prompt = _build_prompt(messages)
        proc = await asyncio.create_subprocess_exec(
            "codex", "exec", "-", "--json", "-m", model.model_id, "-s", "read-only", "--ephemeral",
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
                if event_type == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            yield text
                elif event_type == "turn.failed":
                    error = event.get("error", {})
                    error_msg = error.get("message", "codex turn failed")
                    raise AgentError(error_msg, kind=_classify_error(Exception(error_msg)))
                elif event_type == "error":
                    error_msg = event.get("message", "codex error")
                    raise AgentError(error_msg, kind=_classify_error(Exception(error_msg)))
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
