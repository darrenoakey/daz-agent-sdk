from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator, Type
from uuid import uuid4

import aiohttp

from agent_sdk.providers.base import Provider
from agent_sdk.types import (
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
# parameter count to tier
# classify a model into a tier based on its parameter count string.
# models with >20B params get FREE_THINKING, everything else FREE_FAST.
# returns FREE_FAST when parameter size is unknown or unparseable.
def _tier_from_param_size(param_size: str) -> Tier:
    if not param_size:
        return Tier.FREE_FAST
    try:
        size_str = param_size.upper().replace(" ", "")
        if size_str.endswith("B"):
            count = float(size_str[:-1])
        elif size_str.endswith("M"):
            count = float(size_str[:-1]) / 1000.0
        else:
            return Tier.FREE_FAST
        return Tier.FREE_THINKING if count > 20.0 else Tier.FREE_FAST
    except (ValueError, AttributeError):
        return Tier.FREE_FAST


# ##################################################################
# classify aiohttp error
# map aiohttp exceptions to the agent-sdk ErrorKind taxonomy.
# connection errors become NOT_AVAILABLE, timeouts become TIMEOUT.
def _classify_aiohttp_error(exc: Exception) -> ErrorKind:
    if isinstance(exc, aiohttp.ServerConnectionError | aiohttp.ClientConnectorError):
        return ErrorKind.NOT_AVAILABLE
    if isinstance(exc, TimeoutError | aiohttp.ServerTimeoutError):
        return ErrorKind.TIMEOUT
    return ErrorKind.INTERNAL


# ##################################################################
# ollama provider
# HTTP provider for a locally-running Ollama instance. communicates
# with the Ollama REST API at base_url. no authentication required.
# a new aiohttp session is created per call — sessions are not safe
# to share across event loops or concurrent long-running tasks.
class OllamaProvider(Provider):
    name = "ollama"

    # ##################################################################
    # init
    # capture the base URL of the Ollama server. defaults to the
    # standard local Ollama address.
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    # ##################################################################
    # available
    # probe the Ollama server by fetching the model list. returns True
    # if the server responds with HTTP 200, False otherwise.
    async def available(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    # ##################################################################
    # list models
    # fetch the model list from Ollama and convert each entry into a
    # ModelInfo. all ollama models support TEXT and STRUCTURED capabilities.
    # tier is inferred from parameter count when available.
    async def list_models(self) -> list[ModelInfo]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
        except Exception:
            return []

        models: list[ModelInfo] = []
        for entry in data.get("models", []):
            model_id = entry.get("name", "")
            if not model_id:
                continue
            details = entry.get("details", {})
            param_size = (details.get("parameter_size") or "").strip()
            tier = _tier_from_param_size(param_size)
            display = model_id.split(":")[0].replace("-", " ").replace("_", " ").title()
            if param_size:
                display = f"{display} ({param_size})"
            models.append(
                ModelInfo(
                    provider="ollama",
                    model_id=model_id,
                    display_name=display,
                    capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
                    tier=tier,
                    supports_streaming=True,
                    supports_structured=True,
                    supports_conversation=True,
                    supports_tools=False,
                )
            )
        return models

    # ##################################################################
    # build messages payload
    # convert agent Message objects to the format Ollama expects.
    # system messages are included as-is; others use role+content form.
    def _build_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    # ##################################################################
    # complete
    # POST to /api/chat with stream:false. for structured output, the
    # JSON schema is appended as an instruction to the last system message
    # (or a new system message is added). the response text is parsed
    # and validated against the schema when one is provided.
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
        conversation_id = uuid4()
        turn_id = uuid4()

        msg_list = self._build_messages(messages)

        if schema is not None:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            instruction = (
                f"\n\nRespond ONLY with valid JSON that matches this schema:\n{schema_json}\n"
                "Do not include any explanation or markdown. Output raw JSON only."
            )
            # append schema instruction to last system message or add new one
            system_indices = [i for i, m in enumerate(msg_list) if m["role"] == "system"]
            if system_indices:
                last_sys = system_indices[-1]
                msg_list[last_sys] = {
                    "role": "system",
                    "content": msg_list[last_sys]["content"] + instruction,
                }
            else:
                msg_list.insert(0, {"role": "system", "content": instruction.strip()})

        payload = {
            "model": model.model_id,
            "messages": msg_list,
            "stream": False,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 429:
                        raise AgentError(
                            f"Ollama rate limit: {resp.status}",
                            kind=ErrorKind.RATE_LIMIT,
                        )
                    if resp.status >= 400:
                        body = await resp.text()
                        raise AgentError(
                            f"Ollama error {resp.status}: {body}",
                            kind=ErrorKind.INTERNAL,
                        )
                    data = await resp.json()
        except AgentError:
            raise
        except Exception as exc:
            kind = _classify_aiohttp_error(exc)
            raise AgentError(str(exc), kind=kind) from exc

        text = (data.get("message") or {}).get("content") or ""
        usage_data = data.get("usage", {})

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
    # POST to /api/chat with stream:true. ollama returns newline-delimited
    # JSON objects. each object with a non-empty message.content is yielded
    # as a string chunk. the final object (done:true) is skipped.
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model.model_id,
            "messages": self._build_messages(messages),
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 429:
                        raise AgentError(
                            f"Ollama rate limit: {resp.status}",
                            kind=ErrorKind.RATE_LIMIT,
                        )
                    if resp.status >= 400:
                        body = await resp.text()
                        raise AgentError(
                            f"Ollama error {resp.status}: {body}",
                            kind=ErrorKind.INTERNAL,
                        )
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = (obj.get("message") or {}).get("content") or ""
                        if chunk:
                            yield chunk
                        if obj.get("done"):
                            break
        except AgentError:
            raise
        except Exception as exc:
            kind = _classify_aiohttp_error(exc)
            raise AgentError(str(exc), kind=kind) from exc
