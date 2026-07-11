from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Type
from uuid import uuid4

import aiohttp

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
    validate_structured_json,
)


# ##################################################################
# known arbiter LLM tiers
# map of served-model-name to tier, for models the arbiter exposes
# via its OpenAI-compatible /v1/chat/completions endpoint. used for
# ModelInfo construction when the /v1/models probe does not report
# enough metadata to infer the tier.
_KNOWN_TIERS: dict[str, Tier] = {
    "qwen3.6-27b": Tier.SUMMARIES,
    "qwen3.6-35b": Tier.FREE_THINKING,
    "gpt-oss-20b": Tier.FREE_FAST,
    "gemma4-31b": Tier.FREE_THINKING,
    "gemma4-26b": Tier.FREE_THINKING,
}


# ##################################################################
# classify aiohttp error
# map aiohttp exceptions to the agent-sdk ErrorKind taxonomy. mirrors
# the helper in the ollama provider so arbiter failures surface as
# the same NOT_AVAILABLE / TIMEOUT / INTERNAL kinds fallback expects.
def _classify_aiohttp_error(exc: Exception) -> ErrorKind:
    if isinstance(exc, aiohttp.ServerConnectionError | aiohttp.ClientConnectorError):
        return ErrorKind.NOT_AVAILABLE
    if isinstance(exc, TimeoutError | aiohttp.ServerTimeoutError):
        return ErrorKind.TIMEOUT
    return ErrorKind.INTERNAL


# ##################################################################
# arbiter provider
# HTTP provider for the spark arbiter (GPU job server) at
# http://10.0.0.254:8400 by default. speaks OpenAI-compatible
# /v1/chat/completions so any vLLM-served model registered with the
# arbiter is reachable by its served-model-name.
class ArbiterProvider(Provider):
    name = "arbiter"

    # ##################################################################
    # init
    # capture the arbiter base URL. defaults to the hardwired
    # spark endpoint; override via config.yaml when running off-network.
    def __init__(self, base_url: str = "http://10.0.0.254:8400") -> None:
        self._base_url = base_url.rstrip("/")

    # ##################################################################
    # available
    # probe /v1/models. the arbiter serves a JSON body there (native
    # format, not OpenAI envelope) — any HTTP 200 means the arbiter
    # is up and routable.
    async def available(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    # ##################################################################
    # list models
    # fetch /v1/models and keep entries that expose an `llm_name` —
    # those are the text LLM workers routable through
    # /v1/chat/completions. vision, image, tts, video workers are
    # filtered out. tier is taken from _KNOWN_TIERS when available,
    # else FREE_THINKING (arbiter LLMs are all 20B+).
    async def list_models(self) -> list[ModelInfo]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
        except Exception:
            return []

        entries: list[dict[str, Any]] = []
        if isinstance(data, list):
            entries = [e for e in data if isinstance(e, dict)]
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            entries = [e for e in data["data"] if isinstance(e, dict)]

        models: list[ModelInfo] = []
        for entry in entries:
            llm_name = entry.get("llm_name")
            if not llm_name:
                continue
            tier = _KNOWN_TIERS.get(llm_name, Tier.FREE_THINKING)
            display = llm_name.replace("-", " ").replace("_", " ").title()
            models.append(
                ModelInfo(
                    provider="arbiter",
                    model_id=llm_name,
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
    # convert agent Message objects to the OpenAI chat payload form.
    # system/user/assistant roles pass straight through.
    def _build_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    # ##################################################################
    # complete
    # POST to /v1/chat/completions with stream:false. structured output
    # is delivered via schema-in-prompt (appended to the last system
    # message) plus an OpenAI response_format hint — the arbiter's
    # vLLM worker ignores response_format it doesn't support, but the
    # prompt instruction ensures the model still returns valid JSON.
    async def complete(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        schema: Type[T] | None = None,
        tools: list[str] | None = None,
        cwd: str | Path | None = None,
        max_turns: int = 1,
        timeout: float = 900.0,
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
            system_indices = [i for i, m in enumerate(msg_list) if m["role"] == "system"]
            if system_indices:
                last_sys = system_indices[-1]
                msg_list[last_sys] = {
                    "role": "system",
                    "content": msg_list[last_sys]["content"] + instruction,
                }
            else:
                msg_list.insert(0, {"role": "system", "content": instruction.strip()})

        payload: dict[str, Any] = {
            "model": model.model_id,
            "messages": msg_list,
            "stream": False,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 429:
                        raise AgentError(
                            f"Arbiter rate limit: {resp.status}",
                            kind=ErrorKind.RATE_LIMIT,
                        )
                    if resp.status >= 400:
                        body = await resp.text()
                        raise AgentError(
                            f"Arbiter error {resp.status}: {body}",
                            kind=ErrorKind.INTERNAL,
                        )
                    data = await resp.json()
        except AgentError:
            raise
        except Exception as exc:
            kind = _classify_aiohttp_error(exc)
            raise AgentError(str(exc), kind=kind) from exc

        choices = data.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            # reasoning models (qwen3 via vLLM --reasoning-parser qwen3) put
            # the chain-of-thought in `reasoning` and the final answer in
            # `content`. callers want the answer; fall through to reasoning
            # only when content is empty (e.g. the model truncated before
            # emitting a final answer).
            text = message.get("content") or message.get("reasoning") or ""
        usage_data = data.get("usage", {}) or {}

        if schema is not None:
            try:
                parsed_json = parse_json_from_llm(text)
                parsed_instance = validate_structured_json(schema, parsed_json)
            except Exception as exc:
                # the MODEL failed to satisfy the schema — a retry or another
                # provider may succeed, so this must stay retryable (INTERNAL),
                # never INVALID_REQUEST which aborts the whole fallback chain.
                raise AgentError(
                    f"Failed to parse structured response: {exc}",
                    kind=ErrorKind.INTERNAL,
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
    # POST to /v1/chat/completions with stream:true. the arbiter emits
    # OpenAI Server-Sent Events — each line prefixed `data: ` and the
    # terminator `data: [DONE]`. choices[0].delta.content is yielded
    # for each non-empty chunk.
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 300.0,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model.model_id,
            "messages": self._build_messages(messages),
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 429:
                        raise AgentError(
                            f"Arbiter rate limit: {resp.status}",
                            kind=ErrorKind.RATE_LIMIT,
                        )
                    if resp.status >= 400:
                        body = await resp.text()
                        raise AgentError(
                            f"Arbiter error {resp.status}: {body}",
                            kind=ErrorKind.INTERNAL,
                        )
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[len("data:"):].strip()
                        if payload_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        chunk = delta.get("content") or delta.get("reasoning") or ""
                        if chunk:
                            yield chunk
        except AgentError:
            raise
        except Exception as exc:
            kind = _classify_aiohttp_error(exc)
            raise AgentError(str(exc), kind=kind) from exc

    # ##################################################################
    # embed
    # submit an embed-text job to the arbiter and poll until complete.
    # the arbiter job API is async-by-design — submit returns a job_id,
    # then poll /v1/jobs/<id> until a terminal state. nomic-embed-text-v1.5
    # is ~50ms/text warm, ~5s cold; we use a 1s poll interval.
    async def embed(
        self,
        texts: list[str],
        *,
        task: str = "search_document",
        batch_size: int = 16,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        if not texts:
            raise AgentError(
                "embed() requires at least one text",
                kind=ErrorKind.INVALID_REQUEST,
            )

        payload: dict[str, Any] = {
            "type": "embed-text",
            "params": {
                "texts": list(texts),
                "task": task,
                "batch_size": batch_size,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/v1/jobs",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120.0),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise AgentError(
                            f"Arbiter embed submit error {resp.status}: {body}",
                            kind=ErrorKind.INTERNAL,
                        )
                    submit = await resp.json()

                job_id = submit.get("job_id")
                if not job_id:
                    raise AgentError(
                        f"Arbiter returned no job_id: {submit}",
                        kind=ErrorKind.INTERNAL,
                    )

                import asyncio as _asyncio
                deadline = _asyncio.get_event_loop().time() + timeout
                while True:
                    if _asyncio.get_event_loop().time() > deadline:
                        raise AgentError(
                            f"Arbiter embed job {job_id} timed out after {timeout}s",
                            kind=ErrorKind.TIMEOUT,
                        )
                    async with session.get(
                        f"{self._base_url}/v1/jobs/{job_id}",
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as poll:
                        if poll.status >= 400:
                            body = await poll.text()
                            raise AgentError(
                                f"Arbiter embed poll error {poll.status}: {body}",
                                kind=ErrorKind.INTERNAL,
                            )
                        status = await poll.json()
                    state = status.get("status", "")
                    if state == "completed":
                        return status.get("result") or {}
                    if state in ("failed", "cancelled"):
                        raise AgentError(
                            f"Arbiter embed job {job_id} {state}: {status.get('error','')}",
                            kind=ErrorKind.INTERNAL,
                        )
                    await _asyncio.sleep(1.0)
        except AgentError:
            raise
        except Exception as exc:
            kind = _classify_aiohttp_error(exc)
            raise AgentError(str(exc), kind=kind) from exc
