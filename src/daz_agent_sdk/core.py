from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Protocol, Type, cast

from daz_agent_sdk.capabilities.image import (
    download_image_job,
    generate_image,
    get_image_job,
    resume_image_job,
    resume_image_operation,
    submit_image_job,
)
from daz_agent_sdk.capabilities.stt import transcribe as _transcribe
from daz_agent_sdk.capabilities.tts import synthesize_speech
from daz_agent_sdk.config import Config, get_tier_chain, load_config
from daz_agent_sdk.conversation import Conversation
from daz_agent_sdk.fallback import execute_with_fallback
from daz_agent_sdk.registry import get_models_for_tier, get_provider, resolve_model
from daz_agent_sdk.types import (
    AgentError,
    AudioResult,
    Capability,
    EmbeddingResult,
    ErrorKind,
    ImageJobStatus,
    ImageResult,
    ImageSubmission,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    T,
    Tier,
)


class EmbeddingProvider(Protocol):
    async def embed(
        self,
        texts: list[str],
        *,
        task: str,
        batch_size: int,
        timeout: float,
    ) -> dict[str, Any]: ...


# ##################################################################
# agent
# top-level singleton that provides the primary API surface for
# all agent-sdk operations: ask, conversation, image, speak, models.
# all methods are async except conversation() which returns a context
# manager. a single module-level instance is created in __init__.py.
class Agent:
    # ##################################################################
    # init
    # accepts an optional Config; falls back to load_config() defaults
    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()

    # ##################################################################
    # ask
    # send a single-turn prompt and return a Response or StructuredResponse.
    # if provider and model are given, uses that pair directly bypassing
    # the tier chain. otherwise resolves the chain and uses fallback.
    # pass schema (a pydantic BaseModel subclass) to get StructuredResponse.
    async def ask(
        self,
        prompt: str,
        *,
        tier: Tier = Tier.HIGH,
        schema: Type[T] | None = None,
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,
        max_turns: int = 1,
        max_tokens: int | None = None,
        tools: list[str] | None = None,
        cwd: str | Path | None = None,
        setting_sources: list[str] | tuple[str, ...] | None = None,
    ) -> Response | StructuredResponse:
        # setting_sources restricts which on-disk config trees the spawned Claude CLI
        # loads (global ~/.claude plugins/skills/MCP vs project/local). An empty
        # collection means "load nothing global" — a lean environment. None keeps the
        # provider default. Only forwarded to providers that understand it (claude).
        complete_extra: dict[str, Any] = {}
        if setting_sources is not None:
            complete_extra["setting_sources"] = list(setting_sources)
        if max_tokens is not None:
            complete_extra["max_tokens"] = max_tokens
        messages: list[Message] = []
        if system is not None:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        if provider is not None and model is not None:
            prov = get_provider(provider)
            if prov is None:
                raise AgentError(
                    f"Provider '{provider}' is not available",
                    kind=ErrorKind.NOT_AVAILABLE,
                )
            minfo = resolve_model(provider, model, tier=tier)
            if minfo is None:
                raise AgentError(
                    f"Model '{provider}:{model}' could not be resolved",
                    kind=ErrorKind.NOT_AVAILABLE,
                )
            return await prov.complete(
                messages,
                minfo,
                schema=schema,
                timeout=timeout,
                tools=tools,
                cwd=cwd,
                max_turns=max_turns,
                **complete_extra,
            )

        chain = get_tier_chain(tier, self._config)

        async def execute_fn(provider_entry: str) -> Response | StructuredResponse:
            if ":" not in provider_entry:
                raise AgentError(
                    f"Invalid provider entry in chain: '{provider_entry}'",
                    kind=ErrorKind.INTERNAL,
                )
            prov_name, model_id = provider_entry.split(":", 1)
            prov = get_provider(prov_name)
            if prov is None:
                raise AgentError(
                    f"Provider '{prov_name}' is not available",
                    kind=ErrorKind.NOT_AVAILABLE,
                )
            minfo = resolve_model(prov_name, model_id, tier=tier)
            if minfo is None:
                raise AgentError(
                    f"Model '{prov_name}:{model_id}' could not be resolved",
                    kind=ErrorKind.INTERNAL,
                )
            return await prov.complete(
                messages,
                minfo,
                schema=schema,
                timeout=timeout,
                tools=tools,
                cwd=cwd,
                max_turns=max_turns,
                **complete_extra,
            )

        return await execute_with_fallback(
            tier.value,
            chain,
            execute_fn,
            config=self._config,
        )

    # ##################################################################
    # conversation
    # returns a Conversation async context manager for multi-turn exchanges.
    # the name is used for logging; tier, system, provider, and model are
    # forwarded to the Conversation constructor unchanged.
    # mcp_servers maps server names to McpStdioServerConfig-compatible dicts
    # and is forwarded to the claude provider on each turn when set.
    def conversation(
        self,
        name: str | None = None,
        *,
        tier: Tier = Tier.HIGH,
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        mcp_servers: dict[str, Any] | None = None,
    ) -> Conversation:
        return Conversation(
            name=name,
            tier=tier,
            system=system,
            provider=provider,
            model=model,
            config=self._config,
            mcp_servers=mcp_servers,
        )

    # ##################################################################
    # image
    # submit one generation or edit job to the durable Mac mini Codex image service.
    # width and height are required; output is optional. transparent is handled
    # by the service as part of the same durable job.
    async def image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output: str | Path | None = None,
        image: str | Path | list[str | Path] | None = None,
        tier: Tier = Tier.HIGH,
        transparent: bool = False,
        timeout: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        steps: int | None = None,
        idempotency_key: str | None = None,
        operation_state: str | Path | None = None,
    ) -> ImageResult:
        return await generate_image(
            prompt,
            width=width,
            height=height,
            output=output,
            image=image,
            tier=tier,
            transparent=transparent,
            timeout=timeout,
            provider=provider,
            model=model,
            steps=steps,
            idempotency_key=idempotency_key,
            operation_state=operation_state,
            config=self._config,
            conversation_id=uuid.uuid4(),
        )

    # ##################################################################
    # image job status
    # inspect durable generation state without creating a replacement job
    async def image_job_status(self, job_id: str) -> ImageJobStatus:
        return await get_image_job(job_id, config=self._config)

    # ##################################################################
    # submit image job
    # expose the durable submission identity without waiting for generation
    async def submit_image_job(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        image: str | Path | list[str | Path] | None = None,
        transparent: bool = False,
        idempotency_key: str,
    ) -> ImageSubmission:
        return await submit_image_job(
            prompt,
            width=width,
            height=height,
            image=image,
            transparent=transparent,
            idempotency_key=idempotency_key,
            config=self._config,
        )

    # ##################################################################
    # resume image job
    # continue waiting on a durable id and download its validated artifact
    async def resume_image_job(
        self,
        job_id: str,
        *,
        output: str | Path,
        timeout: float | None = None,
        transparent: bool = False,
    ) -> ImageResult:
        return await resume_image_job(
            job_id,
            output,
            timeout=timeout,
            transparent=transparent,
            config=self._config,
        )

    # ##################################################################
    # resume image operation
    # continue the exact request recorded before submission, including recovery
    # from an accepted response whose job identity was not yet persisted
    async def resume_image_operation(
        self,
        state_path: str | Path,
        *,
        timeout: float | None = None,
    ) -> ImageResult:
        return await resume_image_operation(
            state_path, timeout=timeout, config=self._config
        )

    # ##################################################################
    # download image job
    # fetch a completed durable artifact without polling or resubmitting
    async def download_image_job(
        self,
        job_id: str,
        *,
        output: str | Path,
        transparent: bool = False,
    ) -> ImageResult:
        return await download_image_job(
            job_id, output, transparent=transparent, config=self._config
        )

    # ##################################################################
    # speak
    # convert text to speech using the local tts subprocess.
    # voice defaults to "gary". output path is optional; a temp file is
    # used when omitted. speed adjusts playback rate (1.0 = normal).
    async def speak(
        self,
        text: str,
        *,
        voice: str = "gary",
        output: str | Path | None = None,
        speed: float = 1.0,
        timeout: float = 300.0,
    ) -> AudioResult:
        return await synthesize_speech(
            text,
            voice=voice,
            output=output,
            speed=speed,
            timeout=timeout,
            conversation_id=uuid.uuid4(),
        )

    # ##################################################################
    # transcribe
    # convert audio to text using the local whisper subprocess.
    # model_size controls the whisper variant (base, small, large-v3-turbo).
    # language is optional — whisper auto-detects if not provided.
    async def transcribe(
        self,
        audio: str | Path,
        *,
        model_size: str = "small",
        language: str | None = None,
        timeout: float = 300.0,
    ) -> str:
        return await _transcribe(
            audio,
            model_size=model_size,
            language=language,
            timeout=timeout,
            conversation_id=uuid.uuid4(),
        )

    # ##################################################################
    # remove_background
    # retain the former public method as an explicit fail-closed migration aid.
    async def remove_background(
        self,
        image: str | Path,
        *,
        timeout: float = 120.0,
    ) -> Path:
        from daz_agent_sdk.capabilities.image import _validate_legacy_image_config

        _validate_legacy_image_config(self._config)
        raise AgentError(
            "background removal is actively disabled — use the durable Mac mini Codex image service for still-image edits",
            kind=ErrorKind.INVALID_REQUEST,
        )

    # ##################################################################
    # embed
    # produce vector embeddings for a list of texts via the arbiter's
    # embed-text adapter on spark. Returns an EmbeddingResult with one
    # vector per input. Task may be "search_document", "search_query",
    # "classification", or "clustering" — nomic-embed-text uses prefix
    # instructions and this value determines the prefix used.
    async def embed(
        self,
        texts: list[str],
        *,
        task: str = "search_document",
        batch_size: int = 16,
        timeout: float = 600.0,
    ) -> EmbeddingResult:
        prov = get_provider("arbiter")
        if prov is None:
            raise AgentError(
                "Arbiter provider is not available for embeddings",
                kind=ErrorKind.NOT_AVAILABLE,
            )
        if not hasattr(prov, "embed"):
            raise AgentError(
                "Arbiter provider does not implement embed()",
                kind=ErrorKind.INTERNAL,
            )

        embedding_provider = cast(EmbeddingProvider, prov)
        raw = await embedding_provider.embed(
            texts,
            task=task,
            batch_size=batch_size,
            timeout=timeout,
        )

        embeddings = raw.get("embeddings") or []
        dimension = int(
            raw.get("dimension") or (len(embeddings[0]) if embeddings else 0)
        )
        model_id = raw.get("model_repository") or "embed-text"

        model_info = ModelInfo(
            provider="arbiter",
            model_id="embed-text",
            display_name=model_id,
            capabilities=frozenset({Capability.EMBEDDING}),
            tier=Tier.FREE_FAST,
            supports_streaming=False,
            supports_structured=False,
            supports_conversation=False,
            supports_tools=False,
        )

        return EmbeddingResult(
            embeddings=embeddings,
            model_used=model_info,
            dimension=dimension,
            task=task,
            usage={"elapsed_ms": raw.get("elapsed_ms"), "count": raw.get("count")},
        )

    # ##################################################################
    # models
    # return all known ModelInfo objects, optionally filtered by tier
    # and/or capability. loads providers lazily via the registry.
    async def models(
        self,
        *,
        tier: Tier | None = None,
        capability: Capability | None = None,
    ) -> list[ModelInfo]:
        if tier is not None:
            return get_models_for_tier(tier, capability=capability, config=self._config)

        # collect across all tiers, deduplicating by qualified name
        seen: set[str] = set()
        results: list[ModelInfo] = []
        for t in Tier:
            for m in get_models_for_tier(t, capability=capability, config=self._config):
                qn = m.qualified_name
                if qn not in seen:
                    seen.add(qn)
                    results.append(m)
        return results
