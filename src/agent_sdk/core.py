from __future__ import annotations

import uuid
from pathlib import Path
from typing import Type

from agent_sdk.capabilities.image import generate_image
from agent_sdk.capabilities.stt import transcribe as _transcribe
from agent_sdk.capabilities.tts import synthesize_speech
from agent_sdk.config import Config, get_tier_chain, load_config
from agent_sdk.conversation import Conversation
from agent_sdk.fallback import execute_with_fallback
from agent_sdk.registry import get_models_for_tier, get_provider, resolve_model
from agent_sdk.types import (
    AgentError,
    AudioResult,
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
        timeout: float = 120.0,
        max_turns: int = 1,
        tools: list[str] | None = None,
        cwd: str | Path | None = None,
    ) -> Response | StructuredResponse:
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
    def conversation(
        self,
        name: str | None = None,
        *,
        tier: Tier = Tier.HIGH,
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> Conversation:
        return Conversation(
            name=name,
            tier=tier,
            system=system,
            provider=provider,
            model=model,
            config=self._config,
        )

    # ##################################################################
    # image
    # generate an image from a text prompt using the local generate_image
    # subprocess. width and height are required. output path is optional;
    # a temp file is used when omitted. transparent triggers background removal.
    async def image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output: str | Path | None = None,
        tier: Tier = Tier.HIGH,
        steps: int | None = None,
        transparent: bool = False,
        model: str | None = None,
        timeout: float = 600.0,
    ) -> ImageResult:
        return await generate_image(
            prompt,
            width=width,
            height=height,
            output=output,
            tier=tier,
            steps=steps,
            transparent=transparent,
            model=model,
            timeout=timeout,
            config=self._config,
            conversation_id=uuid.uuid4(),
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
        timeout: float = 120.0,
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
        timeout: float = 120.0,
    ) -> str:
        return await _transcribe(
            audio,
            model_size=model_size,
            language=language,
            timeout=timeout,
            conversation_id=uuid.uuid4(),
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
