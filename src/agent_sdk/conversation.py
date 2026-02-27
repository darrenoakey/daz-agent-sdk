from __future__ import annotations

import copy
from typing import AsyncIterator, Type
from uuid import UUID, uuid4

from agent_sdk.config import Config, get_tier_chain, load_config
from agent_sdk.fallback import execute_with_fallback
from agent_sdk.logging_ import ConversationLogger
from agent_sdk.providers.base import Provider
from agent_sdk.registry import get_provider, resolve_model
from agent_sdk.types import Message, ModelInfo, Response, StructuredResponse, T, Tier


# ##################################################################
# conversation
# multi-turn conversation that maintains message history and delegates
# to providers via the fallback engine. use as an async context manager.
#
# example:
#   async with agent.conversation("book-editor") as chat:
#       outline = await chat.say("Write an outline", schema=Outline)
#       chapter1 = await chat.say("Write chapter 1")
#       async for chunk in chat.stream("Write chapter 2"):
#           print(chunk, end="", flush=True)
class Conversation:

    # ##################################################################
    # init
    # set up a new conversation with optional name, tier, system prompt,
    # provider/model overrides, and config. the logger is created on enter.
    def __init__(
        self,
        name: str | None = None,
        *,
        tier: Tier = Tier.HIGH,
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        config: Config | None = None,
    ) -> None:
        self._name = name
        self._tier = tier
        self._system = system
        self._provider_name = provider
        self._model_id = model
        self._config = config
        self._history: list[Message] = []
        self._logger: ConversationLogger | None = None
        self._conversation_id: UUID | None = None
        # ##################################################################
        # pre-populate system message if provided
        # the system message is always first in the history list
        if system is not None:
            self._history.append(Message(role="system", content=system))

    # ##################################################################
    # aenter
    # create the conversation logger and assign a unique conversation id
    async def __aenter__(self) -> Conversation:
        self._conversation_id = uuid4()
        cfg = self._config or load_config()
        self._logger = ConversationLogger(
            self._conversation_id,
            name=self._name,
            tier=self._tier.value,
            provider=self._provider_name,
            model=self._model_id,
            log_base=cfg.logging.directory,
        )
        return self

    # ##################################################################
    # aexit
    # close the logger on context exit regardless of exception state
    async def __aexit__(self, *args) -> None:
        if self._logger is not None:
            self._logger.close()

    # ##################################################################
    # say
    # add a user message, call the provider via fallback, append the
    # assistant response to history, and return the response object.
    # pass schema to receive a StructuredResponse with a parsed pydantic model.
    # tier overrides the instance-level tier for this call only.
    async def say(
        self,
        content: str,
        *,
        schema: Type[T] | None = None,
        tier: Tier | None = None,
        timeout: float = 120.0,
    ) -> Response | StructuredResponse:
        effective_tier = tier if tier is not None else self._tier
        self._history.append(Message(role="user", content=content))
        messages = list(self._history)

        provider_entry, model_info = self._resolve_provider_model(effective_tier)
        providers_chain = self._build_providers_chain(effective_tier, provider_entry)

        async def execute_fn(chain_entry: str) -> Response | StructuredResponse:
            pname, mid = _split_entry(chain_entry)
            prov = _require_provider(pname)
            minfo = _require_model(pname, mid, effective_tier)
            return await prov.complete(
                messages,
                minfo,
                schema=schema,
                timeout=timeout,
            )

        result = await execute_with_fallback(
            effective_tier.value,
            providers_chain,
            execute_fn,
            is_conversation=True,
            config=self._config,
            logger=self._logger,
        )

        self._history.append(Message(role="assistant", content=result.text))
        return result

    # ##################################################################
    # stream
    # add a user message, stream chunks from the provider, collect the
    # full text, append it to history, and yield each chunk to the caller.
    # tier overrides the instance-level tier for this call only.
    async def stream(
        self,
        content: str,
        *,
        tier: Tier | None = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        effective_tier = tier if tier is not None else self._tier
        self._history.append(Message(role="user", content=content))
        messages = list(self._history)

        provider_entry, _ = self._resolve_provider_model(effective_tier)
        providers_chain = self._build_providers_chain(effective_tier, provider_entry)

        # resolve a concrete provider/model for streaming — use first in chain
        first_entry = providers_chain[0] if providers_chain else provider_entry or ""
        pname, mid = _split_entry(first_entry)
        prov = _require_provider(pname)
        minfo = _require_model(pname, mid, effective_tier)

        chunks: list[str] = []

        async def _generate() -> AsyncIterator[str]:
            async for chunk in prov.stream(messages, minfo, timeout=timeout):
                chunks.append(chunk)
                yield chunk

        async for chunk in _generate():
            yield chunk

        full_text = "".join(chunks)
        self._history.append(Message(role="assistant", content=full_text))

    # ##################################################################
    # history property
    # returns a snapshot of the current conversation history
    @property
    def history(self) -> list[Message]:
        return list(self._history)

    # ##################################################################
    # name property
    # returns the conversation name
    @property
    def name(self) -> str | None:
        return self._name

    # ##################################################################
    # tier property
    # returns the configured tier for this conversation
    @property
    def tier(self) -> Tier:
        return self._tier

    # ##################################################################
    # fork
    # create a new Conversation with a copy of the current history.
    # changes to the fork do not affect the original.
    def fork(self, name: str | None = None) -> Conversation:
        forked = Conversation(
            name=name,
            tier=self._tier,
            provider=self._provider_name,
            model=self._model_id,
            config=self._config,
        )
        # system is already baked into history — copy history directly,
        # skipping the __init__ system message injection
        forked._history = copy.deepcopy(self._history)
        return forked

    # ##################################################################
    # resolve provider model
    # find the explicit provider+model override or derive from tier chain
    def _resolve_provider_model(self, tier: Tier) -> tuple[str | None, ModelInfo | None]:
        if self._provider_name and self._model_id:
            entry = f"{self._provider_name}:{self._model_id}"
            minfo = resolve_model(self._provider_name, self._model_id, tier=tier)
            return entry, minfo
        return None, None

    # ##################################################################
    # build providers chain
    # produce an ordered list of provider:model entries for fallback.
    # if a single override entry is set it forms a one-element chain;
    # otherwise the tier config chain is used.
    def _build_providers_chain(self, tier: Tier, override_entry: str | None) -> list[str]:
        if override_entry is not None:
            return [override_entry]
        cfg = self._config or load_config()
        return get_tier_chain(tier, cfg)


# ##################################################################
# split entry
# split a "provider:model_id" string into its two parts.
# returns ("", "") for malformed entries.
def _split_entry(entry: str) -> tuple[str, str]:
    if ":" not in entry:
        return ("", "")
    provider_name, model_id = entry.split(":", 1)
    return provider_name, model_id


# ##################################################################
# require provider
# look up a provider by name and raise RuntimeError if not available.
def _require_provider(name: str) -> Provider:
    prov = get_provider(name)
    if prov is None:
        raise RuntimeError(f"Provider '{name}' is not available")
    return prov


# ##################################################################
# require model
# look up a ModelInfo by provider name and model id, raising RuntimeError
# if the model cannot be resolved.
def _require_model(provider_name: str, model_id: str, tier: Tier) -> ModelInfo:
    minfo = resolve_model(provider_name, model_id, tier=tier)
    if minfo is None:
        raise RuntimeError(f"Model '{provider_name}:{model_id}' could not be resolved")
    return minfo
