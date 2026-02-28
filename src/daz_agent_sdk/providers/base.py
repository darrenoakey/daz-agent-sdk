from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Type

from daz_agent_sdk.types import ImageResult, Message, ModelInfo, Response, StructuredResponse, T


# ##################################################################
# provider
# abstract base class for all AI providers — defines the interface
# that every provider must implement for completion, streaming, and
# model enumeration. image generation is optional (raises by default).
class Provider(ABC):
    name: str

    # ##################################################################
    # available
    # check if this provider is reachable and ready to serve requests.
    # should be fast and non-blocking — used during fallback selection.
    @abstractmethod
    async def available(self) -> bool: ...

    # ##################################################################
    # list models
    # return all models this provider currently offers. called to populate
    # the registry and for agent.models() enumeration.
    @abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...

    # ##################################################################
    # complete
    # send messages and return a full response. if schema is provided,
    # returns StructuredResponse with a validated pydantic instance in .parsed.
    # tools and cwd are used by agentic providers (claude, codex) only.
    @abstractmethod
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
    ) -> Response | StructuredResponse: ...

    # ##################################################################
    # stream
    # send messages and yield response text chunks as they arrive.
    # callers consume this as an async generator.
    # NOTE: not declared async — subclasses implement as async generators
    # which are AsyncIterator[str], not coroutines returning AsyncIterator[str].
    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]: ...

    # ##################################################################
    # generate image
    # generate an image from a text prompt. raises NotImplementedError
    # for providers that do not support image generation — callers
    # should check capabilities before calling.
    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output: Path,
        **kwargs,
    ) -> ImageResult:
        raise NotImplementedError(f"{self.name} does not support image generation")
