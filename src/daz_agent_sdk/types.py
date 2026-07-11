from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ##################################################################
# tier
# logical model quality tier — config maps these to concrete
# provider:model pairs. HIGH is the default everywhere.
class Tier(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FREE_FAST = "free_fast"
    FREE_THINKING = "free_thinking"
    SUMMARIES = "summaries"


# ##################################################################
# capability
# what kind of work a model can do
class Capability(Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    AGENTIC = "agentic"
    IMAGE = "image"
    TTS = "tts"
    STT = "stt"
    EMBEDDING = "embedding"


# ##################################################################
# error kind
# classifies errors for fallback decision making
class ErrorKind(Enum):
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    INTERNAL = "internal"
    NOT_AVAILABLE = "not_available"


# ##################################################################
# model info
# a concrete model offered by a provider, with its capabilities
@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model_id: str
    display_name: str
    capabilities: frozenset[Capability]
    tier: Tier
    supports_streaming: bool = True
    supports_structured: bool = True
    supports_conversation: bool = True
    supports_tools: bool = False
    max_context: int | None = None

    # ##################################################################
    # qualified name
    # provider:model_id format for config files and logging
    @property
    def qualified_name(self) -> str:
        return f"{self.provider}:{self.model_id}"


# ##################################################################
# message
# a single message in a conversation history
@dataclass
class Message:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # ##################################################################
    # to dict
    # serialise for logging and provider APIs
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# ##################################################################
# response
# result of an ask/say call
@dataclass
class Response:
    text: str
    model_used: ModelInfo
    conversation_id: UUID
    turn_id: UUID
    usage: dict[str, Any] = field(default_factory=dict)


# ##################################################################
# structured response
# result when a pydantic schema was provided
@dataclass
class StructuredResponse(Response):
    parsed: Any = None


# ##################################################################
# embedding result
# result of an embed() call — one float vector per input text, plus
# the model that produced them and the embedding dimension.
@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]
    model_used: ModelInfo
    dimension: int
    task: str = "search_document"
    usage: dict[str, Any] = field(default_factory=dict)


# ##################################################################
# image result
# result of an image generation call
@dataclass
class ImageResult:
    path: Path
    model_used: ModelInfo
    conversation_id: UUID
    prompt: str
    width: int
    height: int


# ##################################################################
# audio result
# result of a tts call
@dataclass
class AudioResult:
    path: Path
    model_used: ModelInfo
    conversation_id: UUID
    text: str
    voice: str
    duration_seconds: float | None = None


# ##################################################################
# agent error
# base error for all agent-sdk failures, carries structured context
# for debugging including all provider attempts
class AgentError(Exception):

    # ##################################################################
    # init
    # capture error kind and all provider attempts for diagnostics
    def __init__(
        self,
        message: str,
        kind: ErrorKind,
        attempts: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts or []

    # ##################################################################
    # to dict
    # structured representation for logging
    def to_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "kind": self.kind.value,
            "attempts": self.attempts,
        }


# ##################################################################
# parse json from llm
# extract JSON from LLM responses that may be wrapped in markdown
# code blocks
def parse_json_from_llm(text: str) -> Any:
    """Parse JSON from LLM response text.

    Handles: raw JSON, markdown-fenced JSON, and JSON embedded at the end
    of a prose response (common when agentic providers summarize their work
    then append the JSON).
    """
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown fences anywhere in the text
    import re
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the last JSON object in the text (scan backwards for '{')
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '}':
            # Found end of potential JSON, find matching opening brace
            depth = 0
            for j in range(i, -1, -1):
                if text[j] == '}':
                    depth += 1
                elif text[j] == '{':
                    depth -= 1
                    if depth == 0:
                        candidate = text[j:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            break

    raise json.JSONDecodeError("No valid JSON found in text", text, 0)


# ##################################################################
# validate structured json
# validate parsed LLM JSON against a pydantic schema, coercing the common
# model mistakes seen in the wild before giving up:
#   1. the "schema echo" quirk — the model returns {"properties": {...}}
#      (the JSON-schema shape) instead of an instance.
#   2. the "missing wrapper" quirk — for a schema whose single field is a
#      list of items, the model returns ONE item dict (or a bare list of
#      item dicts) instead of {"field": [...]}. Observed live: a
#      CharactersList {"characters": [...]} request answered with a single
#      character object, which previously failed the whole call.
# raises pydantic.ValidationError if nothing validates.
def validate_structured_json(schema: type[BaseModel], parsed_json: Any) -> BaseModel:
    from typing import get_args, get_origin

    # quirk 1: schema echo
    if isinstance(parsed_json, dict) and "properties" in parsed_json:
        props = parsed_json["properties"]
        if isinstance(props, dict) and all(not isinstance(v, dict) for v in props.values()):
            parsed_json = props

    try:
        return schema.model_validate(parsed_json)
    except Exception:
        # quirk 2: missing wrapper — only for single-list-field schemas
        fields = getattr(schema, "model_fields", {})
        if len(fields) == 1:
            (field_name, field_info), = fields.items()
            annotation = field_info.annotation
            if get_origin(annotation) is list:
                (item_type,) = get_args(annotation) or (None,)
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    if isinstance(parsed_json, list):
                        items = [item_type.model_validate(x) for x in parsed_json]
                        return schema.model_validate({field_name: items})
                    if isinstance(parsed_json, dict):
                        item = item_type.model_validate(parsed_json)
                        return schema.model_validate({field_name: [item]})
        raise
