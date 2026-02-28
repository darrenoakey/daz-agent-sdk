# daz-agent-sdk Design Document

## Philosophy

One library. Every AI capability. Provider-agnostic. The caller says **what** they need
(tier + capability), never **which** provider — unless they want to.

```python
from daz_agent_sdk import agent

# simplest possible call
answer = await agent.ask("Summarise this text", text=doc)

# structured output
recipe = await agent.ask("Give me a pancake recipe", schema=Recipe)

# conversation
async with agent.conversation("book-editor") as chat:
    outline = await chat.say("Write an outline for a thriller", schema=Outline)
    chapter1 = await chat.say("Write chapter 1 based on that outline")
    async for chunk in chat.stream("Now write chapter 2"):
        print(chunk, end="", flush=True)

# image
img = await agent.image("A cyberpunk city at sunset", width=1024, height=1024)

# tts
audio = await agent.speak("Welcome to the show", voice="gary")
```

---

## Package Layout

```
daz-agent-sdk/
├── src/
│   └── daz_agent_sdk/
│       ├── __init__.py          # public API: agent, Conversation, Tier, etc.
│       ├── core.py              # Agent singleton, top-level ask/image/speak/conversation
│       ├── conversation.py      # Conversation class (context manager)
│       ├── config.py            # ~/.daz-agent-sdk/ loader + defaults
│       ├── registry.py          # provider/model registry + tier mapping
│       ├── fallback.py          # retry, backoff, provider cascade
│       ├── logging_.py          # per-conversation GUID logging + tracing
│       ├── types.py             # Tier, Capability, Message, ModelInfo, etc.
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py          # Provider ABC
│       │   ├── claude.py        # claude_agent_sdk wrapper
│       │   ├── codex.py         # openai-codex-sdk wrapper
│       │   ├── gemini.py        # google-genai wrapper
│       │   ├── ollama.py        # HTTP to localhost:11434
│       │   └── vllm.py          # HTTP to local vLLM server
│       └── capabilities/
│           ├── __init__.py
│           ├── image.py         # image generation (mflux, DALL-E, Gemini)
│           ├── tts.py           # text-to-speech (Qwen3-TTS, etc.)
│           └── stt.py           # speech-to-text (Whisper, etc.)
├── run                          # facade script (venv, pip install, delegate)
├── requirements.txt
└── src/daz_agent_sdk_test.py        # co-located tests
```

---

## Types

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Type, TypeVar
from uuid import UUID
from pathlib import Path
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Tier(Enum):
    """Logical model tier. Config maps these to concrete provider:model pairs.

    HIGH is the default everywhere — callers who don't specify a tier get the
    best available model. Use lower tiers explicitly when cost/speed matters.
    """
    HIGH = "high"              # best quality (claude-opus, gpt-5.3-codex, gemini-2.5-pro) — DEFAULT
    MEDIUM = "medium"          # balanced (claude-sonnet, gemini-2.5-flash)
    LOW = "low"                # fast + cheap (claude-haiku, gemini-2.5-flash-lite)
    FREE_FAST = "free_fast"    # local, no cost, fast (ollama small model)
    FREE_THINKING = "free_thinking"  # local, no cost, deeper (ollama large model)


class Capability(Enum):
    """What kind of work the model needs to do."""
    TEXT = "text"
    STRUCTURED = "structured"  # JSON/Pydantic output
    AGENTIC = "agentic"        # tool use, multi-turn autonomous
    IMAGE = "image"
    TTS = "tts"
    STT = "stt"


@dataclass(frozen=True)
class ModelInfo:
    """A concrete model offered by a provider."""
    provider: str              # "claude", "codex", "gemini", "ollama", "vllm"
    model_id: str              # "claude-opus-4-6", "gpt-5.3-codex", "gemini-2.5-flash"
    display_name: str
    capabilities: frozenset[Capability]
    tier: Tier
    supports_streaming: bool = True
    supports_structured: bool = True
    supports_conversation: bool = True
    supports_tools: bool = False
    max_context: int | None = None


@dataclass
class Message:
    """A single message in a conversation."""
    role: str                  # "user", "assistant", "system"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Result of an ask/say call."""
    text: str
    model_used: ModelInfo
    conversation_id: UUID
    turn_id: UUID
    usage: dict[str, Any] = field(default_factory=dict)  # tokens, cost, etc.


@dataclass
class StructuredResponse(Response):
    """Result when a schema was provided."""
    parsed: Any = None         # the Pydantic model instance


@dataclass
class ImageResult:
    """Result of an image generation call."""
    path: Path
    model_used: ModelInfo
    conversation_id: UUID
    prompt: str
    width: int
    height: int


@dataclass
class AudioResult:
    """Result of a TTS call."""
    path: Path
    model_used: ModelInfo
    conversation_id: UUID
    text: str
    voice: str
    duration_seconds: float | None = None
```

---

## Configuration: ~/.daz-agent-sdk/

```yaml
# ~/.daz-agent-sdk/config.yaml
# Everything here is optional. Sensible defaults apply.

# Map tiers to ordered provider:model lists (first = preferred, rest = fallbacks)
tiers:
  high:
    - claude:claude-opus-4-6
    - codex:gpt-5.3-codex
    - gemini:gemini-2.5-pro
  medium:
    - claude:claude-sonnet-4-6
    - codex:gpt-5.3-codex
    - gemini:gemini-2.5-flash
  low:
    - claude:claude-haiku-4-5-20251001
    - gemini:gemini-2.5-flash-lite
    - ollama:qwen3-8b
  free_fast:
    - ollama:qwen3-8b
    - vllm:default
  free_thinking:
    - ollama:qwen3-30b-32k
    - ollama:deepseek-r1:14b

# Provider-specific configuration
providers:
  claude:
    # No config needed — uses ambient Claude Code auth
    permission_mode: bypassPermissions
  codex:
    # No config needed — uses ambient Codex auth (or OPENAI_API_KEY)
  gemini:
    api_key_env: GEMINI_API_KEY       # or uses keyring
  ollama:
    base_url: http://localhost:11434
  vllm:
    base_url: http://localhost:8000

# Image generation — tier controls quality/speed tradeoff via steps
# All tiers use z-image-turbo for now; steps vary by tier
image:
  model: z-image-turbo                    # default model for all tiers
  tiers:
    high:   { steps: 8 }                  # best quality, ~40s
    medium: { steps: 4 }                  # balanced, ~20s
    low:    { steps: 1 }                  # draft quality, ~5s
  fallback:                               # if local fails, try these
    - gemini:gemini-2.5-flash
    - openai:dall-e-3
  transparent:
    post_process: birefnet                # background removal after generation

# TTS configuration
tts:
  default:
    - local:qwen3-tts                 # ~/bin/tts
  voices:
    gary: { provider: local, voice_id: gary, description: "British newsreader" }
    aiden: { provider: local, voice_id: aiden }

# Logging
logging:
  directory: ~/.daz-agent-sdk/logs        # {conversation_id}/ subdirectories
  level: info                         # debug for full request/response bodies
  retention_days: 30

# Fallback behaviour
fallback:
  single_shot:
    # On rate limit / capacity error, immediately try next provider
    strategy: immediate_cascade
  conversation:
    # Exponential backoff first, then summarise + cascade
    strategy: backoff_then_cascade
    max_backoff_seconds: 60
    summarise_with: free_thinking     # tier to use for conversation summarisation
```

---

## Core API

### agent — Module-Level Singleton

```python
# src/daz_agent_sdk/__init__.py

from daz_agent_sdk.core import Agent

# Module-level singleton — ready to use on import
agent = Agent()
```

### agent.ask() — Single-Shot Query

```python
async def ask(
    prompt: str,
    *,
    tier: Tier = Tier.HIGH,
    schema: Type[T] | None = None,       # Pydantic model → structured output
    system: str | None = None,
    provider: str | None = None,          # override: "claude", "codex", "gemini", "ollama"
    model: str | None = None,             # override: exact model id
    timeout: float = 120.0,
    max_turns: int = 1,                   # >1 enables agentic tool use
    tools: list[str] | None = None,       # ["Read", "Bash", ...] for agentic
    cwd: str | Path | None = None,
) -> Response | StructuredResponse:
    """
    One-shot query. Returns when complete.

    Tier defaults to HIGH — callers get the best model unless they
    explicitly opt for something cheaper/faster.

    Fallback: on rate-limit/capacity error, immediately tries next
    provider in the tier's fallback chain.

    If schema is provided, returns StructuredResponse with .parsed
    containing the validated Pydantic instance.
    """
```

**Usage:**

```python
from daz_agent_sdk import agent, Tier
from pydantic import BaseModel

# plain text — defaults to Tier.HIGH (best available model)
answer = await agent.ask("Explain quantum tunnelling in one paragraph")

# explicit lower tier when cost/speed matters
answer = await agent.ask("Summarise this", tier=Tier.LOW)

# structured
class Sentiment(BaseModel):
    label: str    # positive, negative, neutral
    confidence: float

result = await agent.ask(
    "Classify: 'I love this product'",
    schema=Sentiment,
    tier=Tier.LOW,
)
print(result.parsed.label)  # "positive"

# agentic (with tools)
result = await agent.ask(
    "Find all TODO comments in this project and list them",
    tools=["Read", "Glob", "Grep"],
    max_turns=10,
    cwd="/path/to/project",
)
```

### agent.conversation() — Persistent Multi-Turn

```python
def conversation(
    name: str | None = None,              # human-readable label for logging
    *,
    tier: Tier = Tier.HIGH,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    tools: list[str] | None = None,
    cwd: str | Path | None = None,
    max_turns_per_message: int = 1,
) -> Conversation:
    """
    Create a persistent conversation.

    Usage:
        async with agent.conversation("my-task") as chat:
            answer = await chat.say("Hello")
            async for chunk in chat.stream("Tell me more"):
                print(chunk, end="")
    """
```

### Conversation Class

```python
class Conversation:
    """
    A multi-turn conversation with a model.

    Maintains message history. Handles fallback by summarising
    history and retrying on a different provider.
    """

    id: UUID                               # conversation GUID (for logging)
    name: str | None
    tier: Tier
    provider: str                          # current active provider
    model: ModelInfo                       # current active model
    history: list[Message]                 # full conversation history

    async def __aenter__(self) -> Conversation: ...
    async def __aexit__(self, *exc) -> None: ...

    async def say(
        self,
        message: str,
        *,
        schema: Type[T] | None = None,
        tier: Tier | None = None,          # override tier for this turn only
        timeout: float = 120.0,
    ) -> Response | StructuredResponse:
        """
        Send a message, wait for complete response.
        Appends both user message and assistant response to history.
        """

    async def stream(
        self,
        message: str,
        *,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """
        Send a message, yield response chunks as they arrive.
        Appends both user message and full assistant response to history.
        """

    async def summarise(self) -> str:
        """
        Summarise the conversation so far using free_thinking tier.
        Used internally when cascading to a different provider.
        Returns the summary text.
        """

    def fork(self, name: str | None = None) -> Conversation:
        """
        Create a child conversation that shares history up to this point
        but diverges from here. Useful for exploring alternatives.
        """
```

### agent.image() — Image Generation

```python
async def image(
    prompt: str,
    *,
    width: int = 512,
    height: int = 512,
    output: str | Path | None = None,     # default: temp file
    tier: Tier = Tier.HIGH,               # HIGH=8 steps, MEDIUM=4, LOW=1
    transparent: bool = False,
    model: str | None = None,             # override: "mflux", "dall-e-3", "gemini"
    steps: int | None = None,             # override: explicit step count (bypasses tier)
    guidance: float | None = None,
    seed: int | None = None,
    timeout: float = 600.0,
) -> ImageResult:
    """Generate an image from a text prompt.

    Tier controls quality via inference steps (configurable in config.yaml).
    Pass steps= explicitly to override the tier's default.
    """
```

**Usage:**

```python
result = await agent.image(
    "Professional book cover, atmospheric thriller",
    width=768, height=1024,
    output="cover.jpg",
)
print(result.path)  # 8 steps (HIGH default)

# quick draft for previewing
draft = await agent.image("Robot logo", tier=Tier.LOW)  # 1 step, ~5s
```

### agent.speak() — Text-to-Speech

```python
async def speak(
    text: str,
    *,
    voice: str = "aiden",
    output: str | Path | None = None,
    speed: float = 1.0,
    timeout: float = 120.0,
) -> AudioResult:
    """Convert text to speech audio."""
```

### agent.transcribe() — Speech-to-Text

```python
async def transcribe(
    audio: str | Path,
    *,
    model_size: str = "small",            # "base", "small", "large-v3-turbo"
    language: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Transcribe audio to text."""
```

### agent.models() — Enumerate Available Models

```python
async def models(
    *,
    capability: Capability | None = None,
    tier: Tier | None = None,
    provider: str | None = None,
) -> list[ModelInfo]:
    """
    List available models, optionally filtered.

    Checks provider availability (is Ollama running? is Codex installed?).

    Usage:
        all_models = await agent.models()
        high_text = await agent.models(tier=Tier.HIGH, capability=Capability.TEXT)
        local = await agent.models(provider="ollama")
    """
```

---

## Provider ABC

```python
class Provider(ABC):
    """Base class for all AI providers."""

    name: str                              # "claude", "codex", "gemini", "ollama", "vllm"

    @abstractmethod
    async def available(self) -> bool:
        """Check if this provider is reachable and authenticated."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List models this provider offers."""

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
    ) -> Response | StructuredResponse:
        """Send messages and get a complete response."""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Send messages and yield response chunks."""

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        *,
        width: int, height: int,
        output: Path,
        **kwargs,
    ) -> ImageResult:
        """Generate an image. Raise NotImplementedError if unsupported."""
```

**Provider implementations:**

| Provider | `complete()` | `stream()` | `generate_image()` | Notes |
|----------|-------------|-----------|-------------------|-------|
| **claude** | `claude_agent_sdk.query()` | same, yield chunks | N/A | Strips CLAUDECODE env; handles tool_use |
| **codex** | `thread.run()` | `thread.run_streamed()` | N/A | Manages thread lifecycle; binary install check |
| **gemini** | `client.models.generate_content()` | `generate_content_stream()` | `generate_content()` with image config | Uses google-genai SDK |
| **ollama** | `POST /api/chat` | same with `stream: true` | N/A | Local HTTP, no auth |
| **vllm** | OpenAI-compatible endpoint | same | N/A | Local HTTP |
| **local** (image) | subprocess `generate_image` | N/A | `subprocess.run(["generate_image", ...])` | mflux + BiRefNet |
| **local** (tts) | subprocess `tts` | N/A | N/A | Qwen3-TTS |

---

## Fallback Engine

The core value proposition. Two strategies:

### Single-Shot Cascade (for `agent.ask()`)

```
Request to Tier.HIGH
  → Try provider #1 (claude-opus)
    → Rate limit / capacity error?
      → IMMEDIATELY try provider #2 (codex-gpt-5.3)
        → Rate limit?
          → IMMEDIATELY try provider #3 (gemini-2.5-pro)
            → All failed? Raise AgentError with all attempts logged
```

No delay. No backoff. Just cascade through the fallback chain.

### Conversation Cascade (for `chat.say()` / `chat.stream()`)

```
Request on current provider (claude-sonnet)
  → Rate limit / capacity error?
    → Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (max)
      → Still failing after max_backoff_seconds?
        → Summarise conversation history using free_thinking tier
        → Start new provider session (codex) with:
            system = original system prompt
            first message = "[Conversation summary]\n\n" + summary + "\n\n[Continuing]\n\n" + current_message
        → Continue conversation on new provider
        → Log the provider switch in conversation trace
```

### Error Classification

```python
class ErrorKind(Enum):
    RATE_LIMIT = "rate_limit"          # 429, capacity, overloaded → cascade
    AUTH = "auth"                       # 401, 403 → skip provider permanently
    TIMEOUT = "timeout"                 # deadline exceeded → cascade
    INVALID_REQUEST = "invalid_request" # 400, bad schema → raise immediately (caller bug)
    INTERNAL = "internal"              # 500, unexpected → cascade with warning
    NOT_AVAILABLE = "not_available"    # provider offline → skip
```

---

## Logging and Tracing

Every conversation gets a UUID. All interactions are logged to `~/.daz-agent-sdk/logs/{uuid}/`.

```
~/.daz-agent-sdk/logs/
  a1b2c3d4-.../
    meta.json          # { name, tier, provider, model, created_at, ... }
    events.jsonl       # append-only event log
    summary.txt        # final summary if conversation was summarised
```

**Event types in events.jsonl:**

```jsonl
{"ts":"...","event":"conversation_start","tier":"medium","provider":"claude","model":"claude-sonnet-4-6"}
{"ts":"...","event":"user_message","turn":1,"content":"Write an outline...","tokens_est":42}
{"ts":"...","event":"assistant_response","turn":1,"content":"Here is...","tokens":1523,"duration_ms":3200}
{"ts":"...","event":"rate_limit","provider":"claude","status":429,"retry_after":null}
{"ts":"...","event":"backoff","attempt":1,"delay_ms":1000}
{"ts":"...","event":"cascade","from_provider":"claude","to_provider":"codex","reason":"rate_limit","summarised":true}
{"ts":"...","event":"image_request","prompt":"...","width":1024,"height":1024,"provider":"local"}
{"ts":"...","event":"image_complete","path":"/tmp/gen.jpg","duration_ms":45000}
{"ts":"...","event":"structured_output","schema":"Sentiment","parsed":{"label":"positive","confidence":0.95}}
{"ts":"...","event":"conversation_end","turns":5,"total_tokens":12340,"providers_used":["claude","codex"]}
```

**Programmatic access to logs:**

```python
# The conversation exposes its log
async with agent.conversation("my-task") as chat:
    print(chat.id)           # UUID
    print(chat.log_dir)      # Path to log directory
```

---

## Complete Usage Examples

### Example 1: What most projects will look like (noveliser2-style)

```python
from daz_agent_sdk import agent, Tier
from pydantic import BaseModel


class Chapter(BaseModel):
    title: str
    summary: str
    word_target: int


class Outline(BaseModel):
    chapters: list[Chapter]


async def write_novel(premise: str) -> None:
    # Structured output for planning (high tier)
    outline = await agent.ask(
        f"Create a 10-chapter novel outline for: {premise}",
        schema=Outline,
        tier=Tier.HIGH,
    )

    # Write each chapter (low tier — fast, cheap)
    for ch in outline.parsed.chapters:
        text = await agent.ask(
            f"Write chapter '{ch.title}'. Summary: {ch.summary}. "
            f"Target: {ch.word_target} words.",
            tier=Tier.LOW,
        )
        save_chapter(ch.title, text.text)

    # Generate cover (image)
    await agent.image(
        f"Book cover for novel about: {premise}",
        width=768, height=1024,
        output="cover.jpg",
    )
```

### Example 2: Conversational agent (claude_server-style)

```python
from daz_agent_sdk import agent, Tier


async def handle_user_session(user_messages):
    async with agent.conversation(
        "user-session",
        system="You are a helpful coding assistant.",
        tools=["Read", "Write", "Bash"],
        cwd="/home/user/project",
    ) as chat:
        for msg in user_messages:
            async for chunk in chat.stream(msg):
                yield chunk  # SSE to browser


# Rate limits are handled transparently:
# - First tries claude-sonnet
# - If rate limited, backs off exponentially
# - If still failing, summarises conversation + cascades to codex
# - Caller never sees the cascade — just gets chunks
```

### Example 3: Cheap local classification (beezle-style)

```python
from daz_agent_sdk import agent, Tier
from pydantic import BaseModel


class AlertClassification(BaseModel):
    severity: str        # critical, high, medium, low
    should_notify: bool
    summary: str


async def classify_alert(alert_text: str) -> AlertClassification:
    result = await agent.ask(
        f"Classify this alert:\n{alert_text}",
        schema=AlertClassification,
        tier=Tier.FREE_FAST,  # ollama — free, fast, no rate limits
    )
    return result.parsed
```

### Example 4: Multi-modal pipeline (auto-blog-style)

```python
from daz_agent_sdk import agent, Tier


async def create_podcast_episode(topic: str):
    # Research and write script (high quality)
    script = await agent.ask(
        f"Write a 5-minute podcast script about: {topic}",
        tier=Tier.HIGH,
    )

    # Generate cover art
    cover = await agent.image(
        f"Podcast cover art for episode about {topic}",
        width=1024, height=1024,
        output=f"episodes/{topic}/cover.jpg",
    )

    # Narrate
    audio = await agent.speak(
        script.text,
        voice="gary",
        output=f"episodes/{topic}/episode.mp3",
    )

    return {"script": script.text, "cover": cover.path, "audio": audio.path}
```

### Example 5: Enumerating models

```python
from daz_agent_sdk import agent, Tier, Capability

# What's available right now?
all_models = await agent.models()
for m in all_models:
    print(f"{m.provider}:{m.model_id} — {m.display_name} [{m.tier.value}]")

# Just local models
local = await agent.models(provider="ollama")

# What can do structured output at high tier?
high_structured = await agent.models(tier=Tier.HIGH, capability=Capability.STRUCTURED)
```

---

## What Existing Projects Would Change

Migration is mechanical. Before:

```python
# OLD: 15 lines of boilerplate per query
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

async def ask_claude(prompt):
    response_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(allowed_tools=[], permission_mode="bypassPermissions")
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
    return response_text.strip()
```

After:

```python
# NEW: 1 line
from daz_agent_sdk import agent
answer = await agent.ask(prompt)
```

Before (image generation):

```python
# OLD: subprocess with 8 arguments
subprocess.run([
    "generate_image", "--prompt", prompt,
    "--width", "1024", "--height", "1024",
    "--output", str(path)
], check=True, timeout=300)
```

After:

```python
# NEW: 1 line
result = await agent.image(prompt, width=1024, height=1024, output=path)
```

---

## Dependency Strategy

**Core (always installed):**
- `pyyaml` — config parsing
- `pydantic` — structured output schemas
- `aiohttp` — HTTP client for Ollama/vLLM

**Optional (installed on first use or via extras):**
- `claude-agent-sdk` — Claude provider (`pip install daz-agent-sdk[claude]`)
- `openai-codex-sdk` — Codex provider (`pip install daz-agent-sdk[codex]`)
- `google-genai` — Gemini provider (`pip install daz-agent-sdk[gemini]`)

**Not dependencies (external tools, already installed):**
- `~/bin/generate_image` — mflux image generation
- `~/bin/tts` — Qwen3-TTS
- `~/bin/remove-background` — BiRefNet

A provider that isn't installed simply reports `available() → False` and gets
skipped in the fallback chain. No import errors, no crashes.

---

## Key Design Decisions

1. **Tier-based, not provider-based.** Callers say `Tier.HIGH` and the config
   decides that means claude-opus today but could mean gemini-2.5-pro tomorrow.

2. **Conversation is the primitive.** Even `agent.ask()` creates a 1-turn
   conversation internally. Everything flows through the same fallback engine.

3. **Config is optional.** Zero config works with sensible defaults (Claude as
   primary for all tiers if available). The config file only exists to override.

4. **Structured output is native.** Pass a Pydantic model, get a Pydantic model
   back. The library handles JSON extraction, markdown stripping, and validation
   across all providers.

5. **Image/TTS/STT are peers, not afterthoughts.** Same `agent.*` namespace,
   same logging, same error handling. A podcast pipeline reads identically to
   a text analysis pipeline.

6. **Logging is automatic and structured.** Every conversation gets a UUID
   directory with JSONL events. Debug any failure by reading the log.

7. **Fallback is the #1 feature.** Rate limit on Claude? Codex picks up
   instantly for single-shot. Conversations get backoff + summarise + cascade.
   The caller never writes retry logic.

8. **Provider-specific features degrade gracefully.** Claude tools, Codex
   threads, Gemini function calling — each provider exposes what it can.
   The library normalises the interface. Unsupported features on a fallback
   provider get best-effort handling or clear errors.
