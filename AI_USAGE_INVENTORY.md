# AI Tool Usage Inventory

Comprehensive analysis of all AI tool usage across ~/local, ~/bin, and ~/src.
Goal: inform the design of a unified AI library that all projects can share.

---

## Part 1: Text Generation (LLM)

### 1.1 Claude Agent SDK Projects

Every project below uses `claude_agent_sdk` with async/await and streaming (`async for message in query(...)`).

| Project | Path | Models | Turns | Tools | Output Type | Use Case |
|---------|------|--------|-------|-------|-------------|----------|
| **temporal_assistant** | ~/local/temporal_assistant | default | single | Read,Write,Edit,Bash,Glob,Grep,WebFetch,WebSearch | plain text | Discord message handling |
| **claude_server** | ~/local/claude_server | default | conversation (50-turn reset) | File/Bash (bypassPermissions) | streaming SSE | General assistant HTTP API |
| **n8n_server** | ~/local/n8n_server | default | conversation (50-turn reset) | n8n MCP | streaming SSE | n8n workflow interaction |
| **arsis** | ~/local/arsis | default | conversation (context mgr) | implicit | plain text | Autonomous code agent |
| **auto-blog** | ~/src/auto-blog | haiku (summarize), opus (write) | multi (20 turns read, 5 write) | file reading | markdown + JSON | Blog/podcast generation |
| **chatty** | ~/src/chatty | default | single (max_turns=1) | none (bypassPermissions) | plain text (1-2 sentences) | Conversational companion |
| **beezle** | ~/src/beezle | haiku (fast), opus (complex) | single per query | varies | plain text + JSON | Health monitoring, code improvement |
| **auto-gui** | ~/src/auto-gui | default | single per step | WebFetch | text prompts for image gen | App summaries, icon prompts |
| **book-reader** | ~/src/book-reader | haiku (chapters), sonnet (dedup) | single per chapter | none | structured JSON | Character extraction, voice assignment |
| **noveliser2** | ~/src/noveliser2 | opus (structure), haiku (prose) | single per step | none | Pydantic structured + markdown | Full novel generation |
| **auto-dev** | ~/src/auto-dev | sonnet (configurable) | multi-turn agentic | full (git, code, browser) | tool calls + text | Development automation |
| **ai-chat** | ~/src/ai-chat | configurable | conversation | configurable | plain text | Interactive CLI chat |
| **resume-tailor** | ~/src/resume-tailor | opus | single | none | plain text | Resume customization |
| **dazflow2** | ~/src/dazflow2 | default | single | none | plain text | Workflow node integration |
| **portfolio** | ~/src/portfolio | default | single | none | plain text | Podcast title generation |

**Common SDK Patterns:**
- Async streaming: `async for message in query(prompt, options)`
- Message extraction: `message.content[].text` for `AssistantMessage`
- Options: `ClaudeAgentOptions(cwd, permission_mode, allowed_tools, model, max_turns, mcp_servers)`
- Thread safety: Flask servers use dedicated event loop thread + query_lock
- CLAUDECODE workaround: `os.environ.pop("CLAUDECODE", None)` before SDK calls
- Memoization: temporal_assistant and noveliser2 cache responses by SHA256 hash

**SDK Usage Modes:**
1. **Single-shot query()** — most projects (auto-blog, beezle, book-reader, noveliser2, auto-gui)
2. **Persistent ClaudeSDKClient** — claude_server, n8n_server, arsis, auto-dev
3. **Structured output** — noveliser2 (Pydantic + JSON schema appended to prompt), book-reader (JSON)

### 1.2 Ollama (Local LLM)

| Project | Path | Model | Interface | Sync/Async | Streaming | Output Type | Use Case |
|---------|------|-------|-----------|-----------|-----------|-------------|----------|
| **arsis** | ~/local/arsis | gpt-oss:20b-16384 | HTTP POST localhost:11434 | sync | no | JSON (structured) | Task planning |
| **temporal_assistant** | ~/local/temporal_assistant | ollama:qwen3-30b-32k | dazllm library | sync | no | Pydantic structured | News filtering, fact extraction |

**Ollama Patterns:**
- Direct HTTP: `POST http://localhost:11434/api/chat` with `{"model": ..., "messages": ..., "stream": false}`
- dazllm wrapper: `Llm.model_named("ollama:qwen3-30b-32k")` → `llm.chat_structured(messages, schema)`
- Both use structured/JSON output with retry logic

### 1.3 vLLM (Local LLM Server)

| Project | Path | Interface | Use Case |
|---------|------|-----------|----------|
| **auto-blog** | ~/src/auto-blog | HTTP POST to local vLLM API | Lightweight chunk summarization (temp 0.3) |

### 1.4 MLX-LM (On-Device)

| Project | Path | Interface | Use Case |
|---------|------|-----------|----------|
| **chatty** | ~/src/chatty | Python library import | Fallback local conversation model |

### 1.5 Gemini CLI

| Project | Path | Interface | Status |
|---------|------|-----------|--------|
| **arsis** | ~/local/arsis | subprocess `gemini prompt "..."` | Available but inactive (fallback) |

### 1.6 dazllm Library

**Status:** Installed as dependency, source not in these directories.

**API Surface (inferred):**
```python
from dazllm.core import Llm, ModelType

llm = Llm.model_named("ollama:qwen3-30b-32k")  # singleton
response = llm.chat_structured(messages, schema)  # → Pydantic model
response = llm.chat(messages)                      # → plain text
Llm.chat_structured_static(prompt, schema, model_type=ModelType.LOCAL_LARGE)
```

---

## Part 2: Still-Image Generation and Editing

### 2.1 Required production path

Every still-image generation or edit uses `~/bin/generate_image`, which submits a durable Codex job to the single Mac mini `image_generation_service` (IGS). The SDK itself never selects or dispatches a backing image model. Every client is hard-pinned to `http://10.0.0.46:8830`; no caller, environment variable, host detection, or configuration can select another origin.

The only accepted SDK provider selector is omitted or `codex`. Spark, Arbiter, Flux, Z-Image, Ollama, Gemini/Nano Banana, direct OpenAI, and explicit image model or step pins fail closed. There is no image-provider fallback.

| Project | Path | Dimensions | Use Case |
|---------|------|-----------|----------|
| **auto-gui** | ~/src/auto-gui | 128x128 | App icons (+ background removal) |
| **billboard** | ~/src/billboard | 1024x1024 | Playlist cover art |
| **noveliser2** | ~/src/noveliser2 | 768x1024, 1200x400 | Book covers, chapter illustrations |
| **auto-blog** | ~/src/auto-blog | varies | Podcast speaker portraits |
| **db-app** | ~/src/db-app | 64x64, 512x512 | Project thumbnails |
| **app-publish** | ~/src/app-publish | 1024x1024 | iOS app icons (8 resize variants) |
| **auto-dev** | ~/src/auto-dev | 1920x1080, 512x512 | Hero images, project icons |
| **portfolio** | ~/src/portfolio | 512x512 | Project thumbnails (fallback) |

**Invocation:** Use `~/bin/generate_image` for user-facing image work, or the SDK's stable image capability when integrating this library. Both routes terminate at the same Mac mini IGS/Codex service.

### 2.2 Background Removal (BiRefNet)

Tool: `~/bin/remove-background` → `/Volumes/T9/darrenoakey/src/background-removal/`

Used by: auto-gui and auto-blog. Background removal is permitted as non-generative processing of an existing image; it is not an alternative still-image generator.

### 2.3 Retired direct backends

Older inventory snapshots found local diffusion, direct OpenAI, Gemini, and multi-backend image implementations. They are not valid integration guidance. New and migrated callers must use Mac mini IGS/Codex only, and the SDK must reject attempts to pin those retired providers or models.

---

## Part 3: Text-to-Speech (TTS)

### 3.1 Qwen3-TTS (MLX Audio) — Primary Production Engine

Tool: `~/bin/tts` → `/Volumes/T9/darrenoakey/src/tts/`

| Project | Path | Mode | Use Case |
|---------|------|------|----------|
| **auto-blog** | ~/src/auto-blog | single + multi-speaker (JSONL) | Podcast narration |
| **book-reader** | ~/src/book-reader | multi-speaker (JSONL) | Audiobook narration |
| **beezle** | ~/src/beezle | single speaker | Alert audio notifications |

**Invocation:** Subprocess: `subprocess.run(["tts", "tts", "--text", ..., "--voice", ..., "--output", ...])`

**Features:** 9 voices, custom voice registry, voice cloning, multi-speaker dialogue, resume capability, speed control (0.5-2.0x)

### 3.2 Chatterbox TTS

Path: `~/src/chatterbox` — standalone tool, parallel chunk processing with ThreadPoolExecutor. 23+ languages, voice cloning.

### 3.3 MLX Whisper (STT) + Kokoro TTS

Used by: **chatty** (~/src/chatty)
- STT: `mlx_whisper` library, lazy-loaded, GPU lock for thread safety
- TTS: `mlx_audio.tts.generate`, Kokoro voice synthesis

### 3.4 macOS Native (SFSpeechRecognizer + NSSpeechSynthesizer)

Used by: **low-latency-voice** (~/src/low-latency-voice) — Swift implementation, streaming word-by-word STT, hardware echo cancellation

### 3.5 Research/Experimental TTS

**huanyuan-play** (~/src/huanyuan-play): Coqui, Tortoise (Docker), StyleTTS2, F5-TTS, Bark — multi-engine comparison

---

## Part 4: Speech-to-Text (STT)

| Project | Tool | Interface |
|---------|------|-----------|
| **chatty** | MLX Whisper | Python library (lazy-loaded, 3 model sizes) |
| **low-latency-voice** | SFSpeechRecognizer | macOS native (Swift, on-device) |

---

## Part 5: Capability Matrix

### What the unified library needs to expose:

| Capability | Current Tools | Projects Using | Interface Style |
|-----------|--------------|----------------|----------------|
| **LLM text gen (cloud)** | Claude Agent SDK | 15 projects | async streaming |
| **LLM text gen (local)** | Ollama, vLLM, MLX-LM | 4 projects | sync HTTP / library |
| **LLM structured output** | Claude SDK + JSON parse, dazllm | 5 projects | Pydantic schema → JSON |
| **LLM conversation** | Claude SDK (ClaudeSDKClient) | 5 projects | async context manager |
| **LLM single-shot** | Claude SDK (query()) | 10 projects | async generator |
| **LLM with tools** | Claude SDK + allowed_tools | 6 projects | tool list config |
| **Still-image generation/editing** | Mac mini IGS/Codex | 10 projects | durable job API via `generate_image` or SDK |
| **Background removal** | BiRefNet | 3 projects | subprocess |
| **TTS** | Qwen3-TTS, Chatterbox, Kokoro | 6 projects | subprocess / library |
| **STT** | Whisper, SFSpeechRecognizer | 2 projects | library / native |
| **Response caching** | Custom per-project | 4 projects | SHA256 hash → JSON file |

### Recurring Pain Points (to solve in unified library):

1. **Every project re-implements message extraction** — parsing `AssistantMessage.content[].text`
2. **Every project re-implements async-to-sync bridging** — event loop threads, locks, queues
3. **Caching is duplicated** — temporal_assistant, noveliser2, beezle all have their own cache
4. **CLAUDECODE env stripping** — needed in every project running inside Claude Code
5. **Model selection is hardcoded** — each project picks its own model string
6. **Image routing must remain singular** — all callers converge on Mac mini IGS/Codex; no provider-specific fallback
7. **TTS is always subprocess** — no Python API, just shell out to ~/bin/tts
8. **No unified error handling** — each project handles timeouts, rate limits differently
9. **No unified budget/cost tracking** — only beezle tracks costs
10. **Structured output parsing** — each project appends JSON schema to prompts manually

---

## Part 6: Project Quick Reference

### ~/local projects:
- `temporal_assistant` — Claude SDK (streaming, memoized) + dazllm/Ollama (structured)
- `claude_server` — Claude SDK (persistent client, SSE streaming, Flask)
- `n8n_server` — Claude SDK (persistent client, MCP, Flask)
- `arsis` — Claude SDK (conversation) + Ollama (JSON planning) + Gemini CLI (fallback)

### ~/src projects (AI-heavy):
- `auto-blog` — Claude SDK (haiku+opus) + vLLM + TTS
- `chatty` — Claude SDK + MLX-LM + Whisper STT + Kokoro TTS
- `beezle` — Claude SDK (multi-model, budget tracking) + TTS
- `auto-gui` — Claude SDK + image gen + background removal
- `book-reader` — Claude SDK (haiku+sonnet) + voice description + TTS
- `noveliser2` — Claude SDK (opus+haiku, structured/Pydantic, cached) + image gen
- `auto-dev` — Claude SDK (persistent client, full agentic, multi-turn)
- `ai-chat` — Claude SDK (configurable conversation CLI)
- `resume-tailor` — Claude SDK (single-shot)
- `dazflow2` — Claude SDK (workflow node, thread isolation)
- `portfolio` — Claude SDK (minimal, title gen)

### ~/src projects (media):
- `tts` — Qwen3-TTS engine (MLX Audio)
- `chatterbox` — Chatterbox TTS (HuggingFace/PyTorch)
- `low-latency-voice` — macOS native STT/TTS + Claude
- `background-removal` — BiRefNet background removal
- Mac mini `image_generation_service` — sole still-image generation/editing backend, using logged-in Codex
- `huanyuan-play` — TTS research (multi-engine)
- `ltx2` — Video generation with audio conditioning

### ~/bin tools:
- `generate_image` → durable Mac mini IGS/Codex job
- `remove-background` → BiRefNet
- `tts` → Qwen3-TTS
- `tts-on-google` → Google Cloud TTS
- `gemini` → Google Gemini CLI
