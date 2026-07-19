# daz-agent-sdk

Provider-agnostic AI library with tier-based routing and automatic fallback.

## Quick Reference

- **Run tests**: `.venv/bin/python -m pytest src/ -v --tb=short`
- **Lint**: `.venv/bin/ruff check src/`
- **Type check**: `.venv/bin/pyright src/daz_agent_sdk/`
- **CLI**: `.venv/bin/python run --help`
- **All three gates must pass before commit**: pytest, ruff, pyright

## Architecture

- `types.py` — Tier, Capability, ErrorKind, ModelInfo, Message, Response, AgentError
- `config.py` — YAML config loader from `~/.daz-agent-sdk/config.yaml`, sensible defaults
- `registry.py` — lazy provider loading, model resolution, tier chain mapping
- `fallback.py` — error classification + single-shot cascade (used by `Agent.ask`, not conversations)
- `logging_.py` — per-conversation UUID JSONL event logger
- `conversation.py` — multi-turn Conversation class with history, fork, stream
- `core.py` — Agent singleton: ask, conversation, image, remove_background, speak, transcribe, models
- `main.py` — CLI entry point (ask, models, image subcommands)
- `__init__.py` — exports `agent` singleton + all public types

## Providers

| Provider | Module | Status |
|----------|--------|--------|
| Claude | `providers/claude.py` | Uses claude_agent_sdk, strips CLAUDECODE env |
| Codex | `providers/codex.py` | CLI subprocess (`codex exec --json`), ChatGPT auth |
| Gemini | `providers/gemini.py` | CLI subprocess (`gemini -p -o json`), Google auth |
| Ollama | `providers/ollama.py` | HTTP to localhost:11434 |
| Arbiter | `providers/arbiter.py` | OpenAI-compat HTTP to spark arbiter at `10.0.0.254:8400/v1/chat/completions`. Default for `low`/`free_fast`/`free_thinking` tiers → `arbiter:qwen3.6-27b`. Reasoning models (qwen3 via vLLM `--reasoning-parser qwen3`) populate `message.reasoning` when `message.content` is empty; the provider falls through to reasoning so `Response.text` is never blank. |

## Capabilities

| Capability | Module |
|------------|--------|
| Image | `capabilities/image.py` — durable Mac mini Codex image-service `/jobs` client; no image-provider fallback |
| TTS | `capabilities/tts.py` — subprocess to `tts` |
| STT | `capabilities/stt.py` — subprocess to `whisper` |

## Auth Policy — NO API KEYS

**CRITICAL**: This project does NOT use API keys for Claude, Codex, or Gemini text providers. All three use ambient/subscription auth:
- **Claude** — wraps `claude` CLI via `claude_agent_sdk` (Python) or subprocess (Go). Uses the logged-in Claude subscription. No `ANTHROPIC_API_KEY`.
- **Codex** — wraps `codex` CLI. Uses ChatGPT auth. No `OPENAI_API_KEY`.
- **Gemini text** — wraps `gemini` CLI. Uses Google auth. No `GEMINI_API_KEY` for text.
- **Ollama** — local HTTP, no auth at all.
- **Arbiter** — spark arbiter HTTP, no auth. The arbiter is the GPU job server on the spark machine (GB10 CUDA, 128 GB unified mem) at `10.0.0.254:8400`. Its `/v1/chat/completions` endpoint is OpenAI-compatible and proxies to vLLM-served LLMs (`qwen3.6-27b` default, `gemma4-31b`, `gemma4-26b`).

The Go OpenAI provider uses `OPENAI_API_KEY` because it calls the API directly (no codex CLI wrapper in Go). This is the one text provider exception in Go.

## Key Patterns

- Base `Provider.stream()` is NOT async — subclasses implement as async generators (which are AsyncIterator, not coroutines returning AsyncIterator)
- `fallback.py` uses `EventLogger` Protocol for logger param — allows both ConversationLogger and test fakes
- Conversation calls provider directly (no fallback chain) — conversations are tied to one provider's state; only `Agent.ask` uses `execute_with_fallback`
- Codex and Gemini providers use CLI subprocesses — no API keys needed, auth handled by CLI tools
- Test fakes for Provider must use async generators (with yield), not async functions returning iterators
- Config defaults have no vllm — removed by design
- Conversation accepts `mcp_servers` for MCP tool integration (passed through to Claude provider)
- Published as pip package: `pip install -e .` for local dev, `pyproject.toml` with setuptools
- `./run install` command installs editable package system-wide
- 15 projects converted from claude_agent_sdk/dazllm to daz-agent-sdk (see AI_USAGE_INVENTORY.md for full list)
- Image generation: `provider=None` / `provider="codex"` routes only to the always-on Mac mini image generation service at `http://10.0.0.46:8830`. The contract is `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/image`; callers cannot override the origin.
- The mac mini service is the shared image path for ltx2, `~/bin/generate_image`, and `~/publish`. It returns PNG bytes and handles source images through uploaded base64 `source_images`. JPEG callers are converted locally after the service returns PNG.
- Image provider/model/step pins for Spark, Arbiter, Flux, Z-Image, Ollama, Gemini/Nano Banana, and direct OpenAI fail closed. Image config fallback entries are parsed only for backward compatibility and never dispatched.
- `ImageResult.job_id`, `status`, and `ready` expose durable state. A local wait timeout returns `ready=False` without cancelling or replacing the server job.
- `get_image_job`, `resume_image_job`, and `download_image_job` recover existing IDs using GET-only IGS routes and retain provider/attempt provenance. Go exposes the equivalent `capability.GetImageJob`, `ResumeImageJob`, and `DownloadImageJob` APIs.
- `./run deploy` bumps patch version, builds, uploads to PyPI via twine (keyring token), waits 30s, installs globally
- Claude provider uses native `output_format` for structured output — SDK injects a `StructuredOutput` tool, data comes back in `ToolUseBlock(name='StructuredOutput', input={...})`. Falls back to parsing response text if native output unavailable
- Claude provider `_collect_response` returns `(text, structured_output)` tuple — captures `ResultMessage.result`, `ResultMessage.structured_output`, and `StructuredOutput` tool calls
- Codex JSONL events: `item.completed` with `item.type=="agent_message"` carries response text; `turn.failed` and `error` events carry error messages; pipe prompt via stdin with `codex exec - --json -m MODEL -s read-only --ephemeral`
- Gemini CLI JSON: `-o json` returns `{"response": "text", "stats": {...}}`; `-o stream-json` emits JSONL with `type=message role=assistant` for chunks and `type=result` for completion
- Codex models with ChatGPT auth: `gpt-5.3-codex` works, `o4-mini` and `gpt-4.1-nano` do NOT; `gpt-4.1` works
- Package ships `py.typed` (PEP 561) — callers get full type info; `pyproject.toml` has `[tool.setuptools.package-data]` to include it in wheel
- `./run deploy` handles version bump automatically — don't bump version manually before running it

## Go Port

Full Go implementation in `go/` directory. See `go/README.md` for complete documentation.

- **Run tests**: `cd go && go test ./... -short`
- **Vet**: `cd go && go vet ./...`
- **Build CLI**: `cd go/cmd/agent-sdk && go build -o agent-sdk .`
- **Module**: `github.com/darrenoakey/daz-agent-sdk/go`
- Image generation: durable Mac mini Codex `/jobs` client only, hard-pinned to `http://10.0.0.46:8830` with no origin override or provider fallback
- Legacy `ImageOpts.Provider`, `Model`, and `Steps` values fail closed; timeout results retain durable job state
- Conversation.Say() accepts `WithSaySchema()` and `WithSayTier()` for structured output and per-call tier override
- TTS/STT use same CLI subprocesses as Python (tts, whisper)
- Provider interface in root package, implementations in `provider/` subpackage
- Capabilities in `capability/` subpackage wired to Agent via function fields (avoids circular imports)
- Claude provider wraps `claude` CLI (ambient subscription login, no API key) — structured output via `--json-schema`
- OpenAI provider uses `openai/openai-go` — native structured output via `ResponseFormatJSONSchemaParam`
- Gemini provider uses `google.golang.org/genai` — native structured output via `ResponseJsonSchema` + `ResponseMIMEType`
- Ollama provider uses raw HTTP (no heavy SDK dep) — structured output via `format` field in chat request
- Registration is done at call sites (main.go, tests), not via init(); CLI registers all 4 providers
- 241 tests, all passing
