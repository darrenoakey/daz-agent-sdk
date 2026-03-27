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

## Capabilities

| Capability | Module |
|------------|--------|
| Image | `capabilities/image.py` — Spark (default, CUDA FLUX.1-schnell on spark:8100), mflux Z-Image-Turbo (local Apple Silicon), Nano Banana 2 (Gemini API), inline BiRefNet background removal |
| TTS | `capabilities/tts.py` — subprocess to `tts` |
| STT | `capabilities/stt.py` — subprocess to `whisper` |

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
- Image generation: `provider=None` (default) uses Spark (CUDA FLUX.1-schnell at spark:8100), `provider="mflux"` uses local Apple Silicon, `provider="nano-banana-2"` uses Gemini API. Configurable fallback chain via `image.fallback` in config.yaml — on primary failure, tries each fallback in order
- Image fallback refactor: `_generate_one()` dispatches per-provider, `generate_image()` loops `[primary] + fallbacks`. Each provider handles its own transparency
- Spark image server runs on spark (GB10 CUDA) via arbiter (`spark:8400` for jobs, `spark:8100` for direct image gen)
- Background removal (`transparent=True`): spark provider submits `background-remove` job to arbiter (BiRefNet on GPU); mflux provider runs BiRefNet locally (CPU); arbiter returns result in `result.data` (not `result.image`)
- Local BiRefNet fallback requires `pip install "daz-agent-sdk[transparent]"` (torch, torchvision, transformers, einops, kornia, timm)
- BiRefNet model cached as module-level singleton (`_birefnet_model`), loaded with `.float()` for CPU compatibility
- `./run deploy` bumps patch version, builds, uploads to PyPI via twine (keyring token), waits 30s, installs globally
- Claude provider uses native `output_format` for structured output — SDK injects a `StructuredOutput` tool, data comes back in `ToolUseBlock(name='StructuredOutput', input={...})`. Falls back to parsing response text if native output unavailable
- Claude provider `_collect_response` returns `(text, structured_output)` tuple — captures `ResultMessage.result`, `ResultMessage.structured_output`, and `StructuredOutput` tool calls
- mflux Z-Image-Turbo uses `ZImageTurbo` from `mflux.models.z_image` (NOT `Flux1` — that's for FLUX.1 schnell/dev only). `generate_image()` returns `GeneratedImage` (not PIL), save with `.save(path=..., overwrite=True)`
- Exclude `image_test.py` from normal test runs (`--ignore`) — mflux test downloads multi-GB model
- Codex JSONL events: `item.completed` with `item.type=="agent_message"` carries response text; `turn.failed` and `error` events carry error messages; pipe prompt via stdin with `codex exec - --json -m MODEL -s read-only --ephemeral`
- Gemini CLI JSON: `-o json` returns `{"response": "text", "stats": {...}}`; `-o stream-json` emits JSONL with `type=message role=assistant` for chunks and `type=result` for completion
- Codex models with ChatGPT auth: `gpt-5.3-codex` works, `o4-mini` and `gpt-4.1-nano` do NOT; `gpt-4.1` works
- mflux package lacks `py.typed` marker — pyright can't resolve imports statically; use `# pyright: ignore[reportMissingImports]` on mflux import lines
- mflux generation uses system-wide `fcntl.flock` at `/tmp/daz-agent-sdk-mflux.lock` — only one generation across all processes at a time
- Package ships `py.typed` (PEP 561) — callers get full type info; `pyproject.toml` has `[tool.setuptools.package-data]` to include it in wheel
- `./run deploy` handles version bump automatically — don't bump version manually before running it

## Go Port

Full Go implementation in `go/` directory. See `go/README.md` for complete documentation.

- **Run tests**: `cd go && go test ./... -short`
- **Vet**: `cd go && go vet ./...`
- **Build CLI**: `cd go/cmd/agent-sdk && go build -o agent-sdk .`
- **Module**: `github.com/darrenoakey/daz-agent-sdk/go`
- Image gen uses Ollama HTTP (`x/z-image-turbo`) instead of mflux — pure Go, no Python deps
- TTS/STT use same CLI subprocesses as Python (tts, whisper)
- Provider interface in root package, implementations in `provider/` subpackage
- Capabilities in `capability/` subpackage wired to Agent via function fields (avoids circular imports)
- Claude provider uses `anthropics/anthropic-sdk-go` — native structured output via `JSONOutputFormatParam`
- OpenAI provider uses `openai/openai-go` — native structured output via `ResponseFormatJSONSchemaParam`
- Gemini provider uses `google.golang.org/genai` — native structured output via `ResponseJsonSchema` + `ResponseMIMEType`
- Ollama provider uses raw HTTP (no heavy SDK dep) — structured output via `format` field in chat request
- Registration is done at call sites (main.go, tests), not via init()
- 184 tests, all passing
