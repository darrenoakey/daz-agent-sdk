# agent-sdk

Provider-agnostic AI library with tier-based routing and automatic fallback.

## Quick Reference

- **Run tests**: `.venv/bin/python -m pytest src/ -v --tb=short`
- **Lint**: `.venv/bin/ruff check src/`
- **Type check**: `.venv/bin/pyright src/agent_sdk/`
- **CLI**: `.venv/bin/python run --help`
- **All three gates must pass before commit**: pytest, ruff, pyright

## Architecture

- `types.py` — Tier, Capability, ErrorKind, ModelInfo, Message, Response, AgentError
- `config.py` — YAML config loader from `~/.agent-sdk/config.yaml`, sensible defaults
- `registry.py` — lazy provider loading, model resolution, tier chain mapping
- `fallback.py` — error classification + single-shot cascade / conversation backoff+cascade
- `logging_.py` — per-conversation UUID JSONL event logger
- `conversation.py` — multi-turn Conversation class with history, fork, stream
- `core.py` — Agent singleton: ask, conversation, image, speak, transcribe, models
- `main.py` — CLI entry point (ask, models subcommands)
- `__init__.py` — exports `agent` singleton + all public types

## Providers

| Provider | Module | Status |
|----------|--------|--------|
| Claude | `providers/claude.py` | Uses claude_agent_sdk, strips CLAUDECODE env |
| Codex | `providers/codex.py` | OpenAI SDK, thread lifecycle |
| Gemini | `providers/gemini.py` | google-genai SDK |
| Ollama | `providers/ollama.py` | HTTP to localhost:11434 |

## Capabilities

| Capability | Module |
|------------|--------|
| Image | `capabilities/image.py` — subprocess to `generate_image` (--transparent flag for background removal) |
| TTS | `capabilities/tts.py` — subprocess to `tts` |
| STT | `capabilities/stt.py` — subprocess to `whisper` |

## Key Patterns

- Base `Provider.stream()` is NOT async — subclasses implement as async generators (which are AsyncIterator, not coroutines returning AsyncIterator)
- `fallback.py` uses `EventLogger` Protocol for logger param — allows both ConversationLogger and test fakes
- `google.genai` import needs `# type: ignore[attr-defined]` for pyright
- Test fakes for Provider must use async generators (with yield), not async functions returning iterators
- Config defaults have no vllm — removed by design
- Conversation accepts `mcp_servers` for MCP tool integration (passed through to Claude provider)
- Published as pip package: `pip install -e .` for local dev, `pyproject.toml` with setuptools
- `./run install` command installs editable package system-wide
- 15 projects converted from claude_agent_sdk/dazllm to agent-sdk (see AI_USAGE_INVENTORY.md for full list)
