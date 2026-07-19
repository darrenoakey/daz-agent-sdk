# daz-agent-sdk (Go)

Go port of the daz-agent-sdk — provider-routed text, durable Codex image jobs, TTS, and STT.

Shares the same `~/.daz-agent-sdk/config.yaml` as the Python version.

## Install

```bash
go get github.com/darrenoakey/daz-agent-sdk/go
```

## Quick Start

### Single-Turn Ask

```go
package main

import (
    "context"
    "fmt"
    "log"

    sdk "github.com/darrenoakey/daz-agent-sdk/go"
    "github.com/darrenoakey/daz-agent-sdk/go/provider"
)

func main() {
    // Register Ollama provider
    sdk.RegisterProviderFactory("ollama", func(cfg *sdk.Config) sdk.Provider {
        return provider.NewOllamaProvider("http://localhost:11434")
    })

    agent := sdk.NewAgent(nil)
    ctx := context.Background()

    resp, err := agent.Ask(ctx, "What is the capital of France?")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(resp.Text)
}
```

### With Tier Selection

```go
resp, err := agent.Ask(ctx, "Explain quantum computing",
    sdk.WithAskTier(sdk.TierMedium),
    sdk.WithAskSystem("You are a physics teacher."),
)
```

### Multi-Turn Conversation

```go
conv := agent.Conversation("my-chat",
    sdk.WithTier(sdk.TierHigh),
    sdk.WithSystem("You are a helpful assistant."),
)
defer conv.Close()

resp1, _ := conv.Say(ctx, "What is Go?")
fmt.Println(resp1.Text)

resp2, _ := conv.Say(ctx, "How does it compare to Rust?")
fmt.Println(resp2.Text)

// Fork creates an independent copy
fork := conv.Fork("tangent")
defer fork.Close()
resp3, _ := fork.Say(ctx, "Tell me more about memory safety")
```

### Streaming

```go
chunks, err := conv.Stream(ctx, "Write a haiku about Go")
if err != nil {
    log.Fatal(err)
}
for chunk := range chunks {
    if chunk.Err != nil {
        log.Fatal(chunk.Err)
    }
    fmt.Print(chunk.Text)
}
fmt.Println()
```

### Image Generation

Uses only the durable Mac mini Codex image service at `http://10.0.0.46:8830`. The origin is compiled in and cannot be supplied by a caller, environment variable, host detection, or configuration. Requests use the signed system `/usr/bin/curl` transport to avoid per-binary macOS Local Network denial without introducing another route. Legacy provider, model, and step pins fail closed.

```go
import "github.com/darrenoakey/daz-agent-sdk/go/capability"

// Wire up the image capability
agent.ImageFn = func(ctx context.Context, prompt string, opts sdk.ImageCallOpts) (*sdk.ImageResult, error) {
    return capability.GenerateImage(ctx, prompt, capability.ImageOpts{
        Width:  opts.Width,
        Height: opts.Height,
        Output: opts.Output,
        Config: opts.Config,
    })
}

result, err := agent.Image(ctx, "a red apple on a white table", sdk.ImageOpts{
    Width:  512,
    Height: 512,
    Output: "/tmp/apple.png",
})
fmt.Println("Image saved to:", result.Path)
fmt.Println("Durable job:", result.JobID, result.Status, result.Ready)
fmt.Println("Submission identity:", result.IdempotencyKey, result.Replayed)
```

When the local wait expires, `Ready` is false and the durable `JobID` remains recoverable. No alternate image provider is attempted.

`GenerateImage` creates one UUID key when omitted and reuses the exact encoded body and key for bounded transport retries. Direct asynchronous submission uses `capability.SubmitImageJob` and requires `ImageOpts.IdempotencyKey`. Structured `409` and `410` failures retain the original job ID and key in `AgentError.Attempts`.

```go
status, err := capability.GetImageJob(ctx, result.JobID)
result, err = capability.ResumeImageJob(ctx, status.JobID, capability.ImageJobOpts{
    Output: "/tmp/apple.png",
    Timeout: 10 * time.Minute,
})
// For an already-complete job:
result, err = capability.DownloadImageJob(ctx, status.JobID, "/tmp/apple.png", false)
```

These recovery APIs issue only `GET /jobs/{id}` and `GET /jobs/{id}/image`; they preserve service provider and attempt provenance and never submit or fall back.

For crash-safe CLI submission and recovery, use `--state FILE`, optional `--idempotency-key KEY`, and `--recover FILE`. The versioned `0600` state is atomically synced before POST and records the job ID before polling.

### Text-to-Speech

Shells out to the `tts` CLI tool.

```go
agent.SpeakFn = func(ctx context.Context, text string, opts sdk.SpeakCallOpts) (*sdk.AudioResult, error) {
    return capability.SynthesizeSpeech(ctx, text, capability.SpeakOpts{
        Voice:  opts.Voice,
        Output: opts.Output,
        Speed:  opts.Speed,
    })
}

audio, err := agent.Speak(ctx, "Hello, world!", sdk.SpeakOpts{
    Voice:  "aiden",
    Output: "/tmp/hello.wav",
})
```

### Speech-to-Text

Shells out to the `whisper` CLI tool.

```go
agent.TranscribeFn = func(ctx context.Context, path string, opts sdk.TranscribeCallOpts) (string, error) {
    return capability.Transcribe(ctx, path, capability.TranscribeOpts{
        ModelSize: opts.ModelSize,
        Language:  opts.Language,
    })
}

text, err := agent.Transcribe(ctx, "/tmp/recording.wav", sdk.TranscribeOpts{
    ModelSize: "small",
})
```

### Structured Output

```go
type Answer struct {
    Capital string `json:"capital"`
    Country string `json:"country"`
}

resp, _ := agent.Ask(ctx, "What is the capital of France? Reply as JSON.",
    sdk.WithAskSystem(sdk.SchemaInstructions(`{"type":"object","properties":{"capital":{"type":"string"},"country":{"type":"string"}}}`)),
)

var answer Answer
if err := sdk.ExtractStructured(resp, &answer); err != nil {
    log.Fatal(err)
}
fmt.Printf("%s is the capital of %s\n", answer.Capital, answer.Country)
```

### List Models

```go
models, _ := agent.Models(ctx)
for _, m := range models {
    fmt.Printf("%s  %s  [%s]\n", m.QualifiedName(), m.DisplayName, m.Tier)
}

// Filter by tier
models, _ = agent.Models(ctx, sdk.WithModelsTier(sdk.TierLow))
```

## CLI

Build and install:

```bash
cd go/cmd/agent-sdk
go build -o agent-sdk .
```

Usage:

```bash
# Ask a question
agent-sdk ask "What is Go?" --tier medium

# Generate an image
agent-sdk image --prompt "a sunset over mountains" --width 1024 --height 768 --output sunset.png

# List available models
agent-sdk models
agent-sdk models --tier high
```

## Architecture

```
go/
  types.go          — Tier, Capability, ErrorKind, ModelInfo, Message, Response, AgentError
  config.go         — YAML config loader (reads ~/.daz-agent-sdk/config.yaml)
  provider.go       — Provider interface, CompleteOpts, StreamOpts, StreamChunk
  registry.go       — Provider registry, model resolution, tier chain mapping
  fallback.go       — Error classification + cascade/backoff fallback engine
  logging.go        — Per-conversation UUID JSONL event logger
  conversation.go   — Multi-turn Conversation with Say, Stream, Fork, History
  agent.go          — Agent: Ask, Conversation, Image, Speak, Transcribe, Models
  structured.go     — JSON extraction from LLM responses, schema instructions
  provider/
    ollama.go       — Ollama HTTP text provider
  capability/
    image.go        — Codex image-service routing and validation
    image_codex.go  — durable `/jobs` service client
    tts.go          — Text-to-speech via tts CLI subprocess
    stt.go          — Speech-to-text via whisper CLI subprocess
  cmd/
    agent-sdk/
      main.go       — CLI entry point
```

## Tier System

Models are organized into tiers, each with a fallback chain:

| Tier | Models |
|------|--------|
| `very_high` | claude-opus-4-6, gpt-5.6-sol, gemini-2.5-pro |
| `high` | claude-opus-4-6, gpt-5.6-sol, gemini-2.5-pro |
| `medium` | claude-sonnet-4-6, gpt-5.6-sol, gemini-2.5-flash |
| `low` | claude-haiku-4-5, gemini-2.5-flash-lite, qwen3-8b |
| `free_fast` | qwen3-8b (Ollama) |
| `free_thinking` | qwen3-30b-32k, deepseek-r1:14b (Ollama) |

When a provider fails, the fallback engine automatically tries the next one in the chain. Rate limits, timeouts, and unavailable providers cascade; auth and invalid request errors stop immediately.

## Configuration

Reads `~/.daz-agent-sdk/config.yaml` (same file as the Python version):

```yaml
tiers:
  high:
    - claude:claude-opus-4-6
    - codex:gpt-5.6-sol
    - gemini:gemini-2.5-pro
  low:
    - ollama:qwen3-8b

providers:
  ollama:
    base_url: http://localhost:11434
```

There is no image backend configuration. Legacy `image.model`, `image.tiers`, and `image.fallback` values may still parse for compatibility but never affect dispatch.

## Comparison with Python SDK

| Feature | Python | Go |
|---------|--------|----|
| Text generation | All providers | Ollama (others coming) |
| Image generation | Mac mini Codex durable jobs | Mac mini Codex durable jobs |
| TTS | `tts` subprocess | `tts` subprocess |
| STT | `whisper` subprocess | `whisper` subprocess |
| Background removal | BiRefNet (inline) | Not yet supported |
| Structured output | Pydantic schemas | JSON struct tags |
| Config | Same YAML | Same YAML |
| Streaming | async generators | Go channels |
| Fallback | async cascade | goroutine cascade |

## Dependencies

- `gopkg.in/yaml.v3` — YAML config parsing
- `github.com/google/uuid` — UUID generation
- Standard library for everything else (net/http, os/exec, encoding/json)

## Tests

```bash
# Quick tests (skips slow integration tests)
go test ./... -short

# Full tests (live text-provider tests may require their services)
go test ./... -v

# Vet
go vet ./...
```
