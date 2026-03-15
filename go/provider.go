package dazagentsdk

import "context"

// CompleteOpts holds optional parameters for Provider.Complete.
type CompleteOpts struct {
	// Schema is the JSON schema for structured output (nil for plain text).
	Schema any
	// Timeout in seconds. Zero means use provider default.
	Timeout float64
	// Tools is a list of tool names for agentic providers.
	Tools []string
	// Cwd is the working directory for agentic providers.
	Cwd string
	// MaxTurns is the maximum number of agentic turns.
	MaxTurns int
}

// StreamOpts holds optional parameters for Provider.Stream.
type StreamOpts struct {
	// Timeout in seconds. Zero means use provider default.
	Timeout float64
}

// StreamChunk carries a single piece of streamed text or an error.
type StreamChunk struct {
	Text string
	Err  error
}

// Provider is the interface that all AI providers implement.
type Provider interface {
	// Name returns the provider identifier (e.g. "ollama", "claude").
	Name() string

	// Available checks if the provider is reachable and ready to serve.
	Available(ctx context.Context) (bool, error)

	// ListModels returns all models this provider currently offers.
	ListModels(ctx context.Context) ([]ModelInfo, error)

	// Complete sends messages and returns a full response.
	Complete(ctx context.Context, messages []Message, model ModelInfo, opts CompleteOpts) (*Response, error)

	// Stream sends messages and returns a channel of text chunks.
	Stream(ctx context.Context, messages []Message, model ModelInfo, opts StreamOpts) (<-chan StreamChunk, error)
}
