package provider

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// openAIModels is the static catalog of supported OpenAI models.
var openAIModels = []sdk.ModelInfo{
	{
		Provider:             "openai",
		ModelID:              "gpt-4.1",
		DisplayName:          "GPT-4.1",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierHigh,
		SupportsStreaming:     true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "openai",
		ModelID:              "gpt-4.1-mini",
		DisplayName:          "GPT-4.1 Mini",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierMedium,
		SupportsStreaming:     true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "openai",
		ModelID:              "gpt-4.1-nano",
		DisplayName:          "GPT-4.1 Nano",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierLow,
		SupportsStreaming:     true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "openai",
		ModelID:              "o4-mini",
		DisplayName:          "O4 Mini",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierMedium,
		SupportsStreaming:     true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
}

// OpenAIProvider calls the OpenAI Chat Completions API using the official Go SDK.
// Authentication is via OPENAI_API_KEY environment variable (read automatically by the SDK).
type OpenAIProvider struct{}

// NewOpenAIProvider returns a new OpenAIProvider.
func NewOpenAIProvider() *OpenAIProvider {
	return &OpenAIProvider{}
}

// Name returns "openai".
func (o *OpenAIProvider) Name() string {
	return "openai"
}

// Available returns true when OPENAI_API_KEY is set and the API responds.
// Returns false (not an error) when the key is absent or the API is unreachable.
func (o *OpenAIProvider) Available(ctx context.Context) (bool, error) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		return false, nil
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	client := openai.NewClient(option.WithMaxRetries(0))
	_, err := client.Models.List(ctx)
	if err != nil {
		return false, nil
	}
	return true, nil
}

// ListModels returns the static OpenAI model catalog.
// No network call is made; the list is hard-coded to match the requirements.
func (o *OpenAIProvider) ListModels(_ context.Context) ([]sdk.ModelInfo, error) {
	result := make([]sdk.ModelInfo, len(openAIModels))
	copy(result, openAIModels)
	return result, nil
}

// buildOpenAIMessages converts sdk.Message slice into OpenAI ChatCompletionMessageParamUnion entries.
// System, user, and assistant roles all map directly to their OpenAI equivalents.
func buildOpenAIMessages(messages []sdk.Message) []openai.ChatCompletionMessageParamUnion {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, m := range messages {
		switch m.Role {
		case "system":
			result = append(result, openai.SystemMessage(m.Content))
		case "assistant":
			result = append(result, openai.AssistantMessage(m.Content))
		default:
			result = append(result, openai.UserMessage(m.Content))
		}
	}
	return result
}

// classifyOpenAIError maps an OpenAI SDK error to an ErrorKind.
// It first checks for the typed openai.Error to get the HTTP status code.
// For non-API errors it falls back to string matching for network/context errors.
func classifyOpenAIError(err error) sdk.ErrorKind {
	// Check for the typed API error first — avoids calling .Error() which can
	// panic when Request/Response are nil (e.g. in tests that construct bare structs).
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return classifyOpenAIStatusCode(apiErr.StatusCode)
	}

	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "timeout") || strings.Contains(msg, "deadline exceeded") {
		return sdk.ErrorTimeout
	}
	if strings.Contains(msg, "connection refused") || strings.Contains(msg, "no such host") {
		return sdk.ErrorNotAvailable
	}

	// Fall back: scan integer tokens in the error message for HTTP status codes.
	var code int
	for _, part := range strings.Fields(err.Error()) {
		if _, scanErr := fmt.Sscanf(part, "%d", &code); scanErr == nil && code >= 400 {
			return classifyOpenAIStatusCode(code)
		}
	}
	return sdk.ErrorInternal
}

// classifyOpenAIStatusCode maps HTTP status codes to ErrorKind values.
func classifyOpenAIStatusCode(statusCode int) sdk.ErrorKind {
	switch {
	case statusCode == 429:
		return sdk.ErrorRateLimit
	case statusCode == 401 || statusCode == 403:
		return sdk.ErrorAuth
	case statusCode == 408:
		return sdk.ErrorTimeout
	case statusCode == 400 || statusCode == 422:
		return sdk.ErrorInvalidRequest
	case statusCode == 503:
		return sdk.ErrorNotAvailable
	default:
		return sdk.ErrorInternal
	}
}

// Complete sends messages to the OpenAI API and returns a full response.
// When opts.Schema is non-nil it is sent as the JSON schema response format.
func (o *OpenAIProvider) Complete(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.CompleteOpts) (*sdk.Response, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	msgParams := buildOpenAIMessages(messages)
	if len(msgParams) == 0 {
		return nil, sdk.NewAgentError("no messages provided", sdk.ErrorInvalidRequest, nil)
	}

	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(model.ModelID),
		Messages: msgParams,
	}

	if opts.Schema != nil {
		schema, ok := opts.Schema.(map[string]any)
		if !ok {
			return nil, sdk.NewAgentError("schema must be map[string]any", sdk.ErrorInvalidRequest, nil)
		}
		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				Type: "json_schema",
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   "response",
					Schema: schema,
					Strict: param.NewOpt(true),
				},
			},
		}
	}

	client := openai.NewClient(option.WithMaxRetries(0))
	resp, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		kind := classifyOpenAIError(err)
		return nil, sdk.NewAgentError(fmt.Sprintf("openai complete error: %v", err), kind, nil)
	}

	text := ""
	if len(resp.Choices) > 0 {
		text = resp.Choices[0].Message.Content
	}

	return &sdk.Response{
		Text:           text,
		ModelUsed:      model,
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
		Usage: map[string]any{
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
			"total_tokens":      resp.Usage.TotalTokens,
		},
	}, nil
}

// Stream sends messages to the OpenAI API and returns a channel of text chunks.
func (o *OpenAIProvider) Stream(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.StreamOpts) (<-chan sdk.StreamChunk, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))

	msgParams := buildOpenAIMessages(messages)
	if len(msgParams) == 0 {
		cancel()
		return nil, sdk.NewAgentError("no messages provided", sdk.ErrorInvalidRequest, nil)
	}

	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(model.ModelID),
		Messages: msgParams,
	}

	client := openai.NewClient(option.WithMaxRetries(0))
	stream := client.Chat.Completions.NewStreaming(ctx, params)

	ch := make(chan sdk.StreamChunk)
	go func() {
		defer close(ch)
		defer cancel()
		defer stream.Close()

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) > 0 {
				text := chunk.Choices[0].Delta.Content
				if text != "" {
					ch <- sdk.StreamChunk{Text: text}
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- sdk.StreamChunk{Err: fmt.Errorf("openai stream error: %w", err)}
		}
	}()

	return ch, nil
}

// Compile-time check that OpenAIProvider satisfies Provider.
var _ sdk.Provider = (*OpenAIProvider)(nil)
