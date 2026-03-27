package provider

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/google/uuid"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// claudeModels is the static catalog of supported Claude models.
var claudeModels = []sdk.ModelInfo{
	{
		Provider:             "claude",
		ModelID:              "claude-opus-4-6",
		DisplayName:          "Claude Opus 4.6",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured, sdk.CapabilityAgentic},
		Tier:                 sdk.TierHigh,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "claude",
		ModelID:              "claude-sonnet-4-6",
		DisplayName:          "Claude Sonnet 4.6",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured, sdk.CapabilityAgentic},
		Tier:                 sdk.TierMedium,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "claude",
		ModelID:              "claude-haiku-4-5-20251001",
		DisplayName:          "Claude Haiku 4.5",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierLow,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
}

// ClaudeProvider calls the Anthropic API using the official Go SDK.
// Authentication is via ANTHROPIC_API_KEY environment variable.
type ClaudeProvider struct{}

// NewClaudeProvider returns a new ClaudeProvider.
func NewClaudeProvider() *ClaudeProvider {
	return &ClaudeProvider{}
}

// Name returns "claude".
func (c *ClaudeProvider) Name() string {
	return "claude"
}

// Available returns true when ANTHROPIC_API_KEY is set and the API responds.
// Returns false (not an error) when the key is absent.
func (c *ClaudeProvider) Available(ctx context.Context) (bool, error) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		return false, nil
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	client := anthropic.NewClient(option.WithMaxRetries(0))
	_, err := client.Models.List(ctx, anthropic.ModelListParams{})
	if err != nil {
		return false, nil
	}
	return true, nil
}

// ListModels returns the static Claude model catalog.
// No network call is made; the list is hard-coded to match the requirements.
func (c *ClaudeProvider) ListModels(_ context.Context) ([]sdk.ModelInfo, error) {
	result := make([]sdk.ModelInfo, len(claudeModels))
	copy(result, claudeModels)
	return result, nil
}

// buildAnthropicMessages converts sdk.Message slice into Anthropic API params.
// System messages are collected separately; multiple system messages are merged
// with newlines. Non-system messages become MessageParam entries.
func buildAnthropicMessages(messages []sdk.Message) ([]anthropic.TextBlockParam, []anthropic.MessageParam) {
	var systemParts []string
	var params []anthropic.MessageParam

	for _, m := range messages {
		if m.Role == "system" {
			systemParts = append(systemParts, m.Content)
			continue
		}
		block := anthropic.NewTextBlock(m.Content)
		var role anthropic.MessageParamRole
		if m.Role == "assistant" {
			role = anthropic.MessageParamRoleAssistant
		} else {
			role = anthropic.MessageParamRoleUser
		}
		params = append(params, anthropic.MessageParam{
			Role:    role,
			Content: []anthropic.ContentBlockParamUnion{block},
		})
	}

	var systemBlocks []anthropic.TextBlockParam
	if len(systemParts) > 0 {
		systemBlocks = []anthropic.TextBlockParam{
			{Text: strings.Join(systemParts, "\n")},
		}
	}
	return systemBlocks, params
}

// classifyAnthropicStatusCode maps HTTP status codes to ErrorKind values.
func classifyAnthropicStatusCode(statusCode int) sdk.ErrorKind {
	switch {
	case statusCode == http.StatusTooManyRequests:
		return sdk.ErrorRateLimit
	case statusCode == http.StatusUnauthorized || statusCode == http.StatusForbidden:
		return sdk.ErrorAuth
	case statusCode == http.StatusRequestTimeout:
		return sdk.ErrorTimeout
	case statusCode == http.StatusBadRequest || statusCode == http.StatusUnprocessableEntity:
		return sdk.ErrorInvalidRequest
	case statusCode == http.StatusServiceUnavailable:
		return sdk.ErrorNotAvailable
	default:
		return sdk.ErrorInternal
	}
}

// classifyAnthropicError maps an Anthropic SDK error to an ErrorKind.
func classifyAnthropicError(err error) sdk.ErrorKind {
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "timeout") || strings.Contains(msg, "deadline exceeded") {
		return sdk.ErrorTimeout
	}
	if strings.Contains(msg, "connection refused") || strings.Contains(msg, "no such host") {
		return sdk.ErrorNotAvailable
	}

	// Try to extract the HTTP status code from the apierror.Error struct.
	// The SDK error string contains the HTTP status line; parse it numerically.
	var code int
	for _, part := range strings.Fields(err.Error()) {
		if _, scanErr := fmt.Sscanf(part, "%d", &code); scanErr == nil && code >= 400 {
			return classifyAnthropicStatusCode(code)
		}
	}
	return sdk.ErrorInternal
}

// Complete sends messages to the Anthropic API and returns a full response.
// When opts.Schema is non-nil it is sent as the JSON-schema output format.
func (c *ClaudeProvider) Complete(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.CompleteOpts) (*sdk.Response, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	systemBlocks, msgParams := buildAnthropicMessages(messages)
	if len(msgParams) == 0 {
		return nil, sdk.NewAgentError("no user/assistant messages provided", sdk.ErrorInvalidRequest, nil)
	}

	params := anthropic.MessageNewParams{
		Model:     model.ModelID,
		MaxTokens: 16000,
		Messages:  msgParams,
		System:    systemBlocks,
	}

	if opts.Schema != nil {
		schema, ok := opts.Schema.(map[string]any)
		if !ok {
			return nil, sdk.NewAgentError("schema must be map[string]any", sdk.ErrorInvalidRequest, nil)
		}
		params.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{
				Schema: schema,
			},
		}
	}

	client := anthropic.NewClient(option.WithMaxRetries(0))
	resp, err := client.Messages.New(ctx, params)
	if err != nil {
		kind := classifyAnthropicError(err)
		return nil, sdk.NewAgentError(fmt.Sprintf("anthropic complete error: %v", err), kind, nil)
	}

	text := extractTextFromContent(resp.Content)

	return &sdk.Response{
		Text:           text,
		ModelUsed:      model,
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
		Usage: map[string]any{
			"input_tokens":  resp.Usage.InputTokens,
			"output_tokens": resp.Usage.OutputTokens,
		},
	}, nil
}

// extractTextFromContent joins all text blocks from the response content.
func extractTextFromContent(blocks []anthropic.ContentBlockUnion) string {
	var parts []string
	for _, block := range blocks {
		if block.Type == "text" {
			text := block.AsText()
			if text.Text != "" {
				parts = append(parts, text.Text)
			}
		}
	}
	return strings.Join(parts, "")
}

// Stream sends messages to the Anthropic API and returns a channel of text chunks.
func (c *ClaudeProvider) Stream(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.StreamOpts) (<-chan sdk.StreamChunk, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))

	systemBlocks, msgParams := buildAnthropicMessages(messages)
	if len(msgParams) == 0 {
		cancel()
		return nil, sdk.NewAgentError("no user/assistant messages provided", sdk.ErrorInvalidRequest, nil)
	}

	params := anthropic.MessageNewParams{
		Model:     model.ModelID,
		MaxTokens: 16000,
		Messages:  msgParams,
		System:    systemBlocks,
	}

	client := anthropic.NewClient(option.WithMaxRetries(0))
	stream := client.Messages.NewStreaming(ctx, params)

	ch := make(chan sdk.StreamChunk)
	go func() {
		defer close(ch)
		defer cancel()
		defer stream.Close()

		for stream.Next() {
			event := stream.Current()
			if event.Type == "content_block_delta" {
				delta := event.AsContentBlockDelta()
				if delta.Delta.Type == "text_delta" {
					text := delta.Delta.AsTextDelta().Text
					if text != "" {
						ch <- sdk.StreamChunk{Text: text}
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- sdk.StreamChunk{Err: fmt.Errorf("anthropic stream error: %w", err)}
		}
	}()

	return ch, nil
}

// Compile-time check that ClaudeProvider satisfies Provider.
var _ sdk.Provider = (*ClaudeProvider)(nil)
