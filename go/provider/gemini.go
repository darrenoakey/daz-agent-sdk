// Package provider contains concrete AI provider implementations.
package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"google.golang.org/genai"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// geminiModels is the static catalog of supported Gemini models.
var geminiModels = []sdk.ModelInfo{
	{
		Provider:             "gemini",
		ModelID:              "gemini-2.5-pro",
		DisplayName:          "Gemini 2.5 Pro",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured, sdk.CapabilityAgentic},
		Tier:                 sdk.TierHigh,
		SupportsStreaming:     true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "gemini",
		ModelID:              "gemini-2.5-flash",
		DisplayName:          "Gemini 2.5 Flash",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierMedium,
		SupportsStreaming:     true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "gemini",
		ModelID:              "gemini-2.5-flash-lite",
		DisplayName:          "Gemini 2.5 Flash Lite",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierLow,
		SupportsStreaming:     true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        false,
	},
}

// GeminiProvider calls the Gemini API using the official Google GenAI Go SDK.
// Authentication is via GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
type GeminiProvider struct{}

// NewGeminiProvider returns a new GeminiProvider.
func NewGeminiProvider() *GeminiProvider {
	return &GeminiProvider{}
}

// geminiAPIKey returns the first non-empty value of GEMINI_API_KEY or GOOGLE_API_KEY.
func geminiAPIKey() string {
	if k := os.Getenv("GEMINI_API_KEY"); k != "" {
		return k
	}
	return os.Getenv("GOOGLE_API_KEY")
}

// newGeminiClient creates a Gemini API client using the available API key.
// Returns an error when no key is configured.
func newGeminiClient(ctx context.Context) (*genai.Client, error) {
	key := geminiAPIKey()
	if key == "" {
		return nil, sdk.NewAgentError("no Gemini API key set (GEMINI_API_KEY or GOOGLE_API_KEY)", sdk.ErrorAuth, nil)
	}
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  key,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to create Gemini client: %v", err), sdk.ErrorAuth, nil)
	}
	return client, nil
}

// Name returns "gemini".
func (g *GeminiProvider) Name() string {
	return "gemini"
}

// Available returns true when a Gemini API key is set and the API responds.
// Returns false (not an error) when the key is absent or the call fails.
func (g *GeminiProvider) Available(ctx context.Context) (bool, error) {
	if geminiAPIKey() == "" {
		return false, nil
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	client, err := newGeminiClient(ctx)
	if err != nil {
		return false, nil
	}

	// List one page of models to confirm connectivity.
	_, err = client.Models.List(ctx, nil)
	if err != nil {
		return false, nil
	}
	return true, nil
}

// ListModels returns the static Gemini model catalog.
// No network call is made; the list is hard-coded to match the requirements.
func (g *GeminiProvider) ListModels(_ context.Context) ([]sdk.ModelInfo, error) {
	result := make([]sdk.ModelInfo, len(geminiModels))
	copy(result, geminiModels)
	return result, nil
}

// buildGeminiContents converts sdk.Message slice into Gemini Content objects.
// System messages are merged and returned as SystemInstruction.
// User and assistant (model) messages become history Content entries.
func buildGeminiContents(messages []sdk.Message) (*genai.Content, []*genai.Content) {
	var systemParts []string
	var contents []*genai.Content

	for _, m := range messages {
		switch m.Role {
		case "system":
			systemParts = append(systemParts, m.Content)
		case "assistant":
			contents = append(contents, &genai.Content{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{{Text: m.Content}},
			})
		default:
			// user and any other roles treated as user
			contents = append(contents, &genai.Content{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{{Text: m.Content}},
			})
		}
	}

	var systemInstruction *genai.Content
	if len(systemParts) > 0 {
		systemInstruction = &genai.Content{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{{Text: strings.Join(systemParts, "\n")}},
		}
	}
	return systemInstruction, contents
}

// classifyGeminiError maps a Gemini SDK error to an ErrorKind.
func classifyGeminiError(err error) sdk.ErrorKind {
	// Try to match a concrete APIError from the Gemini SDK.
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return classifyGeminiStatusCode(apiErr.Code)
	}

	// Fall back to message-based classification.
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "timeout") || strings.Contains(msg, "deadline exceeded") {
		return sdk.ErrorTimeout
	}
	if strings.Contains(msg, "connection refused") || strings.Contains(msg, "no such host") {
		return sdk.ErrorNotAvailable
	}
	if strings.Contains(msg, "api key") || strings.Contains(msg, "unauthorized") || strings.Contains(msg, "permission denied") {
		return sdk.ErrorAuth
	}
	if strings.Contains(msg, "quota") || strings.Contains(msg, "rate limit") || strings.Contains(msg, "resource exhausted") {
		return sdk.ErrorRateLimit
	}

	// Scan for an embedded HTTP status code.
	for _, part := range strings.Fields(err.Error()) {
		var code int
		if _, scanErr := fmt.Sscanf(part, "%d", &code); scanErr == nil && code >= 400 {
			return classifyGeminiStatusCode(code)
		}
	}
	return sdk.ErrorInternal
}

// classifyGeminiStatusCode maps HTTP status codes to ErrorKind values.
func classifyGeminiStatusCode(statusCode int) sdk.ErrorKind {
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

// Complete sends messages to the Gemini API and returns a full response.
// When opts.Schema is non-nil it is sent as the JSON-schema output format.
func (g *GeminiProvider) Complete(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.CompleteOpts) (*sdk.Response, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	systemInstruction, contents := buildGeminiContents(messages)
	if len(contents) == 0 {
		return nil, sdk.NewAgentError("no user/assistant messages provided", sdk.ErrorInvalidRequest, nil)
	}

	cfg := &genai.GenerateContentConfig{
		SystemInstruction: systemInstruction,
		MaxOutputTokens:   16000,
	}

	if opts.Schema != nil {
		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJsonSchema = opts.Schema
	}

	client, err := newGeminiClient(ctx)
	if err != nil {
		return nil, err
	}

	resp, err := client.Models.GenerateContent(ctx, model.ModelID, contents, cfg)
	if err != nil {
		kind := classifyGeminiError(err)
		return nil, sdk.NewAgentError(fmt.Sprintf("gemini complete error: %v", err), kind, nil)
	}

	text := resp.Text()

	var inputTokens, outputTokens int32
	if resp.UsageMetadata != nil {
		inputTokens = resp.UsageMetadata.PromptTokenCount
		outputTokens = resp.UsageMetadata.CandidatesTokenCount
	}

	return &sdk.Response{
		Text:           text,
		ModelUsed:      model,
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
		Usage: map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		},
	}, nil
}

// Stream sends messages to the Gemini API and returns a channel of text chunks.
func (g *GeminiProvider) Stream(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.StreamOpts) (<-chan sdk.StreamChunk, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))

	systemInstruction, contents := buildGeminiContents(messages)
	if len(contents) == 0 {
		cancel()
		return nil, sdk.NewAgentError("no user/assistant messages provided", sdk.ErrorInvalidRequest, nil)
	}

	cfg := &genai.GenerateContentConfig{
		SystemInstruction: systemInstruction,
		MaxOutputTokens:   16000,
	}

	client, err := newGeminiClient(ctx)
	if err != nil {
		cancel()
		return nil, err
	}

	ch := make(chan sdk.StreamChunk)
	go func() {
		defer close(ch)
		defer cancel()

		for chunk, err := range client.Models.GenerateContentStream(ctx, model.ModelID, contents, cfg) {
			if err != nil {
				ch <- sdk.StreamChunk{Err: fmt.Errorf("gemini stream error: %w", err)}
				return
			}
			text := chunk.Text()
			if text != "" {
				ch <- sdk.StreamChunk{Text: text}
			}
		}
	}()

	return ch, nil
}

// Compile-time check that GeminiProvider satisfies Provider.
var _ sdk.Provider = (*GeminiProvider)(nil)
