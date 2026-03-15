// Package provider contains concrete AI provider implementations.
package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// titleCase capitalises the first letter of each space-separated word.
// Replaces the deprecated strings.Title without adding an external dependency.
func titleCase(s string) string {
	words := strings.Fields(s)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, " ")
}

// tierFromParamSize classifies a model into a tier based on its parameter
// count string. Models with >20B params get FreeThinking, everything else
// FreeFast. Returns FreeFast when parameter size is unknown or unparseable.
func tierFromParamSize(paramSize string) sdk.Tier {
	if paramSize == "" {
		return sdk.TierFreeFast
	}
	s := strings.ToUpper(strings.ReplaceAll(paramSize, " ", ""))
	var count float64
	if strings.HasSuffix(s, "B") {
		if _, err := fmt.Sscanf(s[:len(s)-1], "%f", &count); err != nil {
			return sdk.TierFreeFast
		}
	} else if strings.HasSuffix(s, "M") {
		if _, err := fmt.Sscanf(s[:len(s)-1], "%f", &count); err != nil {
			return sdk.TierFreeFast
		}
		count /= 1000.0
	} else {
		return sdk.TierFreeFast
	}
	if count > 20.0 {
		return sdk.TierFreeThinking
	}
	return sdk.TierFreeFast
}

// classifyHTTPError maps HTTP/network errors to ErrorKind.
func classifyHTTPError(err error) sdk.ErrorKind {
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "no such host") ||
		strings.Contains(msg, "dial tcp") {
		return sdk.ErrorNotAvailable
	}
	if strings.Contains(msg, "timeout") ||
		strings.Contains(msg, "deadline exceeded") {
		return sdk.ErrorTimeout
	}
	return sdk.ErrorInternal
}

// OllamaProvider communicates with a locally-running Ollama instance
// via its REST API. No authentication required.
type OllamaProvider struct {
	baseURL string
	client  *http.Client
}

// NewOllamaProvider creates a new Ollama provider. If baseURL is empty,
// it defaults to http://localhost:11434.
func NewOllamaProvider(baseURL string) *OllamaProvider {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &OllamaProvider{
		baseURL: baseURL,
		client:  &http.Client{},
	}
}

// Name returns "ollama".
func (o *OllamaProvider) Name() string {
	return "ollama"
}

// Available probes the Ollama server by fetching the model list.
// Returns true if the server responds with HTTP 200.
func (o *OllamaProvider) Available(ctx context.Context) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, o.baseURL+"/api/tags", nil)
	if err != nil {
		return false, nil
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return false, nil
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == 200, nil
}

// tagsResponse is the JSON structure returned by /api/tags.
type tagsResponse struct {
	Models []tagModel `json:"models"`
}

type tagModel struct {
	Name    string         `json:"name"`
	Details tagModelDetail `json:"details"`
}

type tagModelDetail struct {
	ParameterSize string `json:"parameter_size"`
}

// ListModels fetches all models from the Ollama server.
func (o *OllamaProvider) ListModels(ctx context.Context) ([]sdk.ModelInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, o.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("ollama /api/tags returned status %d", resp.StatusCode)
	}

	var data tagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode /api/tags response: %w", err)
	}

	var models []sdk.ModelInfo
	for _, entry := range data.Models {
		modelID := entry.Name
		if modelID == "" {
			continue
		}
		paramSize := strings.TrimSpace(entry.Details.ParameterSize)
		tier := tierFromParamSize(paramSize)

		// Build display name: take part before ':', replace - and _ with spaces, title case
		baseName := modelID
		if idx := strings.Index(baseName, ":"); idx >= 0 {
			baseName = baseName[:idx]
		}
		baseName = strings.ReplaceAll(baseName, "-", " ")
		baseName = strings.ReplaceAll(baseName, "_", " ")
		display := titleCase(baseName)
		if paramSize != "" {
			display = fmt.Sprintf("%s (%s)", display, paramSize)
		}

		models = append(models, sdk.ModelInfo{
			Provider:             "ollama",
			ModelID:              modelID,
			DisplayName:          display,
			Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
			Tier:                 tier,
			SupportsStreaming:    true,
			SupportsStructured:  true,
			SupportsConversation: true,
			SupportsTools:       false,
		})
	}
	return models, nil
}

// buildMessages converts Message objects to the format Ollama expects.
func buildMessages(messages []sdk.Message) []map[string]string {
	result := make([]map[string]string, len(messages))
	for i, m := range messages {
		result[i] = map[string]string{
			"role":    m.Role,
			"content": m.Content,
		}
	}
	return result
}

// chatRequest is the JSON payload for /api/chat.
type chatRequest struct {
	Model    string              `json:"model"`
	Messages []map[string]string `json:"messages"`
	Stream   bool                `json:"stream"`
}

// chatResponse is the JSON response from /api/chat (non-streaming).
type chatResponse struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

// Complete sends messages to Ollama and returns a full response.
func (o *OllamaProvider) Complete(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.CompleteOpts) (*sdk.Response, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	payload := chatRequest{
		Model:    model.ModelID,
		Messages: buildMessages(messages),
		Stream:   false,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to marshal request: %v", err), sdk.ErrorInternal, nil)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to create request: %v", err), sdk.ErrorInternal, nil)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(req)
	if err != nil {
		kind := classifyHTTPError(err)
		return nil, sdk.NewAgentError(err.Error(), kind, nil)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 429 {
		return nil, sdk.NewAgentError(fmt.Sprintf("Ollama rate limit: %d", resp.StatusCode), sdk.ErrorRateLimit, nil)
	}
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, sdk.NewAgentError(fmt.Sprintf("Ollama error %d: %s", resp.StatusCode, string(respBody)), sdk.ErrorInternal, nil)
	}

	var chatResp chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to decode response: %v", err), sdk.ErrorInternal, nil)
	}

	return &sdk.Response{
		Text:           chatResp.Message.Content,
		ModelUsed:      model,
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
		Usage:          map[string]any{},
	}, nil
}

// Stream sends messages to Ollama and returns a channel of text chunks.
func (o *OllamaProvider) Stream(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.StreamOpts) (<-chan sdk.StreamChunk, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))

	payload := chatRequest{
		Model:    model.ModelID,
		Messages: buildMessages(messages),
		Stream:   true,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to marshal request: %v", err), sdk.ErrorInternal, nil)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to create request: %v", err), sdk.ErrorInternal, nil)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(req)
	if err != nil {
		cancel()
		kind := classifyHTTPError(err)
		return nil, sdk.NewAgentError(err.Error(), kind, nil)
	}

	if resp.StatusCode == 429 {
		resp.Body.Close()
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("Ollama rate limit: %d", resp.StatusCode), sdk.ErrorRateLimit, nil)
	}
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("Ollama error %d: %s", resp.StatusCode, string(respBody)), sdk.ErrorInternal, nil)
	}

	ch := make(chan sdk.StreamChunk)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		defer cancel()

		decoder := json.NewDecoder(resp.Body)
		for {
			var obj map[string]any
			if err := decoder.Decode(&obj); err != nil {
				if err != io.EOF {
					ch <- sdk.StreamChunk{Err: err}
				}
				return
			}

			msgObj, _ := obj["message"].(map[string]any)
			if msgObj != nil {
				if content, ok := msgObj["content"].(string); ok && content != "" {
					ch <- sdk.StreamChunk{Text: content}
				}
			}

			if done, _ := obj["done"].(bool); done {
				return
			}
		}
	}()

	return ch, nil
}

// Compile-time check that OllamaProvider satisfies Provider.
var _ sdk.Provider = (*OllamaProvider)(nil)
