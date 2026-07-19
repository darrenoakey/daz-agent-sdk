package dazagentsdk

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

// newTestOllamaProvider creates a minimal Ollama provider for integration
// tests. This avoids importing the provider sub-package (which would create
// a circular dependency in the root package tests).
func newTestOllamaProvider(baseURL string) Provider {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	return &testOllamaProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		client:  &http.Client{},
	}
}

type testOllamaProvider struct {
	baseURL string
	client  *http.Client
}

// configuredTestOllamaURL resolves a live Ollama-compatible endpoint from the
// same non-secret local configuration used by the SDK.
func configuredTestOllamaURL(t *testing.T) string {
	t.Helper()
	config, err := LoadConfig()
	if err != nil {
		t.Fatalf("load local SDK configuration: %v", err)
	}
	names := make([]string, 0, len(config.Providers))
	for name := range config.Providers {
		names = append(names, name)
	}
	sort.Strings(names)
	failures := make([]string, 0, len(names))
	for _, name := range names {
		baseURL := configuredProviderURL(config, name)
		if baseURL == "" {
			continue
		}
		available, err := testOllamaAvailable(baseURL)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s=%s: %v", name, baseURL, err))
			continue
		}
		if available {
			return baseURL
		}
		failures = append(failures, fmt.Sprintf("%s=%s: HTTP status was not 200", name, baseURL))
	}
	t.Fatalf("no live Ollama-compatible endpoint found in local SDK configuration: %s", strings.Join(failures, "; "))
	return ""
}

func configuredProviderURL(config *Config, name string) string {
	providerConfig := config.Providers[name]
	baseURL, _ := providerConfig["base_url"].(string)
	return baseURL
}

func testOllamaAvailable(baseURL string) (bool, error) {
	contextWithTimeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(contextWithTimeout, http.MethodGet, strings.TrimRight(baseURL, "/")+"/api/tags", nil)
	if err != nil {
		return false, err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return false, err
	}
	if _, err := io.Copy(io.Discard, response.Body); err != nil {
		if closeErr := response.Body.Close(); closeErr != nil {
			return false, fmt.Errorf("read response: %v; close response: %w", err, closeErr)
		}
		return false, err
	}
	if err := response.Body.Close(); err != nil {
		return false, err
	}
	return response.StatusCode == http.StatusOK, nil
}

func (o *testOllamaProvider) Name() string { return "ollama" }

func (o *testOllamaProvider) Available(ctx context.Context) (bool, error) {
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
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		if closeErr := resp.Body.Close(); closeErr != nil {
			return false, nil
		}
		return false, nil
	}
	if err := resp.Body.Close(); err != nil {
		return false, nil
	}
	return resp.StatusCode == 200, nil
}

func (o *testOllamaProvider) ListModels(_ context.Context) ([]ModelInfo, error) {
	return nil, nil
}

func (o *testOllamaProvider) Complete(ctx context.Context, messages []Message, model ModelInfo, opts CompleteOpts) (*Response, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	type chatReq struct {
		Model    string              `json:"model"`
		Messages []map[string]string `json:"messages"`
		Stream   bool                `json:"stream"`
	}
	msgs := make([]map[string]string, len(messages))
	for i, m := range messages {
		msgs[i] = map[string]string{"role": m.Role, "content": m.Content}
	}
	payload := chatReq{Model: model.ModelID, Messages: msgs, Stream: false}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		respBody, readErr := io.ReadAll(resp.Body)
		closeErr := resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("read ollama error response: %w", readErr)
		}
		if closeErr != nil {
			return nil, fmt.Errorf("close ollama error response: %w", closeErr)
		}
		return nil, fmt.Errorf("ollama error %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		if closeErr := resp.Body.Close(); closeErr != nil {
			return nil, fmt.Errorf("decode ollama response: %v; close response: %w", err, closeErr)
		}
		return nil, fmt.Errorf("decode: %w", err)
	}
	if err := resp.Body.Close(); err != nil {
		return nil, fmt.Errorf("close ollama response: %w", err)
	}

	return &Response{
		Text:           chatResp.Message.Content,
		ModelUsed:      model,
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
		Usage:          map[string]any{},
	}, nil
}

func (o *testOllamaProvider) Stream(ctx context.Context, messages []Message, model ModelInfo, opts StreamOpts) (<-chan StreamChunk, error) {
	return nil, fmt.Errorf("stream not implemented in test provider")
}
