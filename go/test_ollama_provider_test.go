package dazagentsdk

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
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
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
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama error %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
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
