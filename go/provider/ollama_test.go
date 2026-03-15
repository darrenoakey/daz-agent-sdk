package provider

import (
	"context"
	"fmt"
	"testing"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func TestTierFromParamSize(t *testing.T) {
	tests := []struct {
		input string
		want  sdk.Tier
	}{
		{"", sdk.TierFreeFast},
		{"8B", sdk.TierFreeFast},
		{"7B", sdk.TierFreeFast},
		{"20B", sdk.TierFreeFast},
		{"30B", sdk.TierFreeThinking},
		{"70B", sdk.TierFreeThinking},
		{"500M", sdk.TierFreeFast},
		{"3.8B", sdk.TierFreeFast},
		{"20.1B", sdk.TierFreeThinking},
		{"invalid", sdk.TierFreeFast},
		{"B", sdk.TierFreeFast},
		{"0B", sdk.TierFreeFast},
	}
	for _, tt := range tests {
		got := tierFromParamSize(tt.input)
		if got != tt.want {
			t.Errorf("tierFromParamSize(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestNewOllamaProviderDefaults(t *testing.T) {
	p := NewOllamaProvider("")
	if p.Name() != "ollama" {
		t.Errorf("Name() = %q, want ollama", p.Name())
	}
	if p.baseURL != "http://localhost:11434" {
		t.Errorf("baseURL = %q, want http://localhost:11434", p.baseURL)
	}
}

func TestNewOllamaProviderCustomURL(t *testing.T) {
	p := NewOllamaProvider("http://myhost:9999/")
	if p.baseURL != "http://myhost:9999" {
		t.Errorf("baseURL = %q, want http://myhost:9999", p.baseURL)
	}
}

// Integration tests against a real Ollama instance at localhost:11434.
// These are skipped if Ollama is not available.

func skipIfOllamaUnavailable(t *testing.T, p *OllamaProvider) {
	t.Helper()
	ok, err := p.Available(context.Background())
	if err != nil || !ok {
		t.Skip("Ollama is not available at localhost:11434, skipping integration test")
	}
}

func TestOllamaAvailable(t *testing.T) {
	p := NewOllamaProvider("")
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available returned error: %v", err)
	}
	// We don't fail if Ollama is down -- just report
	t.Logf("Ollama available: %v", ok)
}

func TestOllamaListModels(t *testing.T) {
	p := NewOllamaProvider("")
	skipIfOllamaUnavailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) == 0 {
		t.Skip("No models installed in Ollama")
	}

	for _, m := range models {
		if m.Provider != "ollama" {
			t.Errorf("model %q has provider %q, want ollama", m.ModelID, m.Provider)
		}
		if m.ModelID == "" {
			t.Error("model has empty ModelID")
		}
		if m.DisplayName == "" {
			t.Error("model has empty DisplayName")
		}
		if len(m.Capabilities) == 0 {
			t.Errorf("model %q has no capabilities", m.ModelID)
		}
		// All ollama models should have text and structured
		hasText := false
		hasStructured := false
		for _, c := range m.Capabilities {
			if c == sdk.CapabilityText {
				hasText = true
			}
			if c == sdk.CapabilityStructured {
				hasStructured = true
			}
		}
		if !hasText {
			t.Errorf("model %q missing text capability", m.ModelID)
		}
		if !hasStructured {
			t.Errorf("model %q missing structured capability", m.ModelID)
		}
	}

	t.Logf("Found %d Ollama models", len(models))
}

func TestOllamaComplete(t *testing.T) {
	p := NewOllamaProvider("")
	skipIfOllamaUnavailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil || len(models) == 0 {
		t.Skip("No models available for Complete test")
	}

	// Use the first available model
	model := models[0]
	messages := []sdk.Message{
		{Role: "user", Content: "Say exactly: hello world"},
	}

	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete returned empty text")
	}
	if resp.ModelUsed.ModelID != model.ModelID {
		t.Errorf("ModelUsed.ModelID = %q, want %q", resp.ModelUsed.ModelID, model.ModelID)
	}
	t.Logf("Complete response: %q", resp.Text)
}

func TestOllamaStream(t *testing.T) {
	p := NewOllamaProvider("")
	skipIfOllamaUnavailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil || len(models) == 0 {
		t.Skip("No models available for Stream test")
	}

	model := models[0]
	messages := []sdk.Message{
		{Role: "user", Content: "Say exactly: hello"},
	}

	ch, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var fullText string
	chunks := 0
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("Stream chunk error: %v", chunk.Err)
		}
		fullText += chunk.Text
		chunks++
	}

	if fullText == "" {
		t.Error("Stream produced no text")
	}
	t.Logf("Stream produced %d chunks, full text: %q", chunks, fullText)
}

func TestOllamaAvailableUnreachable(t *testing.T) {
	// Test against an unreachable host
	p := NewOllamaProvider("http://127.0.0.1:1")
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available should not return error, got: %v", err)
	}
	if ok {
		t.Error("Available should return false for unreachable host")
	}
}

func TestOllamaCompleteUnreachable(t *testing.T) {
	p := NewOllamaProvider("http://127.0.0.1:1")
	_, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "user", Content: "hello"},
	}, sdk.ModelInfo{ModelID: "test"}, sdk.CompleteOpts{Timeout: 2})

	if err == nil {
		t.Fatal("expected error for unreachable host")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorNotAvailable {
		t.Errorf("expected ErrorNotAvailable, got %q", agentErr.Kind)
	}
}

func TestClassifyHTTPError(t *testing.T) {
	tests := []struct {
		msg  string
		want sdk.ErrorKind
	}{
		{"connection refused", sdk.ErrorNotAvailable},
		{"dial tcp 127.0.0.1:1: connect: connection refused", sdk.ErrorNotAvailable},
		{"timeout exceeded", sdk.ErrorTimeout},
		{"context deadline exceeded", sdk.ErrorTimeout},
		{"something else broke", sdk.ErrorInternal},
	}
	for _, tt := range tests {
		got := classifyHTTPError(fmt.Errorf("%s", tt.msg))
		if got != tt.want {
			t.Errorf("classifyHTTPError(%q) = %q, want %q", tt.msg, got, tt.want)
		}
	}
}

func TestBuildMessages(t *testing.T) {
	msgs := []sdk.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}
	result := buildMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}
	if result[0]["role"] != "system" {
		t.Errorf("first message role = %q", result[0]["role"])
	}
	if result[1]["content"] != "Hello" {
		t.Errorf("second message content = %q", result[1]["content"])
	}
}

