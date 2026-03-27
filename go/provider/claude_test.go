package provider

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// skipIfNoAPIKey skips the test when ANTHROPIC_API_KEY is not set.
func skipIfNoAPIKey(t *testing.T) {
	t.Helper()
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}
}

// --- Unit tests ---

func TestClaudeProviderName(t *testing.T) {
	p := NewClaudeProvider()
	if p.Name() != "claude" {
		t.Errorf("Name() = %q, want claude", p.Name())
	}
}

func TestClaudeListModels_Static(t *testing.T) {
	p := NewClaudeProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) != 3 {
		t.Fatalf("expected 3 models, got %d", len(models))
	}

	byID := make(map[string]sdk.ModelInfo)
	for _, m := range models {
		byID[m.ModelID] = m
	}

	opus, ok := byID["claude-opus-4-6"]
	if !ok {
		t.Fatal("missing claude-opus-4-6")
	}
	if opus.Tier != sdk.TierHigh {
		t.Errorf("opus tier = %q, want high", opus.Tier)
	}
	if !opus.SupportsTools {
		t.Error("opus should support tools")
	}
	hasAgentic := false
	for _, c := range opus.Capabilities {
		if c == sdk.CapabilityAgentic {
			hasAgentic = true
		}
	}
	if !hasAgentic {
		t.Error("opus should have agentic capability")
	}

	sonnet, ok := byID["claude-sonnet-4-6"]
	if !ok {
		t.Fatal("missing claude-sonnet-4-6")
	}
	if sonnet.Tier != sdk.TierMedium {
		t.Errorf("sonnet tier = %q, want medium", sonnet.Tier)
	}

	haiku, ok := byID["claude-haiku-4-5-20251001"]
	if !ok {
		t.Fatal("missing claude-haiku-4-5-20251001")
	}
	if haiku.Tier != sdk.TierLow {
		t.Errorf("haiku tier = %q, want low", haiku.Tier)
	}

	for _, m := range models {
		if m.Provider != "claude" {
			t.Errorf("model %q has provider %q, want claude", m.ModelID, m.Provider)
		}
		if m.DisplayName == "" {
			t.Errorf("model %q has empty DisplayName", m.ModelID)
		}
		if !m.SupportsStreaming {
			t.Errorf("model %q should support streaming", m.ModelID)
		}
		if !m.SupportsStructured {
			t.Errorf("model %q should support structured", m.ModelID)
		}
		if !m.SupportsConversation {
			t.Errorf("model %q should support conversation", m.ModelID)
		}
	}
}

func TestClassifyAnthropicError(t *testing.T) {
	tests := []struct {
		statusCode int
		want       sdk.ErrorKind
	}{
		{429, sdk.ErrorRateLimit},
		{401, sdk.ErrorAuth},
		{403, sdk.ErrorAuth},
		{408, sdk.ErrorTimeout},
		{400, sdk.ErrorInvalidRequest},
		{422, sdk.ErrorInvalidRequest},
		{503, sdk.ErrorNotAvailable},
		{500, sdk.ErrorInternal},
		{502, sdk.ErrorInternal},
	}
	for _, tt := range tests {
		got := classifyAnthropicStatusCode(tt.statusCode)
		if got != tt.want {
			t.Errorf("classifyAnthropicStatusCode(%d) = %q, want %q", tt.statusCode, got, tt.want)
		}
	}
}

func TestBuildAnthropicMessages_SeparatesSystem(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
		{Role: "user", Content: "Bye"},
	}
	system, params := buildAnthropicMessages(messages)
	if len(system) != 1 {
		t.Fatalf("expected 1 system block, got %d", len(system))
	}
	if system[0].Text != "You are helpful." {
		t.Errorf("system text = %q", system[0].Text)
	}
	if len(params) != 3 {
		t.Fatalf("expected 3 message params, got %d", len(params))
	}
	if string(params[0].Role) != "user" {
		t.Errorf("first param role = %q, want user", params[0].Role)
	}
	if string(params[1].Role) != "assistant" {
		t.Errorf("second param role = %q, want assistant", params[1].Role)
	}
}

func TestBuildAnthropicMessages_NoSystem(t *testing.T) {
	messages := []sdk.Message{
		{Role: "user", Content: "Hello"},
	}
	system, params := buildAnthropicMessages(messages)
	if len(system) != 0 {
		t.Fatalf("expected 0 system blocks, got %d", len(system))
	}
	if len(params) != 1 {
		t.Fatalf("expected 1 message param, got %d", len(params))
	}
}

func TestBuildAnthropicMessages_MultipleSystemsMerged(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "First rule."},
		{Role: "system", Content: "Second rule."},
		{Role: "user", Content: "Hello"},
	}
	system, params := buildAnthropicMessages(messages)
	if len(system) != 1 {
		t.Fatalf("expected 1 merged system block, got %d", len(system))
	}
	if !strings.Contains(system[0].Text, "First rule.") {
		t.Error("merged system should contain first rule")
	}
	if !strings.Contains(system[0].Text, "Second rule.") {
		t.Error("merged system should contain second rule")
	}
	if len(params) != 1 {
		t.Fatalf("expected 1 user param, got %d", len(params))
	}
}

// --- Integration tests (skipped if no API key) ---

func TestClaudeAvailable_WithKey(t *testing.T) {
	skipIfNoAPIKey(t)
	p := NewClaudeProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available error: %v", err)
	}
	if !ok {
		t.Error("expected Available to return true with valid API key")
	}
}

func TestClaudeAvailable_WithoutKey(t *testing.T) {
	// Temporarily clear the key to test false-return behaviour.
	orig := os.Getenv("ANTHROPIC_API_KEY")
	if orig == "" {
		t.Skip("ANTHROPIC_API_KEY not set, cannot test key-absent path")
	}
	os.Unsetenv("ANTHROPIC_API_KEY")
	defer os.Setenv("ANTHROPIC_API_KEY", orig)

	p := NewClaudeProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available should not return error when key missing, got: %v", err)
	}
	if ok {
		t.Error("Available should return false when API key is missing")
	}
}

func TestClaudeComplete_BasicText(t *testing.T) {
	skipIfNoAPIKey(t)
	p := NewClaudeProvider()
	models, err := p.ListModels(context.Background())
	if err != nil || len(models) == 0 {
		t.Fatal("could not get models")
	}

	// Use haiku for speed and cost
	var model sdk.ModelInfo
	for _, m := range models {
		if m.ModelID == "claude-haiku-4-5-20251001" {
			model = m
			break
		}
	}
	if model.ModelID == "" {
		model = models[len(models)-1]
	}

	messages := []sdk.Message{
		{Role: "user", Content: "Reply with exactly the word: pong"},
	}
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{
		Timeout: 30,
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

func TestClaudeComplete_StructuredOutput(t *testing.T) {
	skipIfNoAPIKey(t)
	p := NewClaudeProvider()
	models, err := p.ListModels(context.Background())
	if err != nil || len(models) == 0 {
		t.Fatal("could not get models")
	}

	var model sdk.ModelInfo
	for _, m := range models {
		if m.ModelID == "claude-haiku-4-5-20251001" {
			model = m
			break
		}
	}
	if model.ModelID == "" {
		model = models[len(models)-1]
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"result": map[string]any{
				"type":        "integer",
				"description": "The answer",
			},
		},
		"required":             []any{"result"},
		"additionalProperties": false,
	}

	messages := []sdk.Message{
		{Role: "user", Content: "What is 2 + 2? Respond in JSON."},
	}
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 30,
	})
	if err != nil {
		t.Fatalf("Complete with schema error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete returned empty text")
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &parsed); err != nil {
		t.Fatalf("structured output is not valid JSON: %v, text: %q", err, resp.Text)
	}
	result, ok := parsed["result"]
	if !ok {
		t.Errorf("parsed JSON missing 'result' field, got: %v", parsed)
	}
	t.Logf("Structured result: %v", result)
}

func TestClaudeStream_BasicText(t *testing.T) {
	skipIfNoAPIKey(t)
	p := NewClaudeProvider()
	models, err := p.ListModels(context.Background())
	if err != nil || len(models) == 0 {
		t.Fatal("could not get models")
	}

	var model sdk.ModelInfo
	for _, m := range models {
		if m.ModelID == "claude-haiku-4-5-20251001" {
			model = m
			break
		}
	}
	if model.ModelID == "" {
		model = models[len(models)-1]
	}

	messages := []sdk.Message{
		{Role: "user", Content: "Reply with exactly the word: ping"},
	}

	ch, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{
		Timeout: 30,
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

func TestClaudeComplete_SystemMessage(t *testing.T) {
	skipIfNoAPIKey(t)
	p := NewClaudeProvider()
	models, err := p.ListModels(context.Background())
	if err != nil || len(models) == 0 {
		t.Fatal("could not get models")
	}

	var model sdk.ModelInfo
	for _, m := range models {
		if m.ModelID == "claude-haiku-4-5-20251001" {
			model = m
			break
		}
	}
	if model.ModelID == "" {
		model = models[len(models)-1]
	}

	messages := []sdk.Message{
		{Role: "system", Content: "You only ever reply with the single word BANANA."},
		{Role: "user", Content: "What is your favourite fruit?"},
	}
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{
		Timeout: 30,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if !strings.Contains(strings.ToUpper(resp.Text), "BANANA") {
		t.Errorf("expected BANANA in response, got: %q", resp.Text)
	}
	t.Logf("System-prompted response: %q", resp.Text)
}
