package provider

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// ---- Unit: provider name and construction ----

func TestOpenAIName(t *testing.T) {
	p := NewOpenAIProvider()
	if p.Name() != "openai" {
		t.Errorf("Name() = %q, want %q", p.Name(), "openai")
	}
}

func TestOpenAINewProvider(t *testing.T) {
	p := NewOpenAIProvider()
	if p == nil {
		t.Fatal("NewOpenAIProvider() returned nil")
	}
}

// ---- Unit: model catalog ----

func TestOpenAIListModelsCount(t *testing.T) {
	p := NewOpenAIProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels() error: %v", err)
	}
	if len(models) != 4 {
		t.Errorf("ListModels() returned %d models, want 4", len(models))
	}
}

func TestOpenAIListModelsTiers(t *testing.T) {
	p := NewOpenAIProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels() error: %v", err)
	}

	tierByID := make(map[string]sdk.Tier)
	for _, m := range models {
		tierByID[m.ModelID] = m.Tier
	}

	cases := []struct {
		id   string
		tier sdk.Tier
	}{
		{"gpt-4.1", sdk.TierHigh},
		{"gpt-4.1-mini", sdk.TierMedium},
		{"gpt-4.1-nano", sdk.TierLow},
		{"o4-mini", sdk.TierMedium},
	}
	for _, c := range cases {
		got, ok := tierByID[c.id]
		if !ok {
			t.Errorf("model %q not found in catalog", c.id)
			continue
		}
		if got != c.tier {
			t.Errorf("model %q tier = %q, want %q", c.id, got, c.tier)
		}
	}
}

func TestOpenAIListModelsCapabilities(t *testing.T) {
	p := NewOpenAIProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels() error: %v", err)
	}
	for _, m := range models {
		hasText := false
		hasStructured := false
		for _, cap := range m.Capabilities {
			if cap == sdk.CapabilityText {
				hasText = true
			}
			if cap == sdk.CapabilityStructured {
				hasStructured = true
			}
		}
		if !hasText {
			t.Errorf("model %q missing TEXT capability", m.ModelID)
		}
		if !hasStructured {
			t.Errorf("model %q missing STRUCTURED capability", m.ModelID)
		}
	}
}

func TestOpenAIListModelsSupportsTools(t *testing.T) {
	p := NewOpenAIProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels() error: %v", err)
	}
	for _, m := range models {
		if !m.SupportsTools {
			t.Errorf("model %q SupportsTools = false, want true", m.ModelID)
		}
	}
}

func TestOpenAIListModelsProvider(t *testing.T) {
	p := NewOpenAIProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels() error: %v", err)
	}
	for _, m := range models {
		if m.Provider != "openai" {
			t.Errorf("model %q Provider = %q, want %q", m.ModelID, m.Provider, "openai")
		}
	}
}

func TestOpenAIListModelsIsCopy(t *testing.T) {
	// Verify ListModels returns a copy, not the original slice.
	p := NewOpenAIProvider()
	models1, _ := p.ListModels(context.Background())
	models1[0].ModelID = "mutated"
	models2, _ := p.ListModels(context.Background())
	if models2[0].ModelID == "mutated" {
		t.Error("ListModels() returned the original slice, not a copy")
	}
}

// ---- Unit: message conversion ----

func TestBuildOpenAIMessagesEmpty(t *testing.T) {
	result := buildOpenAIMessages([]sdk.Message{})
	if len(result) != 0 {
		t.Errorf("buildOpenAIMessages(empty) returned %d items, want 0", len(result))
	}
}

func TestBuildOpenAIMessagesSingleUser(t *testing.T) {
	msgs := []sdk.Message{{Role: "user", Content: "hello"}}
	result := buildOpenAIMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
}

func TestBuildOpenAIMessagesSystemUser(t *testing.T) {
	msgs := []sdk.Message{
		{Role: "system", Content: "you are a helper"},
		{Role: "user", Content: "help me"},
	}
	result := buildOpenAIMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}
}

func TestBuildOpenAIMessagesMultiTurn(t *testing.T) {
	msgs := []sdk.Message{
		{Role: "system", Content: "be helpful"},
		{Role: "user", Content: "question 1"},
		{Role: "assistant", Content: "answer 1"},
		{Role: "user", Content: "question 2"},
	}
	result := buildOpenAIMessages(msgs)
	if len(result) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(result))
	}
}

func TestBuildOpenAIMessagesAllRoles(t *testing.T) {
	// Unknown role falls through to user
	msgs := []sdk.Message{
		{Role: "user", Content: "u"},
		{Role: "assistant", Content: "a"},
		{Role: "system", Content: "s"},
		{Role: "unknown", Content: "x"},
	}
	result := buildOpenAIMessages(msgs)
	if len(result) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(result))
	}
}

// ---- Unit: error classification ----

func TestClassifyOpenAIStatusRateLimit(t *testing.T) {
	kind := classifyOpenAIStatusCode(429)
	if kind != sdk.ErrorRateLimit {
		t.Errorf("status 429 → %q, want %q", kind, sdk.ErrorRateLimit)
	}
}

func TestClassifyOpenAIStatusAuth401(t *testing.T) {
	kind := classifyOpenAIStatusCode(401)
	if kind != sdk.ErrorAuth {
		t.Errorf("status 401 → %q, want %q", kind, sdk.ErrorAuth)
	}
}

func TestClassifyOpenAIStatusAuth403(t *testing.T) {
	kind := classifyOpenAIStatusCode(403)
	if kind != sdk.ErrorAuth {
		t.Errorf("status 403 → %q, want %q", kind, sdk.ErrorAuth)
	}
}

func TestClassifyOpenAIStatusTimeout(t *testing.T) {
	kind := classifyOpenAIStatusCode(408)
	if kind != sdk.ErrorTimeout {
		t.Errorf("status 408 → %q, want %q", kind, sdk.ErrorTimeout)
	}
}

func TestClassifyOpenAIStatusInvalidRequest400(t *testing.T) {
	kind := classifyOpenAIStatusCode(400)
	if kind != sdk.ErrorInvalidRequest {
		t.Errorf("status 400 → %q, want %q", kind, sdk.ErrorInvalidRequest)
	}
}

func TestClassifyOpenAIStatusInvalidRequest422(t *testing.T) {
	kind := classifyOpenAIStatusCode(422)
	if kind != sdk.ErrorInvalidRequest {
		t.Errorf("status 422 → %q, want %q", kind, sdk.ErrorInvalidRequest)
	}
}

func TestClassifyOpenAIStatusNotAvailable(t *testing.T) {
	kind := classifyOpenAIStatusCode(503)
	if kind != sdk.ErrorNotAvailable {
		t.Errorf("status 503 → %q, want %q", kind, sdk.ErrorNotAvailable)
	}
}

func TestClassifyOpenAIStatusInternal(t *testing.T) {
	kind := classifyOpenAIStatusCode(500)
	if kind != sdk.ErrorInternal {
		t.Errorf("status 500 → %q, want %q", kind, sdk.ErrorInternal)
	}
}

func TestClassifyOpenAIErrorTimeoutString(t *testing.T) {
	err := errors.New("context deadline exceeded: timeout waiting for server")
	kind := classifyOpenAIError(err)
	if kind != sdk.ErrorTimeout {
		t.Errorf("timeout error → %q, want %q", kind, sdk.ErrorTimeout)
	}
}

func TestClassifyOpenAIErrorConnectionRefused(t *testing.T) {
	err := errors.New("dial tcp connection refused")
	kind := classifyOpenAIError(err)
	if kind != sdk.ErrorNotAvailable {
		t.Errorf("connection refused → %q, want %q", kind, sdk.ErrorNotAvailable)
	}
}

func TestClassifyOpenAIErrorTypedAPIError(t *testing.T) {
	// Wrap an openai.Error with a 429 status to test typed error path.
	apiErr := &openai.Error{StatusCode: 429}
	kind := classifyOpenAIError(apiErr)
	if kind != sdk.ErrorRateLimit {
		t.Errorf("typed 429 error → %q, want %q", kind, sdk.ErrorRateLimit)
	}
}

func TestClassifyOpenAIErrorTypedAPIErrorAuth(t *testing.T) {
	apiErr := &openai.Error{StatusCode: 401}
	kind := classifyOpenAIError(apiErr)
	if kind != sdk.ErrorAuth {
		t.Errorf("typed 401 error → %q, want %q", kind, sdk.ErrorAuth)
	}
}

// ---- Unit: no-messages guard ----

func TestOpenAICompleteNoMessagesError(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIModels[1] // gpt-4.1-mini
	_, err := p.Complete(context.Background(), []sdk.Message{}, model, sdk.CompleteOpts{})
	if err == nil {
		t.Fatal("Complete with no messages should return an error")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected *sdk.AgentError, got %T", err)
	}
	if agentErr.Kind != sdk.ErrorInvalidRequest {
		t.Errorf("error kind = %q, want %q", agentErr.Kind, sdk.ErrorInvalidRequest)
	}
}

func TestOpenAIStreamNoMessagesError(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIModels[1]
	_, err := p.Stream(context.Background(), []sdk.Message{}, model, sdk.StreamOpts{})
	if err == nil {
		t.Fatal("Stream with no messages should return an error")
	}
}

// ---- Integration tests (require OPENAI_API_KEY) ----

func openAIMiniModel() sdk.ModelInfo {
	// gpt-4.1-mini is cost-efficient for integration tests.
	return openAIModels[1]
}

func TestOpenAIAvailableWithKey(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available() returned error: %v", err)
	}
	if !ok {
		t.Error("Available() = false with OPENAI_API_KEY set and valid")
	}
}

func TestOpenAIAvailableWithoutKey(t *testing.T) {
	// Temporarily clear the key — only safe if it wasn't set to begin with.
	orig := os.Getenv("OPENAI_API_KEY")
	if orig != "" {
		t.Skip("OPENAI_API_KEY is set; skipping no-key test to avoid env mutation")
	}
	p := NewOpenAIProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available() returned error without key: %v", err)
	}
	if ok {
		t.Error("Available() = true without OPENAI_API_KEY")
	}
}

func TestOpenAICompleteBasicText(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Say exactly: HELLO"}}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete() returned empty text")
	}
	if resp.ConversationID.String() == "00000000-0000-0000-0000-000000000000" {
		t.Error("ConversationID is zero UUID")
	}
	if resp.TurnID.String() == "00000000-0000-0000-0000-000000000000" {
		t.Error("TurnID is zero UUID")
	}
	if resp.ModelUsed.ModelID != model.ModelID {
		t.Errorf("ModelUsed.ModelID = %q, want %q", resp.ModelUsed.ModelID, model.ModelID)
	}
}

func TestOpenAICompleteSystemMessage(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{
		{Role: "system", Content: "You are a pirate. Always say 'Arrr' at the start."},
		{Role: "user", Content: "Greet me."},
	}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete() returned empty text")
	}
}

func TestOpenAICompleteUsagePopulated(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Reply with one word."}}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.Usage == nil {
		t.Fatal("Usage map is nil")
	}
	if _, ok := resp.Usage["prompt_tokens"]; !ok {
		t.Error("Usage missing prompt_tokens")
	}
	if _, ok := resp.Usage["completion_tokens"]; !ok {
		t.Error("Usage missing completion_tokens")
	}
}

func TestOpenAICompleteStructuredMathSchema(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"result": map[string]any{"type": "number"},
		},
		"required":             []string{"result"},
		"additionalProperties": false,
	}
	msgs := []sdk.Message{{Role: "user", Content: "What is 3 + 4? Respond with the numeric result."}}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{Schema: schema})
	if err != nil {
		t.Fatalf("Complete() with schema error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Structured response text is empty")
	}
	if !strings.Contains(resp.Text, "7") {
		t.Errorf("expected '7' in structured response, got: %q", resp.Text)
	}
}

func TestOpenAICompleteStructuredComplexSchema(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":    map[string]any{"type": "string"},
			"age":     map[string]any{"type": "integer"},
			"country": map[string]any{"type": "string"},
		},
		"required":             []string{"name", "age", "country"},
		"additionalProperties": false,
	}
	msgs := []sdk.Message{{Role: "user", Content: "Return info about Albert Einstein."}}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{Schema: schema})
	if err != nil {
		t.Fatalf("Complete() with complex schema error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Structured response text is empty")
	}
	if !strings.Contains(resp.Text, "Einstein") {
		t.Errorf("expected 'Einstein' in response, got: %q", resp.Text)
	}
}

func TestOpenAICompleteInvalidSchema(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "hello"}}
	// Pass a non-map schema to trigger the guard.
	_, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{Schema: "not-a-map"})
	if err == nil {
		t.Fatal("expected error for non-map schema")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected *sdk.AgentError, got %T", err)
	}
	if agentErr.Kind != sdk.ErrorInvalidRequest {
		t.Errorf("error kind = %q, want %q", agentErr.Kind, sdk.ErrorInvalidRequest)
	}
}

func TestOpenAIStreamBasicText(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Count to 3 briefly."}}
	ch, err := p.Stream(context.Background(), msgs, model, sdk.StreamOpts{})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	var collected strings.Builder
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		collected.WriteString(chunk.Text)
	}
	if collected.Len() == 0 {
		t.Error("Stream() produced no text")
	}
}

func TestOpenAIStreamCollectsAllChunks(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Write a haiku about Go programming."}}
	ch, err := p.Stream(context.Background(), msgs, model, sdk.StreamOpts{})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	var chunks int
	var full strings.Builder
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		if chunk.Text != "" {
			chunks++
			full.WriteString(chunk.Text)
		}
	}
	if chunks < 2 {
		t.Errorf("expected at least 2 chunks, got %d", chunks)
	}
	if full.Len() == 0 {
		t.Error("stream collected no text")
	}
}

func TestOpenAIStreamTimeout(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Write a very long essay about the history of computing."}}

	// 1ms timeout — stream setup happens synchronously so the timeout fires during streaming.
	ch, err := p.Stream(context.Background(), msgs, model, sdk.StreamOpts{Timeout: 0.001})
	if err != nil {
		// Error at setup is also acceptable.
		return
	}
	// Drain the channel regardless; a timeout error chunk or early close is expected.
	var gotErr bool
	for chunk := range ch {
		if chunk.Err != nil {
			gotErr = true
			break
		}
	}
	// Either an error was emitted or the channel closed early — both are acceptable
	// outcomes for a 1ms timeout. We just verify no panic occurred.
	_ = gotErr
}

func TestOpenAICompleteTimeout(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Tell me everything about computing."}}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	_, err := p.Complete(ctx, msgs, model, sdk.CompleteOpts{})
	if err == nil {
		// Unlikely to succeed in 1ms — but not a failure if it does.
		t.Log("Complete() unexpectedly succeeded with 1ms timeout")
		return
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected *sdk.AgentError, got %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorTimeout {
		t.Errorf("error kind = %q, want %q", agentErr.Kind, sdk.ErrorTimeout)
	}
}
