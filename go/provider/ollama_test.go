package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
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

// ---------------------------------------------------------------------------
// Unit tests: chatRequest JSON marshaling
// ---------------------------------------------------------------------------

// TestChatRequestMarshalNoFormat verifies that omitempty suppresses the
// format key when Schema is nil.
func TestChatRequestMarshalNoFormat(t *testing.T) {
	req := chatRequest{
		Model:    "llama3",
		Messages: []map[string]string{{"role": "user", "content": "hi"}},
		Stream:   false,
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}
	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if _, ok := obj["format"]; ok {
		t.Error("format key should be absent when Format is nil")
	}
}

// TestChatRequestMarshalWithFormat verifies that Format is included when
// a non-nil schema is set.
func TestChatRequestMarshalWithFormat(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
	}
	req := chatRequest{
		Model:    "llama3",
		Messages: []map[string]string{{"role": "user", "content": "hi"}},
		Stream:   false,
		Format:   schema,
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}
	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if _, ok := obj["format"]; !ok {
		t.Error("format key should be present when Format is set")
	}
	fmtObj, ok := obj["format"].(map[string]any)
	if !ok {
		t.Fatalf("format should be an object, got %T", obj["format"])
	}
	if fmtObj["type"] != "object" {
		t.Errorf("format.type = %q, want object", fmtObj["type"])
	}
}

// TestChatRequestMarshalStreamFlag verifies the stream flag round-trips.
func TestChatRequestMarshalStreamFlag(t *testing.T) {
	for _, streamVal := range []bool{true, false} {
		req := chatRequest{
			Model:    "llama3",
			Messages: []map[string]string{},
			Stream:   streamVal,
		}
		data, _ := json.Marshal(req)
		var obj map[string]any
		json.Unmarshal(data, &obj) //nolint:errcheck
		got, _ := obj["stream"].(bool)
		if got != streamVal {
			t.Errorf("stream = %v, want %v", got, streamVal)
		}
	}
}

// ---------------------------------------------------------------------------
// Unit tests: Complete wires Schema into Format via httptest server
// ---------------------------------------------------------------------------

// newFakeChatServer creates a test HTTP server that captures the last
// /api/chat request body and returns a minimal chat response.
func newFakeChatServer(t *testing.T) (*httptest.Server, *[]byte) {
	t.Helper()
	captured := new([]byte)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/chat" {
			body, _ := io.ReadAll(r.Body)
			*captured = body
			w.Header().Set("Content-Type", "application/json")
			resp := `{"message":{"role":"assistant","content":"42"}}`
			w.Write([]byte(resp)) //nolint:errcheck
			return
		}
		// /api/tags — return empty model list so Available passes
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models":[]}`)) //nolint:errcheck
			return
		}
		http.NotFound(w, r)
	}))
	return srv, captured
}

// TestCompletePassesSchemaAsFormat verifies that when opts.Schema is set,
// Complete sends the schema in the format field.
func TestCompletePassesSchemaAsFormat(t *testing.T) {
	srv, captured := newFakeChatServer(t)
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"result": map[string]any{"type": "number"},
		},
		"required": []string{"result"},
	}

	_, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "user", Content: "What is 6 * 7?"},
	}, sdk.ModelInfo{ModelID: "llama3"}, sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 10,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(*captured, &body); err != nil {
		t.Fatalf("failed to unmarshal captured request: %v", err)
	}
	if _, ok := body["format"]; !ok {
		t.Error("format field missing from request when Schema is set")
	}
	fmtObj, ok := body["format"].(map[string]any)
	if !ok {
		t.Fatalf("format should be an object, got %T", body["format"])
	}
	if fmtObj["type"] != "object" {
		t.Errorf("format.type = %q, want object", fmtObj["type"])
	}
}

// TestCompleteOmitsFormatWhenSchemaIsNil verifies that when opts.Schema is
// nil (plain text request), no format field appears in the request body.
func TestCompleteOmitsFormatWhenSchemaIsNil(t *testing.T) {
	srv, captured := newFakeChatServer(t)
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)

	_, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "user", Content: "Say hello"},
	}, sdk.ModelInfo{ModelID: "llama3"}, sdk.CompleteOpts{
		Timeout: 10,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(*captured, &body); err != nil {
		t.Fatalf("failed to unmarshal captured request: %v", err)
	}
	if _, ok := body["format"]; ok {
		t.Error("format field should be absent when Schema is nil")
	}
}

// TestCompleteResponseFields verifies that a successful Complete response
// has a non-empty Text, non-zero UUIDs, and correct ModelUsed.
func TestCompleteResponseFields(t *testing.T) {
	srv, _ := newFakeChatServer(t)
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	model := sdk.ModelInfo{ModelID: "llama3", Provider: "ollama"}

	resp, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "user", Content: "hello"},
	}, model, sdk.CompleteOpts{Timeout: 10})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Text == "" {
		t.Error("resp.Text should not be empty")
	}
	if resp.Text != "42" {
		t.Errorf("resp.Text = %q, want 42", resp.Text)
	}
	if resp.ConversationID.String() == "00000000-0000-0000-0000-000000000000" {
		t.Error("ConversationID should not be zero UUID")
	}
	if resp.TurnID.String() == "00000000-0000-0000-0000-000000000000" {
		t.Error("TurnID should not be zero UUID")
	}
	if resp.ModelUsed.ModelID != model.ModelID {
		t.Errorf("ModelUsed.ModelID = %q, want %q", resp.ModelUsed.ModelID, model.ModelID)
	}
}

// TestCompleteDefaultTimeout verifies that a zero timeout is replaced with
// the default (300 s) without hanging the test (the fake server responds
// immediately so any positive timeout works).
func TestCompleteDefaultTimeout(t *testing.T) {
	srv, _ := newFakeChatServer(t)
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	// Pass Timeout: 0 — should fall back to 300 s default
	resp, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "user", Content: "ping"},
	}, sdk.ModelInfo{ModelID: "llama3"}, sdk.CompleteOpts{})
	if err != nil {
		t.Fatalf("Complete with zero timeout error: %v", err)
	}
	if resp.Text == "" {
		t.Error("expected non-empty response text")
	}
}

// ---------------------------------------------------------------------------
// Unit tests: error classification edge cases
// ---------------------------------------------------------------------------

// TestClassifyHTTPErrorNoSuchHost ensures "no such host" maps to ErrorNotAvailable.
func TestClassifyHTTPErrorNoSuchHost(t *testing.T) {
	kind := classifyHTTPError(errors.New("no such host"))
	if kind != sdk.ErrorNotAvailable {
		t.Errorf("expected ErrorNotAvailable, got %q", kind)
	}
}

// TestClassifyHTTPErrorDeadlineExceeded ensures "deadline exceeded" maps to ErrorTimeout.
func TestClassifyHTTPErrorDeadlineExceeded(t *testing.T) {
	kind := classifyHTTPError(errors.New("context deadline exceeded"))
	if kind != sdk.ErrorTimeout {
		t.Errorf("expected ErrorTimeout, got %q", kind)
	}
}

// TestClassifyHTTPErrorDialTCP ensures "dial tcp" maps to ErrorNotAvailable.
func TestClassifyHTTPErrorDialTCP(t *testing.T) {
	kind := classifyHTTPError(errors.New("dial tcp [::1]:11434: connect: connection refused"))
	if kind != sdk.ErrorNotAvailable {
		t.Errorf("expected ErrorNotAvailable, got %q", kind)
	}
}

// TestClassifyHTTPErrorUnknown ensures an unrecognised message maps to ErrorInternal.
func TestClassifyHTTPErrorUnknown(t *testing.T) {
	kind := classifyHTTPError(errors.New("unexpected EOF"))
	if kind != sdk.ErrorInternal {
		t.Errorf("expected ErrorInternal, got %q", kind)
	}
}

// ---------------------------------------------------------------------------
// Unit tests: HTTP error status codes via fake server
// ---------------------------------------------------------------------------

// newStatusServer creates a test server that always returns the given status code.
func newStatusServer(status int, body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		w.Write([]byte(body)) //nolint:errcheck
	}))
}

// TestCompleteRateLimitError verifies that HTTP 429 is classified as ErrorRateLimit.
func TestCompleteRateLimitError(t *testing.T) {
	srv := newStatusServer(http.StatusTooManyRequests, "rate limited")
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	_, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "user", Content: "hi"},
	}, sdk.ModelInfo{ModelID: "llama3"}, sdk.CompleteOpts{Timeout: 5})
	if err == nil {
		t.Fatal("expected error for 429 response")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected *sdk.AgentError, got %T", err)
	}
	if agentErr.Kind != sdk.ErrorRateLimit {
		t.Errorf("Kind = %q, want ErrorRateLimit", agentErr.Kind)
	}
}

// TestCompleteServerError verifies that HTTP 500 is classified as ErrorInternal.
func TestCompleteServerError(t *testing.T) {
	srv := newStatusServer(http.StatusInternalServerError, "server broke")
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	_, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "user", Content: "hi"},
	}, sdk.ModelInfo{ModelID: "llama3"}, sdk.CompleteOpts{Timeout: 5})
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected *sdk.AgentError, got %T", err)
	}
	if agentErr.Kind != sdk.ErrorInternal {
		t.Errorf("Kind = %q, want ErrorInternal", agentErr.Kind)
	}
}

// ---------------------------------------------------------------------------
// Unit tests: empty messages and edge cases
// ---------------------------------------------------------------------------

// TestBuildMessagesEmpty verifies buildMessages handles a nil/empty slice.
func TestBuildMessagesEmpty(t *testing.T) {
	result := buildMessages(nil)
	if len(result) != 0 {
		t.Errorf("expected 0 messages, got %d", len(result))
	}
	result2 := buildMessages([]sdk.Message{})
	if len(result2) != 0 {
		t.Errorf("expected 0 messages, got %d", len(result2))
	}
}

// TestCompleteWithEmptyMessages verifies Complete does not panic or error
// when sent with an empty message list (server decides what to do).
func TestCompleteWithEmptyMessages(t *testing.T) {
	srv, captured := newFakeChatServer(t)
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	_, err := p.Complete(context.Background(), []sdk.Message{},
		sdk.ModelInfo{ModelID: "llama3"}, sdk.CompleteOpts{Timeout: 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var body map[string]any
	json.Unmarshal(*captured, &body) //nolint:errcheck
	msgs, _ := body["messages"].([]any)
	if len(msgs) != 0 {
		t.Errorf("expected 0 messages in request, got %d", len(msgs))
	}
}

// ---------------------------------------------------------------------------
// Unit tests: streaming error scenario via fake server
// ---------------------------------------------------------------------------

// TestStreamServerError verifies that HTTP 500 from Stream returns an error
// immediately (before the channel is opened).
func TestStreamServerError(t *testing.T) {
	srv := newStatusServer(http.StatusInternalServerError, "stream broke")
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	_, err := p.Stream(context.Background(), []sdk.Message{
		{Role: "user", Content: "hi"},
	}, sdk.ModelInfo{ModelID: "llama3"}, sdk.StreamOpts{Timeout: 5})
	if err == nil {
		t.Fatal("expected error for 500 stream response")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected *sdk.AgentError, got %T", err)
	}
	if agentErr.Kind != sdk.ErrorInternal {
		t.Errorf("Kind = %q, want ErrorInternal", agentErr.Kind)
	}
}

// TestStreamRateLimitError verifies that HTTP 429 from Stream returns ErrorRateLimit.
func TestStreamRateLimitError(t *testing.T) {
	srv := newStatusServer(http.StatusTooManyRequests, "rate limited")
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	_, err := p.Stream(context.Background(), []sdk.Message{
		{Role: "user", Content: "hi"},
	}, sdk.ModelInfo{ModelID: "llama3"}, sdk.StreamOpts{Timeout: 5})
	if err == nil {
		t.Fatal("expected error for 429 stream response")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected *sdk.AgentError, got %T", err)
	}
	if agentErr.Kind != sdk.ErrorRateLimit {
		t.Errorf("Kind = %q, want ErrorRateLimit", agentErr.Kind)
	}
}

// ---------------------------------------------------------------------------
// Unit tests: complex schema marshaling
// ---------------------------------------------------------------------------

// TestChatRequestMarshalComplexSchema verifies that a multi-field nested
// schema survives a marshal/unmarshal round-trip intact.
func TestChatRequestMarshalComplexSchema(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":  map[string]any{"type": "string"},
			"age":   map[string]any{"type": "integer"},
			"score": map[string]any{"type": "number"},
			"tags":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
		},
		"required": []string{"name", "age"},
	}
	req := chatRequest{
		Model:    "mistral",
		Messages: []map[string]string{{"role": "user", "content": "describe a person"}},
		Stream:   false,
		Format:   schema,
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	// Round-trip
	var out chatRequest
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	// Re-marshal the Format field and check it still decodes to a map with the right keys
	fmtBytes, _ := json.Marshal(out.Format)
	var fmtObj map[string]any
	if err := json.Unmarshal(fmtBytes, &fmtObj); err != nil {
		t.Fatalf("format round-trip decode error: %v", err)
	}
	if fmtObj["type"] != "object" {
		t.Errorf("format.type = %q, want object", fmtObj["type"])
	}
	props, ok := fmtObj["properties"].(map[string]any)
	if !ok {
		t.Fatal("format.properties should be a map")
	}
	for _, key := range []string{"name", "age", "score", "tags"} {
		if _, exists := props[key]; !exists {
			t.Errorf("format.properties missing key %q", key)
		}
	}
}

// TestCompletePassesComplexSchemaToServer verifies that a complex multi-field
// schema is sent verbatim to the server's format field.
func TestCompletePassesComplexSchemaToServer(t *testing.T) {
	srv, captured := newFakeChatServer(t)
	defer srv.Close()

	p := NewOllamaProvider(srv.URL)
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"firstName": map[string]any{"type": "string"},
			"lastName":  map[string]any{"type": "string"},
			"age":       map[string]any{"type": "integer"},
		},
		"required": []string{"firstName", "lastName"},
	}

	_, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "user", Content: "Describe a person"},
	}, sdk.ModelInfo{ModelID: "llama3"}, sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 10,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(*captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	fmtRaw, ok := body["format"]
	if !ok {
		t.Fatal("format field missing from request")
	}
	// Re-encode to check nested properties survive
	fmtBytes, _ := json.Marshal(fmtRaw)
	var fmtObj map[string]any
	json.Unmarshal(fmtBytes, &fmtObj) //nolint:errcheck
	props, ok := fmtObj["properties"].(map[string]any)
	if !ok {
		t.Fatal("format.properties should be a map")
	}
	if _, ok := props["firstName"]; !ok {
		t.Error("format.properties.firstName missing")
	}
	if _, ok := props["lastName"]; !ok {
		t.Error("format.properties.lastName missing")
	}
}

// ---------------------------------------------------------------------------
// Integration tests — require a live Ollama instance
// ---------------------------------------------------------------------------

// firstModelWithStructured returns the first model that advertises structured
// output support, or the first model overall if none advertise it.
func firstModelWithStructured(models []sdk.ModelInfo) sdk.ModelInfo {
	for _, m := range models {
		if m.SupportsStructured {
			return m
		}
	}
	return models[0]
}

// TestOllamaCompleteStructuredSimple is an integration test that asks a
// simple maths question and expects a JSON response matching a basic schema.
func TestOllamaCompleteStructuredSimple(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	p := NewOllamaProvider("")
	skipIfOllamaUnavailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil || len(models) == 0 {
		t.Skip("no models available")
	}
	model := firstModelWithStructured(models)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "number"},
		},
		"required": []string{"answer"},
	}

	resp, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "system", Content: "Respond only with valid JSON matching the given schema."},
		{Role: "user", Content: "What is 6 multiplied by 7? Respond as JSON with key 'answer'."},
	}, model, sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete structured error: %v", err)
	}
	if resp.Text == "" {
		t.Fatal("structured response text is empty")
	}
	t.Logf("Structured response: %q", resp.Text)

	// The response should be parseable JSON
	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &parsed); err != nil {
		t.Errorf("response is not valid JSON: %v", err)
	}
}

// TestOllamaCompleteStructuredComplex is an integration test using a
// multi-field schema to retrieve a structured person description.
func TestOllamaCompleteStructuredComplex(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	p := NewOllamaProvider("")
	skipIfOllamaUnavailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil || len(models) == 0 {
		t.Skip("no models available")
	}
	model := firstModelWithStructured(models)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":       map[string]any{"type": "string"},
			"occupation": map[string]any{"type": "string"},
			"age":        map[string]any{"type": "integer"},
		},
		"required": []string{"name", "occupation", "age"},
	}

	resp, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "system", Content: "Respond only with valid JSON matching the given schema."},
		{Role: "user", Content: "Invent a fictional person and describe them as JSON with keys: name, occupation, age."},
	}, model, sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete structured complex error: %v", err)
	}
	if resp.Text == "" {
		t.Fatal("structured response text is empty")
	}
	t.Logf("Complex structured response: %q", resp.Text)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &parsed); err != nil {
		t.Errorf("response is not valid JSON: %v", err)
	}
	for _, key := range []string{"name", "occupation", "age"} {
		if _, ok := parsed[key]; !ok {
			t.Errorf("response JSON missing required key %q", key)
		}
	}
}

// ---------------------------------------------------------------------------
// Extra unit: verify Format field is part of the chatRequest struct and the
// json tag is correct (compile-time check via reflection-free marshaling).
// ---------------------------------------------------------------------------

// TestChatRequestFormatTagOmitempty verifies that Format with a zero value
// (empty string) is treated as omitted, while a non-zero value is included.
// This guards against accidentally removing the omitempty tag.
func TestChatRequestFormatTagOmitempty(t *testing.T) {
	// nil interface — omitted
	noFmt := chatRequest{Model: "m", Messages: nil, Stream: false, Format: nil}
	data, _ := json.Marshal(noFmt)
	if bytes.Contains(data, []byte(`"format"`)) {
		t.Error("format key should be absent for nil Format")
	}

	// non-nil — present
	withFmt := chatRequest{Model: "m", Messages: nil, Stream: false, Format: "json"}
	data2, _ := json.Marshal(withFmt)
	if !bytes.Contains(data2, []byte(`"format"`)) {
		t.Error("format key should be present for non-nil Format")
	}
}

