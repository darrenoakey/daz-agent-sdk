package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"google.golang.org/genai"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// skipIfNoGeminiKey skips the test when neither GEMINI_API_KEY nor GOOGLE_API_KEY is set.
func skipIfNoGeminiKey(t *testing.T) {
	t.Helper()
	if geminiAPIKey() == "" {
		t.Skip("GEMINI_API_KEY and GOOGLE_API_KEY not set, skipping integration test")
	}
}

// geminiFlashLite returns the gemini-2.5-flash-lite ModelInfo, suitable for
// fast, low-cost integration tests.
func geminiFlashLite() sdk.ModelInfo {
	return sdk.ModelInfo{
		Provider: "gemini",
		ModelID:  "gemini-2.5-flash-lite",
	}
}

// --- Unit tests: error classification ---

func TestClassifyGeminiStatusCode_RateLimit(t *testing.T) {
	got := classifyGeminiStatusCode(429)
	if got != sdk.ErrorRateLimit {
		t.Errorf("status 429 = %q, want %q", got, sdk.ErrorRateLimit)
	}
}

func TestClassifyGeminiStatusCode_Unauthorized(t *testing.T) {
	got := classifyGeminiStatusCode(401)
	if got != sdk.ErrorAuth {
		t.Errorf("status 401 = %q, want %q", got, sdk.ErrorAuth)
	}
}

func TestClassifyGeminiStatusCode_Forbidden(t *testing.T) {
	got := classifyGeminiStatusCode(403)
	if got != sdk.ErrorAuth {
		t.Errorf("status 403 = %q, want %q", got, sdk.ErrorAuth)
	}
}

func TestClassifyGeminiStatusCode_Timeout(t *testing.T) {
	got := classifyGeminiStatusCode(408)
	if got != sdk.ErrorTimeout {
		t.Errorf("status 408 = %q, want %q", got, sdk.ErrorTimeout)
	}
}

func TestClassifyGeminiStatusCode_BadRequest(t *testing.T) {
	got := classifyGeminiStatusCode(400)
	if got != sdk.ErrorInvalidRequest {
		t.Errorf("status 400 = %q, want %q", got, sdk.ErrorInvalidRequest)
	}
}

func TestClassifyGeminiStatusCode_UnprocessableEntity(t *testing.T) {
	got := classifyGeminiStatusCode(422)
	if got != sdk.ErrorInvalidRequest {
		t.Errorf("status 422 = %q, want %q", got, sdk.ErrorInvalidRequest)
	}
}

func TestClassifyGeminiStatusCode_ServiceUnavailable(t *testing.T) {
	got := classifyGeminiStatusCode(503)
	if got != sdk.ErrorNotAvailable {
		t.Errorf("status 503 = %q, want %q", got, sdk.ErrorNotAvailable)
	}
}

func TestClassifyGeminiStatusCode_Internal(t *testing.T) {
	got := classifyGeminiStatusCode(500)
	if got != sdk.ErrorInternal {
		t.Errorf("status 500 = %q, want %q", got, sdk.ErrorInternal)
	}
}

func TestClassifyGeminiError_APIError(t *testing.T) {
	err := genai.APIError{Code: 429, Message: "quota exceeded"}
	got := classifyGeminiError(err)
	if got != sdk.ErrorRateLimit {
		t.Errorf("APIError 429 = %q, want %q", got, sdk.ErrorRateLimit)
	}
}

func TestClassifyGeminiError_TimeoutMessage(t *testing.T) {
	err := genai.APIError{Code: 200, Message: "context deadline exceeded"}
	got := classifyGeminiError(err)
	// APIError with Code 200 falls through to message-based check via classifyGeminiStatusCode(200)->internal,
	// so verify APIError path is taken first (code=200 -> internal, not timeout).
	// The timeout path is triggered on non-APIError errors with timeout in message.
	if got != sdk.ErrorInternal {
		t.Errorf("APIError Code=200 = %q, want %q", got, sdk.ErrorInternal)
	}
}

func TestClassifyGeminiError_MessageTimeout(t *testing.T) {
	// Wrap as a plain error (not genai.APIError) so the message path triggers.
	errMsg := fmt.Errorf("context deadline exceeded: timeout")
	got := classifyGeminiError(errMsg)
	if got != sdk.ErrorTimeout {
		t.Errorf("message timeout error = %q, want %q", got, sdk.ErrorTimeout)
	}
}

func TestClassifyGeminiError_MessageConnection(t *testing.T) {
	errMsg := fmt.Errorf("dial tcp: connection refused")
	got := classifyGeminiError(errMsg)
	if got != sdk.ErrorNotAvailable {
		t.Errorf("connection refused error = %q, want %q", got, sdk.ErrorNotAvailable)
	}
}

func TestClassifyGeminiError_MessageAPIKey(t *testing.T) {
	errMsg := fmt.Errorf("invalid api key provided")
	got := classifyGeminiError(errMsg)
	if got != sdk.ErrorAuth {
		t.Errorf("api key error = %q, want %q", got, sdk.ErrorAuth)
	}
}

func TestClassifyGeminiError_MessageRateLimit(t *testing.T) {
	errMsg := fmt.Errorf("resource exhausted: quota exceeded")
	got := classifyGeminiError(errMsg)
	if got != sdk.ErrorRateLimit {
		t.Errorf("resource exhausted error = %q, want %q", got, sdk.ErrorRateLimit)
	}
}

// --- Unit tests: message conversion ---

func TestBuildGeminiContents_Empty(t *testing.T) {
	sys, contents := buildGeminiContents(nil)
	if sys != nil {
		t.Errorf("expected nil system instruction for empty input, got %+v", sys)
	}
	if len(contents) != 0 {
		t.Errorf("expected 0 contents, got %d", len(contents))
	}
}

func TestBuildGeminiContents_SingleUser(t *testing.T) {
	messages := []sdk.Message{
		{Role: "user", Content: "Hello"},
	}
	sys, contents := buildGeminiContents(messages)
	if sys != nil {
		t.Error("expected no system instruction")
	}
	if len(contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(contents))
	}
	if contents[0].Role != genai.RoleUser {
		t.Errorf("role = %q, want %q", contents[0].Role, genai.RoleUser)
	}
	if contents[0].Parts[0].Text != "Hello" {
		t.Errorf("text = %q, want Hello", contents[0].Parts[0].Text)
	}
}

func TestBuildGeminiContents_SystemAndUser(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "What is 1+1?"},
	}
	sys, contents := buildGeminiContents(messages)
	if sys == nil {
		t.Fatal("expected system instruction, got nil")
	}
	if !strings.Contains(sys.Parts[0].Text, "Be concise.") {
		t.Errorf("system instruction text = %q, want contains 'Be concise.'", sys.Parts[0].Text)
	}
	if len(contents) != 1 {
		t.Fatalf("expected 1 content entry, got %d", len(contents))
	}
	if contents[0].Role != genai.RoleUser {
		t.Errorf("content role = %q, want user", contents[0].Role)
	}
}

func TestBuildGeminiContents_MultipleSystemsMerged(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "Rule one."},
		{Role: "system", Content: "Rule two."},
		{Role: "user", Content: "Hello"},
	}
	sys, contents := buildGeminiContents(messages)
	if sys == nil {
		t.Fatal("expected system instruction")
	}
	text := sys.Parts[0].Text
	if !strings.Contains(text, "Rule one.") {
		t.Error("merged system should contain 'Rule one.'")
	}
	if !strings.Contains(text, "Rule two.") {
		t.Error("merged system should contain 'Rule two.'")
	}
	if len(contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(contents))
	}
}

func TestBuildGeminiContents_MultiTurn(t *testing.T) {
	messages := []sdk.Message{
		{Role: "user", Content: "Hi"},
		{Role: "assistant", Content: "Hello there"},
		{Role: "user", Content: "How are you?"},
	}
	sys, contents := buildGeminiContents(messages)
	if sys != nil {
		t.Error("expected no system instruction")
	}
	if len(contents) != 3 {
		t.Fatalf("expected 3 contents, got %d", len(contents))
	}
	if contents[0].Role != genai.RoleUser {
		t.Errorf("contents[0].Role = %q, want user", contents[0].Role)
	}
	if contents[1].Role != genai.RoleModel {
		t.Errorf("contents[1].Role = %q, want model", contents[1].Role)
	}
	if contents[2].Role != genai.RoleUser {
		t.Errorf("contents[2].Role = %q, want user", contents[2].Role)
	}
}

func TestBuildGeminiContents_AllRoles(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "System prompt."},
		{Role: "user", Content: "First user turn."},
		{Role: "assistant", Content: "First assistant turn."},
		{Role: "user", Content: "Second user turn."},
		{Role: "assistant", Content: "Second assistant turn."},
	}
	sys, contents := buildGeminiContents(messages)
	if sys == nil {
		t.Fatal("expected system instruction")
	}
	if len(contents) != 4 {
		t.Fatalf("expected 4 contents, got %d", len(contents))
	}
	expected := []string{genai.RoleUser, genai.RoleModel, genai.RoleUser, genai.RoleModel}
	for i, want := range expected {
		if contents[i].Role != want {
			t.Errorf("contents[%d].Role = %q, want %q", i, contents[i].Role, want)
		}
	}
}

// --- Unit tests: model catalog ---

func TestGeminiListModels_Count(t *testing.T) {
	p := NewGeminiProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) != 3 {
		t.Fatalf("expected 3 models, got %d", len(models))
	}
}

func TestGeminiListModels_TierAssignments(t *testing.T) {
	p := NewGeminiProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}

	byID := make(map[string]sdk.ModelInfo)
	for _, m := range models {
		byID[m.ModelID] = m
	}

	pro, ok := byID["gemini-2.5-pro"]
	if !ok {
		t.Fatal("missing gemini-2.5-pro")
	}
	if pro.Tier != sdk.TierHigh {
		t.Errorf("gemini-2.5-pro tier = %q, want high", pro.Tier)
	}

	flash, ok := byID["gemini-2.5-flash"]
	if !ok {
		t.Fatal("missing gemini-2.5-flash")
	}
	if flash.Tier != sdk.TierMedium {
		t.Errorf("gemini-2.5-flash tier = %q, want medium", flash.Tier)
	}

	lite, ok := byID["gemini-2.5-flash-lite"]
	if !ok {
		t.Fatal("missing gemini-2.5-flash-lite")
	}
	if lite.Tier != sdk.TierLow {
		t.Errorf("gemini-2.5-flash-lite tier = %q, want low", lite.Tier)
	}
}

func TestGeminiListModels_Capabilities(t *testing.T) {
	p := NewGeminiProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}

	byID := make(map[string]sdk.ModelInfo)
	for _, m := range models {
		byID[m.ModelID] = m
	}

	// gemini-2.5-pro should have agentic capability and supports_tools=true
	pro := byID["gemini-2.5-pro"]
	hasAgentic := false
	for _, c := range pro.Capabilities {
		if c == sdk.CapabilityAgentic {
			hasAgentic = true
		}
	}
	if !hasAgentic {
		t.Error("gemini-2.5-pro should have agentic capability")
	}
	if !pro.SupportsTools {
		t.Error("gemini-2.5-pro should support tools")
	}

	// gemini-2.5-flash-lite should not support tools
	lite := byID["gemini-2.5-flash-lite"]
	if lite.SupportsTools {
		t.Error("gemini-2.5-flash-lite should not support tools")
	}
}

func TestGeminiListModels_CommonFields(t *testing.T) {
	p := NewGeminiProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	for _, m := range models {
		if m.Provider != "gemini" {
			t.Errorf("model %q has provider %q, want gemini", m.ModelID, m.Provider)
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

// --- Unit tests: provider basics ---

func TestGeminiProviderName(t *testing.T) {
	p := NewGeminiProvider()
	if p.Name() != "gemini" {
		t.Errorf("Name() = %q, want gemini", p.Name())
	}
}

func TestNewGeminiProvider_Default(t *testing.T) {
	p := NewGeminiProvider()
	if p == nil {
		t.Fatal("NewGeminiProvider returned nil")
	}
}

func TestGeminiAvailable_WithoutKey(t *testing.T) {
	orig1 := os.Getenv("GEMINI_API_KEY")
	orig2 := os.Getenv("GOOGLE_API_KEY")

	if orig1 == "" && orig2 == "" {
		// Both already absent: Available() must return false, no error.
		p := NewGeminiProvider()
		ok, err := p.Available(context.Background())
		if err != nil {
			t.Fatalf("Available should not error when key missing, got: %v", err)
		}
		if ok {
			t.Error("Available should return false when API keys are absent")
		}
		return
	}

	// At least one key is set — temporarily clear both.
	os.Unsetenv("GEMINI_API_KEY")
	os.Unsetenv("GOOGLE_API_KEY")
	defer func() {
		if orig1 != "" {
			os.Setenv("GEMINI_API_KEY", orig1)
		} else {
			os.Unsetenv("GEMINI_API_KEY")
		}
		if orig2 != "" {
			os.Setenv("GOOGLE_API_KEY", orig2)
		} else {
			os.Unsetenv("GOOGLE_API_KEY")
		}
	}()

	p := NewGeminiProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available should not error when keys missing, got: %v", err)
	}
	if ok {
		t.Error("Available should return false when API keys are missing")
	}
}

func TestGeminiAPIKeyDetection_GeminiVar(t *testing.T) {
	orig1 := os.Getenv("GEMINI_API_KEY")
	orig2 := os.Getenv("GOOGLE_API_KEY")
	os.Unsetenv("GEMINI_API_KEY")
	os.Unsetenv("GOOGLE_API_KEY")
	defer func() {
		if orig1 != "" {
			os.Setenv("GEMINI_API_KEY", orig1)
		} else {
			os.Unsetenv("GEMINI_API_KEY")
		}
		if orig2 != "" {
			os.Setenv("GOOGLE_API_KEY", orig2)
		} else {
			os.Unsetenv("GOOGLE_API_KEY")
		}
	}()

	os.Setenv("GEMINI_API_KEY", "test-key-1")
	got := geminiAPIKey()
	if got != "test-key-1" {
		t.Errorf("geminiAPIKey() = %q, want test-key-1", got)
	}
}

func TestGeminiAPIKeyDetection_GoogleVar(t *testing.T) {
	orig1 := os.Getenv("GEMINI_API_KEY")
	orig2 := os.Getenv("GOOGLE_API_KEY")
	os.Unsetenv("GEMINI_API_KEY")
	os.Unsetenv("GOOGLE_API_KEY")
	defer func() {
		if orig1 != "" {
			os.Setenv("GEMINI_API_KEY", orig1)
		} else {
			os.Unsetenv("GEMINI_API_KEY")
		}
		if orig2 != "" {
			os.Setenv("GOOGLE_API_KEY", orig2)
		} else {
			os.Unsetenv("GOOGLE_API_KEY")
		}
	}()

	os.Setenv("GOOGLE_API_KEY", "test-key-2")
	got := geminiAPIKey()
	if got != "test-key-2" {
		t.Errorf("geminiAPIKey() = %q, want test-key-2", got)
	}
}

func TestGeminiAPIKeyDetection_GeminiTakesPrecedence(t *testing.T) {
	orig1 := os.Getenv("GEMINI_API_KEY")
	orig2 := os.Getenv("GOOGLE_API_KEY")
	os.Setenv("GEMINI_API_KEY", "gemini-key")
	os.Setenv("GOOGLE_API_KEY", "google-key")
	defer func() {
		if orig1 != "" {
			os.Setenv("GEMINI_API_KEY", orig1)
		} else {
			os.Unsetenv("GEMINI_API_KEY")
		}
		if orig2 != "" {
			os.Setenv("GOOGLE_API_KEY", orig2)
		} else {
			os.Unsetenv("GOOGLE_API_KEY")
		}
	}()

	got := geminiAPIKey()
	if got != "gemini-key" {
		t.Errorf("GEMINI_API_KEY should take precedence, got %q", got)
	}
}

// --- Integration tests (skipped if no API key) ---

func TestGeminiAvailable_WithKey(t *testing.T) {
	skipIfNoGeminiKey(t)
	p := NewGeminiProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available error: %v", err)
	}
	if !ok {
		t.Error("expected Available to return true with valid API key")
	}
}

func TestGeminiComplete_BasicText(t *testing.T) {
	skipIfNoGeminiKey(t)
	p := NewGeminiProvider()

	messages := []sdk.Message{
		{Role: "user", Content: "Reply with exactly one word: pong"},
	}
	resp, err := p.Complete(context.Background(), messages, geminiFlashLite(), sdk.CompleteOpts{
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete returned empty text")
	}
	if resp.ModelUsed.ModelID != "gemini-2.5-flash-lite" {
		t.Errorf("ModelUsed.ModelID = %q, want gemini-2.5-flash-lite", resp.ModelUsed.ModelID)
	}
	if resp.ConversationID.String() == "" {
		t.Error("ConversationID should be set")
	}
	if resp.TurnID.String() == "" {
		t.Error("TurnID should be set")
	}
	t.Logf("Complete response: %q", resp.Text)
}

func TestGeminiComplete_StructuredOutputSimple(t *testing.T) {
	skipIfNoGeminiKey(t)
	p := NewGeminiProvider()

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"result": map[string]any{
				"type":        "integer",
				"description": "The numeric answer",
			},
		},
		"required": []any{"result"},
	}

	messages := []sdk.Message{
		{Role: "user", Content: "What is 3 + 4? Respond in JSON with a 'result' field."},
	}
	resp, err := p.Complete(context.Background(), messages, geminiFlashLite(), sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 60,
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

func TestGeminiComplete_StructuredOutputComplex(t *testing.T) {
	skipIfNoGeminiKey(t)
	p := NewGeminiProvider()

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":        "string",
				"description": "The name of the capital city",
			},
			"country": map[string]any{
				"type":        "string",
				"description": "The country name",
			},
			"population_millions": map[string]any{
				"type":        "number",
				"description": "Approximate population in millions",
			},
		},
		"required": []any{"name", "country", "population_millions"},
	}

	messages := []sdk.Message{
		{Role: "user", Content: "Give me information about France's capital city in JSON format."},
	}
	resp, err := p.Complete(context.Background(), messages, geminiFlashLite(), sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete with complex schema error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete returned empty text")
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &parsed); err != nil {
		t.Fatalf("structured output is not valid JSON: %v, text: %q", err, resp.Text)
	}
	for _, field := range []string{"name", "country", "population_millions"} {
		if _, ok := parsed[field]; !ok {
			t.Errorf("parsed JSON missing required field %q, got keys: %v", field, keys(parsed))
		}
	}
	t.Logf("Complex structured result: %v", parsed)
}

func TestGeminiStream_BasicText(t *testing.T) {
	skipIfNoGeminiKey(t)
	p := NewGeminiProvider()

	messages := []sdk.Message{
		{Role: "user", Content: "Reply with exactly one word: ping"},
	}

	ch, err := p.Stream(context.Background(), messages, geminiFlashLite(), sdk.StreamOpts{
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

func TestGeminiComplete_SystemMessage(t *testing.T) {
	skipIfNoGeminiKey(t)
	p := NewGeminiProvider()

	messages := []sdk.Message{
		{Role: "system", Content: "You only ever reply with the single word MANGO."},
		{Role: "user", Content: "What is your favourite fruit?"},
	}
	resp, err := p.Complete(context.Background(), messages, geminiFlashLite(), sdk.CompleteOpts{
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if !strings.Contains(strings.ToUpper(resp.Text), "MANGO") {
		t.Errorf("expected MANGO in response, got: %q", resp.Text)
	}
	t.Logf("System-prompted response: %q", resp.Text)
}

// --- Helpers ---

// keys returns the map keys as a slice for diagnostic output.
func keys(m map[string]any) []string {
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	return result
}

