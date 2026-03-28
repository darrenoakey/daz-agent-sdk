package provider

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"testing"
	"time"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// ── Claude CLI discovery ─────────────────────────────────────────

func TestFindClaudeCLI(t *testing.T) {
	path, err := findClaudeCLI()
	if err != nil {
		t.Skipf("claude CLI not installed: %v", err)
	}
	if path == "" {
		t.Error("findClaudeCLI returned empty path")
	}
	t.Logf("found claude CLI at: %s", path)
}

func TestFindClaudeCLI_MatchesWhich(t *testing.T) {
	path, err := findClaudeCLI()
	if err != nil {
		t.Skip("claude CLI not installed")
	}
	whichPath, err := exec.LookPath("claude")
	if err != nil {
		t.Skip("claude not on PATH")
	}
	if path != whichPath {
		t.Logf("findClaudeCLI=%s, which=%s (may differ if found in fallback path)", path, whichPath)
	}
}

// ── Provider basics ──────────────────────────────────────────────

func TestClaudeName(t *testing.T) {
	p := NewClaudeProvider()
	if p.Name() != "claude" {
		t.Errorf("Name() = %q, want claude", p.Name())
	}
}

func TestClaudeAvailable(t *testing.T) {
	p := NewClaudeProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available error: %v", err)
	}
	_, cliErr := findClaudeCLI()
	expected := cliErr == nil
	if ok != expected {
		t.Errorf("Available() = %v, want %v", ok, expected)
	}
}

func TestClaudeListModels(t *testing.T) {
	p := NewClaudeProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) != 3 {
		t.Errorf("len(models) = %d, want 3", len(models))
	}
	ids := map[string]bool{}
	for _, m := range models {
		ids[m.ModelID] = true
		if m.Provider != "claude" {
			t.Errorf("model %s has provider %q, want claude", m.ModelID, m.Provider)
		}
	}
	for _, expected := range []string{"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"} {
		if !ids[expected] {
			t.Errorf("missing model %s", expected)
		}
	}
}

func TestClaudeListModels_IsCopy(t *testing.T) {
	p := NewClaudeProvider()
	m1, _ := p.ListModels(context.Background())
	m2, _ := p.ListModels(context.Background())
	m1[0].DisplayName = "MUTATED"
	if m2[0].DisplayName == "MUTATED" {
		t.Error("ListModels should return a copy, not a reference to the internal slice")
	}
}

// ── Model catalog ────────────────────────────────────────────────

func TestClaudeModelTiers(t *testing.T) {
	p := NewClaudeProvider()
	models, _ := p.ListModels(context.Background())
	tierMap := map[string]sdk.Tier{}
	for _, m := range models {
		tierMap[m.ModelID] = m.Tier
	}
	if tierMap["claude-opus-4-6"] != sdk.TierHigh {
		t.Errorf("opus tier = %s, want high", tierMap["claude-opus-4-6"])
	}
	if tierMap["claude-sonnet-4-6"] != sdk.TierMedium {
		t.Errorf("sonnet tier = %s, want medium", tierMap["claude-sonnet-4-6"])
	}
	if tierMap["claude-haiku-4-5-20251001"] != sdk.TierLow {
		t.Errorf("haiku tier = %s, want low", tierMap["claude-haiku-4-5-20251001"])
	}
}

func TestClaudeModelCapabilities(t *testing.T) {
	p := NewClaudeProvider()
	models, _ := p.ListModels(context.Background())
	for _, m := range models {
		if !m.SupportsStreaming {
			t.Errorf("model %s should support streaming", m.ModelID)
		}
		if !m.SupportsStructured {
			t.Errorf("model %s should support structured output", m.ModelID)
		}
		if !m.SupportsConversation {
			t.Errorf("model %s should support conversation", m.ModelID)
		}
		if !m.SupportsTools {
			t.Errorf("model %s should support tools", m.ModelID)
		}
	}
}

// ── Prompt building ──────────────────────────────────────────────

func TestBuildPrompt_Simple(t *testing.T) {
	messages := []sdk.Message{{Role: "user", Content: "hello"}}
	system, prompt := buildPrompt(messages)
	if system != "" {
		t.Errorf("system = %q, want empty", system)
	}
	if prompt != "hello" {
		t.Errorf("prompt = %q, want hello", prompt)
	}
}

func TestBuildPrompt_WithSystem(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "you are helpful"},
		{Role: "user", Content: "hi"},
	}
	system, prompt := buildPrompt(messages)
	if system != "you are helpful" {
		t.Errorf("system = %q, want 'you are helpful'", system)
	}
	if prompt != "hi" {
		t.Errorf("prompt = %q, want 'hi'", prompt)
	}
}

func TestBuildPrompt_MultipleSystems(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "rule 1"},
		{Role: "system", Content: "rule 2"},
		{Role: "user", Content: "hello"},
	}
	system, _ := buildPrompt(messages)
	if system != "rule 1\nrule 2" {
		t.Errorf("system = %q, want 'rule 1\\nrule 2'", system)
	}
}

func TestBuildPrompt_MultiTurn(t *testing.T) {
	messages := []sdk.Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "response"},
		{Role: "user", Content: "second"},
	}
	_, prompt := buildPrompt(messages)
	if !strings.Contains(prompt, "first") || !strings.Contains(prompt, "response") || !strings.Contains(prompt, "second") {
		t.Errorf("prompt missing expected parts: %q", prompt)
	}
	if !strings.Contains(prompt, "[Previous assistant response]") {
		t.Errorf("prompt should prefix assistant messages: %q", prompt)
	}
}

func TestBuildPrompt_NoMessages(t *testing.T) {
	system, prompt := buildPrompt(nil)
	if system != "" || prompt != "" {
		t.Errorf("empty messages should produce empty strings, got system=%q prompt=%q", system, prompt)
	}
}

// ── Error classification ─────────────────────────────────────────

func TestClassifyClaudeError_RateLimit(t *testing.T) {
	for _, msg := range []string{"rate_limit exceeded", "HTTP 429", "server overloaded"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorRateLimit {
			t.Errorf("classifyClaudeError(%q) = %s, want rate_limit", msg, got)
		}
	}
}

func TestClassifyClaudeError_Auth(t *testing.T) {
	for _, msg := range []string{"401 Unauthorized", "403 Forbidden", "auth failed"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorAuth {
			t.Errorf("classifyClaudeError(%q) = %s, want auth", msg, got)
		}
	}
}

func TestClassifyClaudeError_Timeout(t *testing.T) {
	for _, msg := range []string{"request timed out", "timeout exceeded"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorTimeout {
			t.Errorf("classifyClaudeError(%q) = %s, want timeout", msg, got)
		}
	}
}

func TestClassifyClaudeError_InvalidRequest(t *testing.T) {
	for _, msg := range []string{"400 bad request invalid", "invalid parameters"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorInvalidRequest {
			t.Errorf("classifyClaudeError(%q) = %s, want invalid_request", msg, got)
		}
	}
}

func TestClassifyClaudeError_NotAvailable(t *testing.T) {
	for _, msg := range []string{"claude not found", "CLI not installed"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorNotAvailable {
			t.Errorf("classifyClaudeError(%q) = %s, want not_available", msg, got)
		}
	}
}

func TestClassifyClaudeError_Internal(t *testing.T) {
	if got := classifyClaudeError(fmt.Errorf("something weird")); got != sdk.ErrorInternal {
		t.Errorf("classifyClaudeError('something weird') = %s, want internal", got)
	}
}

// ── CLAUDECODE env stripping ─────────────────────────────────────

func TestStripClaudeCodeEnv(t *testing.T) {
	env := stripClaudeCodeEnv()
	for _, e := range env {
		if strings.HasPrefix(e, "CLAUDECODE=") {
			t.Error("stripClaudeCodeEnv should remove CLAUDECODE")
		}
	}
}

// ── Integration tests (require claude CLI) ───────────────────────

func skipIfNoClaudeCLI(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping claude CLI test in short mode")
	}
	if _, err := findClaudeCLI(); err != nil {
		t.Skipf("claude CLI not available: %v", err)
	}
}

func haikuModel() sdk.ModelInfo {
	return claudeModels[2] // haiku
}

func TestClaudeComplete_BasicText(t *testing.T) {
	skipIfNoClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "What is 2+2? Reply with just the number."}}
	resp, err := p.Complete(context.Background(), messages, haikuModel(), sdk.CompleteOpts{Timeout: 30})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Text == "" {
		t.Error("expected non-empty response")
	}
	if !strings.Contains(resp.Text, "4") {
		t.Errorf("expected '4' in response, got: %s", resp.Text)
	}
	t.Logf("Response: %s", resp.Text)
}

func TestClaudeComplete_SystemMessage(t *testing.T) {
	skipIfNoClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{
		{Role: "system", Content: "Always respond in exactly one word."},
		{Role: "user", Content: "Say hello."},
	}
	resp, err := p.Complete(context.Background(), messages, haikuModel(), sdk.CompleteOpts{Timeout: 30})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Text == "" {
		t.Error("expected non-empty response")
	}
	t.Logf("Response: %s", resp.Text)
}

func TestClaudeComplete_StructuredOutput(t *testing.T) {
	skipIfNoClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "What is 10 + 5?"}}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "integer"},
		},
		"required": []string{"answer"},
	}

	resp, err := p.Complete(context.Background(), messages, haikuModel(), sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	t.Logf("Structured response: %s", resp.Text)
}

func TestClaudeStream_BasicText(t *testing.T) {
	skipIfNoClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "Say hello in one word."}}
	ch, err := p.Stream(context.Background(), messages, haikuModel(), sdk.StreamOpts{Timeout: 30})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var chunks []string
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		chunks = append(chunks, chunk.Text)
	}
	fullText := strings.Join(chunks, "")
	if fullText == "" {
		t.Error("expected non-empty streamed text")
	}
	t.Logf("Streamed %d chunks: %s", len(chunks), fullText)
}

func TestClaudeComplete_ModelUsed(t *testing.T) {
	skipIfNoClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "Say hi."}}
	resp, err := p.Complete(context.Background(), messages, haikuModel(), sdk.CompleteOpts{Timeout: 30})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.ModelUsed.ModelID != "claude-haiku-4-5-20251001" {
		t.Errorf("ModelUsed.ModelID = %q, want claude-haiku-4-5-20251001", resp.ModelUsed.ModelID)
	}
	if resp.ConversationID.String() == "00000000-0000-0000-0000-000000000000" {
		t.Error("ConversationID should be non-zero")
	}
}

func TestClaudeStream_Timeout(t *testing.T) {
	skipIfNoClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "Write a very long essay."}}
	ch, err := p.Stream(context.Background(), messages, haikuModel(), sdk.StreamOpts{Timeout: 0.001})
	if err != nil {
		return // timeout on start is fine
	}
	timeout := time.After(5 * time.Second)
	for {
		select {
		case _, ok := <-ch:
			if !ok {
				return
			}
		case <-timeout:
			t.Error("stream did not terminate within timeout")
			return
		}
	}
}
