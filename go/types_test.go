package dazagentsdk

import (
	"testing"
)

func TestTierValues(t *testing.T) {
	tests := []struct {
		tier Tier
		want string
	}{
		{TierVeryHigh, "very_high"},
		{TierHigh, "high"},
		{TierMedium, "medium"},
		{TierLow, "low"},
		{TierFreeFast, "free_fast"},
		{TierFreeThinking, "free_thinking"},
	}
	for _, tt := range tests {
		if string(tt.tier) != tt.want {
			t.Errorf("Tier %v = %q, want %q", tt.tier, string(tt.tier), tt.want)
		}
	}
}

func TestCapabilityValues(t *testing.T) {
	tests := []struct {
		cap  Capability
		want string
	}{
		{CapabilityText, "text"},
		{CapabilityStructured, "structured"},
		{CapabilityAgentic, "agentic"},
		{CapabilityImage, "image"},
		{CapabilityTTS, "tts"},
		{CapabilitySTT, "stt"},
	}
	for _, tt := range tests {
		if string(tt.cap) != tt.want {
			t.Errorf("Capability %v = %q, want %q", tt.cap, string(tt.cap), tt.want)
		}
	}
}

func TestErrorKindValues(t *testing.T) {
	tests := []struct {
		kind ErrorKind
		want string
	}{
		{ErrorRateLimit, "rate_limit"},
		{ErrorAuth, "auth"},
		{ErrorTimeout, "timeout"},
		{ErrorInvalidRequest, "invalid_request"},
		{ErrorNotAvailable, "not_available"},
		{ErrorInternal, "internal"},
	}
	for _, tt := range tests {
		if string(tt.kind) != tt.want {
			t.Errorf("ErrorKind %v = %q, want %q", tt.kind, string(tt.kind), tt.want)
		}
	}
}

func TestModelInfoQualifiedName(t *testing.T) {
	m := ModelInfo{
		Provider:    "claude",
		ModelID:     "claude-opus-4-6",
		DisplayName: "Claude Opus",
		Tier:        TierHigh,
	}
	want := "claude:claude-opus-4-6"
	if got := m.QualifiedName(); got != want {
		t.Errorf("QualifiedName() = %q, want %q", got, want)
	}
}

func TestMessageToDict(t *testing.T) {
	t.Run("without metadata", func(t *testing.T) {
		m := Message{Role: "user", Content: "hello"}
		d := m.ToDict()
		if d["role"] != "user" {
			t.Errorf("role = %v, want user", d["role"])
		}
		if d["content"] != "hello" {
			t.Errorf("content = %v, want hello", d["content"])
		}
		if _, ok := d["metadata"]; ok {
			t.Error("metadata should not be present when empty")
		}
	})

	t.Run("with metadata", func(t *testing.T) {
		m := Message{
			Role:     "assistant",
			Content:  "hi",
			Metadata: map[string]any{"key": "val"},
		}
		d := m.ToDict()
		if d["role"] != "assistant" {
			t.Errorf("role = %v, want assistant", d["role"])
		}
		meta, ok := d["metadata"].(map[string]any)
		if !ok {
			t.Fatal("metadata should be present")
		}
		if meta["key"] != "val" {
			t.Errorf("metadata[key] = %v, want val", meta["key"])
		}
	})
}

func TestAgentError(t *testing.T) {
	err := NewAgentError("something broke", ErrorInternal, nil)
	if err.Error() != "something broke" {
		t.Errorf("Error() = %q, want %q", err.Error(), "something broke")
	}
	if err.Kind != ErrorInternal {
		t.Errorf("Kind = %v, want %v", err.Kind, ErrorInternal)
	}
	if len(err.Attempts) != 0 {
		t.Errorf("Attempts should be empty, got %d", len(err.Attempts))
	}

	d := err.ToDict()
	if d["error"] != "something broke" {
		t.Errorf("ToDict error = %v", d["error"])
	}
	if d["kind"] != "internal" {
		t.Errorf("ToDict kind = %v", d["kind"])
	}
}

func TestAgentErrorWithAttempts(t *testing.T) {
	attempts := []map[string]any{
		{"provider": "claude", "error": "rate limited"},
	}
	err := NewAgentError("all failed", ErrorRateLimit, attempts)
	if len(err.Attempts) != 1 {
		t.Fatalf("Attempts = %d, want 1", len(err.Attempts))
	}
	if err.Attempts[0]["provider"] != "claude" {
		t.Errorf("attempt provider = %v", err.Attempts[0]["provider"])
	}
}

func TestParseJSONFromLLM_Plain(t *testing.T) {
	input := `{"name": "test", "value": 42}`
	result, err := ParseJSONFromLLM(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["name"] != "test" {
		t.Errorf("name = %v, want test", result["name"])
	}
	if result["value"] != float64(42) {
		t.Errorf("value = %v, want 42", result["value"])
	}
}

func TestParseJSONFromLLM_WithFences(t *testing.T) {
	input := "```json\n{\"key\": \"val\"}\n```"
	result, err := ParseJSONFromLLM(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["key"] != "val" {
		t.Errorf("key = %v, want val", result["key"])
	}
}

func TestParseJSONFromLLM_WithFencesNoLang(t *testing.T) {
	input := "```\n{\"key\": \"val\"}\n```"
	result, err := ParseJSONFromLLM(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["key"] != "val" {
		t.Errorf("key = %v, want val", result["key"])
	}
}

func TestParseJSONFromLLM_WithProse(t *testing.T) {
	input := "Here is the result of my analysis:\n\n{\"answer\": \"yes\"}"
	result, err := ParseJSONFromLLM(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["answer"] != "yes" {
		t.Errorf("answer = %v, want yes", result["answer"])
	}
}

func TestParseJSONFromLLM_ProseWithFencedBlock(t *testing.T) {
	input := "I analyzed the data and here are my findings:\n\n```json\n{\"status\": \"ok\"}\n```\n\nThat's all."
	result, err := ParseJSONFromLLM(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["status"] != "ok" {
		t.Errorf("status = %v, want ok", result["status"])
	}
}

func TestParseJSONFromLLM_NoJSON(t *testing.T) {
	input := "This is just plain text with no JSON at all."
	_, err := ParseJSONFromLLM(input)
	if err == nil {
		t.Fatal("expected error for text with no JSON")
	}
}

func TestParseJSONFromLLM_Whitespace(t *testing.T) {
	input := "  \n  {\"trimmed\": true}  \n  "
	result, err := ParseJSONFromLLM(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["trimmed"] != true {
		t.Errorf("trimmed = %v, want true", result["trimmed"])
	}
}
