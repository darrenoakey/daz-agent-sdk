package dazagentsdk

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfigDefaults(t *testing.T) {
	// Load from a nonexistent path to get pure defaults
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify all 6 tiers exist
	expectedTiers := []string{"very_high", "high", "medium", "low", "free_fast", "free_thinking"}
	for _, tier := range expectedTiers {
		if _, ok := cfg.Tiers[tier]; !ok {
			t.Errorf("missing default tier %q", tier)
		}
	}
}

func TestDefaultTierChains(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tests := []struct {
		tier     Tier
		wantLen  int
		wantFirst string
	}{
		{TierVeryHigh, 3, "claude:claude-opus-4-6"},
		{TierHigh, 3, "claude:claude-opus-4-6"},
		{TierMedium, 3, "claude:claude-sonnet-4-6"},
		{TierLow, 3, "claude:claude-haiku-4-5-20251001"},
		{TierFreeFast, 1, "ollama:qwen3-8b"},
		{TierFreeThinking, 2, "ollama:qwen3-30b-32k"},
	}

	for _, tt := range tests {
		chain := GetTierChain(tt.tier, cfg)
		if len(chain) != tt.wantLen {
			t.Errorf("GetTierChain(%v) len = %d, want %d", tt.tier, len(chain), tt.wantLen)
			continue
		}
		if chain[0] != tt.wantFirst {
			t.Errorf("GetTierChain(%v)[0] = %q, want %q", tt.tier, chain[0], tt.wantFirst)
		}
	}
}

func TestDefaultImageSteps(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tests := []struct {
		tier      Tier
		wantSteps int
	}{
		{TierVeryHigh, 8},
		{TierHigh, 3},
		{TierMedium, 3},
		{TierLow, 2},
		// free_fast has no image tier, falls back to medium
		{TierFreeFast, 3},
	}

	for _, tt := range tests {
		steps := GetImageSteps(tt.tier, cfg)
		if steps != tt.wantSteps {
			t.Errorf("GetImageSteps(%v) = %d, want %d", tt.tier, steps, tt.wantSteps)
		}
	}
}

func TestDefaultImageModel(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Image.Model != "z-image-turbo" {
		t.Errorf("Image.Model = %q, want %q", cfg.Image.Model, "z-image-turbo")
	}
}

func TestDefaultProviders(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedProviders := []string{"claude", "codex", "gemini", "ollama"}
	for _, p := range expectedProviders {
		if _, ok := cfg.Providers[p]; !ok {
			t.Errorf("missing default provider %q", p)
		}
	}

	// Check claude has bypassPermissions
	if cfg.Providers["claude"]["permission_mode"] != "bypassPermissions" {
		t.Errorf("claude permission_mode = %v, want bypassPermissions", cfg.Providers["claude"]["permission_mode"])
	}

	// Check ollama has base_url
	if cfg.Providers["ollama"]["base_url"] != "http://localhost:11434" {
		t.Errorf("ollama base_url = %v, want http://localhost:11434", cfg.Providers["ollama"]["base_url"])
	}
}

func TestDefaultLogging(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Logging.Directory != "~/.daz-agent-sdk/logs" {
		t.Errorf("Logging.Directory = %q", cfg.Logging.Directory)
	}
	if cfg.Logging.Level != "info" {
		t.Errorf("Logging.Level = %q", cfg.Logging.Level)
	}
	if cfg.Logging.RetentionDays != 30 {
		t.Errorf("Logging.RetentionDays = %d", cfg.Logging.RetentionDays)
	}
}

func TestDefaultFallback(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Fallback.SingleShot.Strategy != "immediate_cascade" {
		t.Errorf("Fallback.SingleShot.Strategy = %q", cfg.Fallback.SingleShot.Strategy)
	}
	if cfg.Fallback.Conversation.Strategy != "backoff_then_cascade" {
		t.Errorf("Fallback.Conversation.Strategy = %q", cfg.Fallback.Conversation.Strategy)
	}
	if cfg.Fallback.Conversation.MaxBackoffSeconds != 60 {
		t.Errorf("Fallback.Conversation.MaxBackoffSeconds = %d", cfg.Fallback.Conversation.MaxBackoffSeconds)
	}
	if cfg.Fallback.Conversation.SummariseWith != "free_thinking" {
		t.Errorf("Fallback.Conversation.SummariseWith = %q", cfg.Fallback.Conversation.SummariseWith)
	}
}

func TestDefaultTTS(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cfg.TTS.Default) != 1 || cfg.TTS.Default[0] != "local:qwen3-tts" {
		t.Errorf("TTS.Default = %v, want [local:qwen3-tts]", cfg.TTS.Default)
	}
	if len(cfg.TTS.Voices) != 2 {
		t.Errorf("TTS.Voices len = %d, want 2", len(cfg.TTS.Voices))
	}
	gary, ok := cfg.TTS.Voices["gary"]
	if !ok {
		t.Fatal("missing gary voice")
	}
	if gary.Description != "British newsreader" {
		t.Errorf("gary description = %q", gary.Description)
	}
}

func TestDefaultTransparentPostProcess(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Image.TransparentPostProcess != "birefnet" {
		t.Errorf("Image.TransparentPostProcess = %q, want birefnet", cfg.Image.TransparentPostProcess)
	}
}

func TestCustomYAMLLoading(t *testing.T) {
	dir := t.TempDir()
	configFile := filepath.Join(dir, "config.yaml")

	yamlContent := `
tiers:
  high:
    - custom:model-a
    - custom:model-b
  low:
    - ollama:tiny

providers:
  custom:
    api_key: "test-key"

image:
  model: "custom-model"
  tiers:
    high:
      steps: 10
  transparent:
    post_process: "none"

logging:
  level: "debug"
  retention_days: 7
`
	if err := os.WriteFile(configFile, []byte(yamlContent), 0644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	cfg, err := LoadConfig(configFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Custom tier overrides default
	highChain := GetTierChain(TierHigh, cfg)
	if len(highChain) != 2 || highChain[0] != "custom:model-a" {
		t.Errorf("high chain = %v, want [custom:model-a custom:model-b]", highChain)
	}

	// Default tiers still present for ones not overridden
	mediumChain := GetTierChain(TierMedium, cfg)
	if len(mediumChain) != 3 {
		t.Errorf("medium chain should have defaults, got len %d", len(mediumChain))
	}

	// Custom provider present
	if cfg.Providers["custom"]["api_key"] != "test-key" {
		t.Errorf("custom provider api_key = %v", cfg.Providers["custom"]["api_key"])
	}

	// Default providers still present
	if _, ok := cfg.Providers["claude"]; !ok {
		t.Error("default claude provider should still be present")
	}

	// Custom image model
	if cfg.Image.Model != "custom-model" {
		t.Errorf("Image.Model = %q, want custom-model", cfg.Image.Model)
	}

	// Custom image steps for high
	if GetImageSteps(TierHigh, cfg) != 10 {
		t.Errorf("custom high steps = %d, want 10", GetImageSteps(TierHigh, cfg))
	}

	// Default image steps still present for medium
	if GetImageSteps(TierMedium, cfg) != 3 {
		t.Errorf("default medium steps = %d, want 3", GetImageSteps(TierMedium, cfg))
	}

	// Custom transparent post process
	if cfg.Image.TransparentPostProcess != "none" {
		t.Errorf("TransparentPostProcess = %q, want none", cfg.Image.TransparentPostProcess)
	}

	// Custom logging
	if cfg.Logging.Level != "debug" {
		t.Errorf("Logging.Level = %q, want debug", cfg.Logging.Level)
	}
	if cfg.Logging.RetentionDays != 7 {
		t.Errorf("Logging.RetentionDays = %d, want 7", cfg.Logging.RetentionDays)
	}
}

func TestGetTierChainUnknownTier(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	chain := GetTierChain(Tier("nonexistent"), cfg)
	if chain != nil {
		t.Errorf("expected nil for unknown tier, got %v", chain)
	}
}

func TestGetImageStepsUnknownTier(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// free_thinking has no image tier, should fall back to medium (3)
	steps := GetImageSteps(TierFreeThinking, cfg)
	if steps != 3 {
		t.Errorf("GetImageSteps(free_thinking) = %d, want 3 (medium fallback)", steps)
	}
}

func TestGetTierChainReturnsCopy(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	chain1 := GetTierChain(TierHigh, cfg)
	chain2 := GetTierChain(TierHigh, cfg)

	// Mutate chain1, chain2 should be unaffected
	chain1[0] = "modified"
	if chain2[0] == "modified" {
		t.Error("GetTierChain should return a copy, not a reference")
	}
}
