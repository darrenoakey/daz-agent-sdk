package dazagentsdk

import (
	"context"
	"testing"
	"time"
)

func TestNewAgent_Defaults(t *testing.T) {
	agent := NewAgent(nil)
	if agent == nil {
		t.Fatal("NewAgent(nil) returned nil")
	}
	if agent.config == nil {
		t.Fatal("Agent.config is nil")
	}
}

func TestNewAgent_WithConfig(t *testing.T) {
	cfg := &Config{}
	cfg.applyDefaults()
	agent := NewAgent(cfg)
	if agent.config != cfg {
		t.Error("Agent.config should be the provided config")
	}
}

func TestAgent_ConversationReturnsConversation(t *testing.T) {
	agent := NewAgent(nil)
	conv := agent.Conversation("test-conv", WithTier(TierLow))
	defer conv.Close()

	if conv.Name() != "test-conv" {
		t.Errorf("Name() = %q, want %q", conv.Name(), "test-conv")
	}
	if conv.tier != TierLow {
		t.Errorf("tier = %q, want %q", conv.tier, TierLow)
	}
}

func TestAgent_ModelsReturnsModels(t *testing.T) {
	agent := NewAgent(nil)
	ctx := context.Background()

	models, err := agent.Models(ctx)
	if err != nil {
		t.Fatalf("Models() error: %v", err)
	}

	// With default config and no providers loaded, models come from tier
	// chain resolution (placeholder ModelInfo objects)
	if len(models) == 0 {
		t.Log("No models returned (expected when no providers registered)")
	}
}

func TestAgent_ModelsWithTierFilter(t *testing.T) {
	// Clear model cache to avoid cross-test contamination
	RefreshProviders()
	defer RefreshProviders()

	agent := NewAgent(nil)
	ctx := context.Background()

	// Use free_thinking tier which has unique models not shared with other tiers
	models, err := agent.Models(ctx, WithModelsTier(TierFreeThinking))
	if err != nil {
		t.Fatalf("Models() error: %v", err)
	}

	// Verify we get the expected models from the free_thinking tier chain
	if len(models) == 0 {
		t.Log("No models returned (expected when no providers registered)")
		return
	}

	for _, m := range models {
		if m.Tier != TierFreeThinking {
			t.Errorf("model %s has tier %q, want %q", m.QualifiedName(), m.Tier, TierFreeThinking)
		}
	}
}

func TestDefaultAgent(t *testing.T) {
	if Default == nil {
		t.Fatal("Default agent is nil")
	}
}

// TestAgent_AskOllama tests Ask against a real Ollama instance.
// Skipped with -short since it requires a running Ollama server.
func TestAgent_AskOllama(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Ollama integration test in short mode")
	}

	RegisterProviderFactory("ollama", func(cfg *Config) Provider {
		baseURL := "http://localhost:11434"
		if cfg != nil {
			if p, ok := cfg.Providers["ollama"]; ok {
				if u, ok := p["base_url"].(string); ok && u != "" {
					baseURL = u
				}
			}
		}
		return newTestOllamaProvider(baseURL)
	})
	defer RefreshProviders()

	agent := NewAgent(nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	resp, err := agent.Ask(ctx, "Say hello in exactly one word.",
		WithAskTier(TierFreeFast),
		WithAskProvider("ollama"),
		WithAskModel("qwen3-8b"),
	)
	if err != nil {
		t.Fatalf("Ask() error: %v", err)
	}

	if resp.Text == "" {
		t.Error("Ask() returned empty text")
	}
}
