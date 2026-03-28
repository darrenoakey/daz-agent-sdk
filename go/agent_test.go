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

func TestWithAskTools_SetsTools(t *testing.T) {
	tools := []string{"bash", "read_file"}
	o := &askOpts{}
	WithAskTools(tools)(o)
	if len(o.tools) != 2 {
		t.Fatalf("tools len = %d, want 2", len(o.tools))
	}
	if o.tools[0] != "bash" || o.tools[1] != "read_file" {
		t.Errorf("tools = %v, want [bash read_file]", o.tools)
	}
}

func TestWithAskCwd_SetsCwd(t *testing.T) {
	o := &askOpts{}
	WithAskCwd("/tmp/workspace")(o)
	if o.cwd != "/tmp/workspace" {
		t.Errorf("cwd = %q, want /tmp/workspace", o.cwd)
	}
}

func TestWithAskMaxTurns_SetsMaxTurns(t *testing.T) {
	o := &askOpts{}
	WithAskMaxTurns(10)(o)
	if o.maxTurns != 10 {
		t.Errorf("maxTurns = %d, want 10", o.maxTurns)
	}
}

func TestAskOptsPassedToCompleteOpts(t *testing.T) {
	// Verify that tools, cwd, and maxTurns are wired into CompleteOpts by
	// inspecting a fake provider that captures the opts it receives.
	captured := &CompleteOpts{}
	fake := &capturingProvider{capturedOpts: captured}
	RegisterProviderFactory("capturing", func(cfg *Config) Provider { return fake })
	defer RefreshProviders()

	cfg := &Config{}
	cfg.applyDefaults()
	cfg.Tiers["high"] = TierConfig{Chain: []string{"capturing:test-model"}}
	cfg.Providers["capturing"] = map[string]any{}

	agent := NewAgent(cfg)
	ctx := context.Background()

	_, _ = agent.Ask(ctx, "hello",
		WithAskTools([]string{"bash", "python"}),
		WithAskCwd("/repo"),
		WithAskMaxTurns(5),
	)

	if len(captured.Tools) != 2 || captured.Tools[0] != "bash" {
		t.Errorf("CompleteOpts.Tools = %v, want [bash python]", captured.Tools)
	}
	if captured.Cwd != "/repo" {
		t.Errorf("CompleteOpts.Cwd = %q, want /repo", captured.Cwd)
	}
	if captured.MaxTurns != 5 {
		t.Errorf("CompleteOpts.MaxTurns = %d, want 5", captured.MaxTurns)
	}
}

// capturingProvider records the CompleteOpts it was called with.
type capturingProvider struct {
	capturedOpts *CompleteOpts
}

func (p *capturingProvider) Name() string { return "capturing" }
func (p *capturingProvider) Available(_ context.Context) (bool, error) { return true, nil }
func (p *capturingProvider) ListModels(_ context.Context) ([]ModelInfo, error) {
	return []ModelInfo{{Provider: "capturing", ModelID: "test-model", Tier: TierHigh}}, nil
}
func (p *capturingProvider) Complete(_ context.Context, _ []Message, _ ModelInfo, opts CompleteOpts) (*Response, error) {
	*p.capturedOpts = opts
	return &Response{Text: "ok"}, nil
}
func (p *capturingProvider) Stream(_ context.Context, _ []Message, _ ModelInfo, _ StreamOpts) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk)
	close(ch)
	return ch, nil
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
