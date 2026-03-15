package dazagentsdk

import (
	"context"
	"testing"
	"time"
)

func TestNewConversation_Defaults(t *testing.T) {
	conv := NewConversation("test-conv")
	defer conv.Close()

	if conv.Name() != "test-conv" {
		t.Errorf("Name() = %q, want %q", conv.Name(), "test-conv")
	}
	if conv.tier != TierHigh {
		t.Errorf("tier = %q, want %q", conv.tier, TierHigh)
	}
	if len(conv.History()) != 0 {
		t.Errorf("History() len = %d, want 0", len(conv.History()))
	}
	if conv.ConversationID().String() == "" {
		t.Error("ConversationID() should not be empty")
	}
}

func TestNewConversation_WithSystem(t *testing.T) {
	conv := NewConversation("sys-test", WithSystem("You are a helpful assistant"))
	defer conv.Close()

	hist := conv.History()
	if len(hist) != 1 {
		t.Fatalf("History() len = %d, want 1", len(hist))
	}
	if hist[0].Role != "system" {
		t.Errorf("History()[0].Role = %q, want %q", hist[0].Role, "system")
	}
	if hist[0].Content != "You are a helpful assistant" {
		t.Errorf("History()[0].Content = %q, want %q", hist[0].Content, "You are a helpful assistant")
	}
}

func TestNewConversation_WithOptions(t *testing.T) {
	conv := NewConversation("opts-test",
		WithTier(TierLow),
		WithProvider("ollama"),
		WithModel("qwen3-8b"),
	)
	defer conv.Close()

	if conv.tier != TierLow {
		t.Errorf("tier = %q, want %q", conv.tier, TierLow)
	}
	if conv.providerName != "ollama" {
		t.Errorf("providerName = %q, want %q", conv.providerName, "ollama")
	}
	if conv.modelID != "qwen3-8b" {
		t.Errorf("modelID = %q, want %q", conv.modelID, "qwen3-8b")
	}
}

func TestConversation_ForkCopiesHistory(t *testing.T) {
	conv := NewConversation("parent", WithSystem("system prompt"))
	defer conv.Close()

	// Manually add a user and assistant message to history
	conv.mu.Lock()
	conv.history = append(conv.history,
		Message{Role: "user", Content: "hello"},
		Message{Role: "assistant", Content: "hi there"},
	)
	conv.mu.Unlock()

	forked := conv.Fork("child")
	defer forked.Close()

	parentHist := conv.History()
	forkedHist := forked.History()

	if len(forkedHist) != len(parentHist) {
		t.Fatalf("forked history len = %d, want %d", len(forkedHist), len(parentHist))
	}

	for i := range parentHist {
		if forkedHist[i].Role != parentHist[i].Role || forkedHist[i].Content != parentHist[i].Content {
			t.Errorf("forked history[%d] = %+v, want %+v", i, forkedHist[i], parentHist[i])
		}
	}

	if forked.Name() != "child" {
		t.Errorf("forked Name() = %q, want %q", forked.Name(), "child")
	}

	if forked.ConversationID() == conv.ConversationID() {
		t.Error("forked should have a different ConversationID")
	}
}

func TestConversation_ForkIsIndependent(t *testing.T) {
	conv := NewConversation("parent", WithSystem("system prompt"))
	defer conv.Close()

	forked := conv.Fork("child")
	defer forked.Close()

	// Add a message to the fork only
	forked.mu.Lock()
	forked.history = append(forked.history, Message{Role: "user", Content: "fork only"})
	forked.mu.Unlock()

	parentHist := conv.History()
	forkedHist := forked.History()

	if len(forkedHist) != len(parentHist)+1 {
		t.Errorf("forked history len = %d, want %d", len(forkedHist), len(parentHist)+1)
	}

	// Parent should not see the forked message
	for _, m := range parentHist {
		if m.Content == "fork only" {
			t.Error("parent should not see fork-only message")
		}
	}
}

func TestConversation_HistoryReturnsCopy(t *testing.T) {
	conv := NewConversation("copy-test")
	defer conv.Close()

	conv.mu.Lock()
	conv.history = append(conv.history, Message{Role: "user", Content: "original"})
	conv.mu.Unlock()

	hist := conv.History()
	hist[0].Content = "modified"

	// Internal history should be unchanged
	if conv.History()[0].Content != "original" {
		t.Error("History() should return a copy, not a reference")
	}
}

// TestConversation_SayOllama tests Say against a real Ollama instance.
// Skipped with -short since it requires a running Ollama server.
func TestConversation_SayOllama(t *testing.T) {
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
		// Use a dynamic import to avoid circular dependency
		// For integration tests, we rely on the provider being registered
		// externally or use a simple HTTP-based provider
		return newTestOllamaProvider(baseURL)
	})
	defer RefreshProviders()

	conv := NewConversation("ollama-test",
		WithTier(TierFreeFast),
		WithProvider("ollama"),
		WithModel("qwen3-8b"),
	)
	defer conv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	resp, err := conv.Say(ctx, "Say hello in exactly one word.")
	if err != nil {
		t.Fatalf("Say() error: %v", err)
	}

	if resp.Text == "" {
		t.Error("Say() returned empty text")
	}

	hist := conv.History()
	if len(hist) < 2 {
		t.Fatalf("History() len = %d, want >= 2 (user + assistant)", len(hist))
	}
	if hist[len(hist)-2].Role != "user" {
		t.Errorf("second-to-last message role = %q, want %q", hist[len(hist)-2].Role, "user")
	}
	if hist[len(hist)-1].Role != "assistant" {
		t.Errorf("last message role = %q, want %q", hist[len(hist)-1].Role, "assistant")
	}
}

func TestConversation_BuildChain_Override(t *testing.T) {
	conv := NewConversation("chain-test",
		WithProvider("ollama"),
		WithModel("qwen3-8b"),
	)
	defer conv.Close()

	chain := conv.buildChain(TierHigh, "ollama", "qwen3-8b")
	if len(chain) != 1 {
		t.Fatalf("buildChain() len = %d, want 1", len(chain))
	}
	if chain[0] != "ollama:qwen3-8b" {
		t.Errorf("buildChain()[0] = %q, want %q", chain[0], "ollama:qwen3-8b")
	}
}

func TestConversation_BuildChain_TierDefault(t *testing.T) {
	conv := NewConversation("chain-test")
	defer conv.Close()

	chain := conv.buildChain(TierFreeFast, "", "")
	if len(chain) == 0 {
		t.Fatal("buildChain() returned empty chain for free_fast tier")
	}
	if chain[0] != "ollama:qwen3-8b" {
		t.Errorf("buildChain()[0] = %q, want %q", chain[0], "ollama:qwen3-8b")
	}
}

func TestSplitEntry(t *testing.T) {
	tests := []struct {
		entry    string
		wantProv string
		wantMod  string
	}{
		{"ollama:qwen3-8b", "ollama", "qwen3-8b"},
		{"claude:claude-opus-4-6", "claude", "claude-opus-4-6"},
		{"invalid", "", ""},
		{"a:b:c", "a", "b:c"},
	}

	for _, tt := range tests {
		prov, mod := splitEntry(tt.entry)
		if prov != tt.wantProv || mod != tt.wantMod {
			t.Errorf("splitEntry(%q) = (%q, %q), want (%q, %q)",
				tt.entry, prov, mod, tt.wantProv, tt.wantMod)
		}
	}
}
