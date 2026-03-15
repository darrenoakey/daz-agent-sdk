package dazagentsdk

import (
	"testing"
)

func TestRegisterAndGetProvider(t *testing.T) {
	// Clean state
	RefreshProviders()
	defer RefreshProviders()

	// Register a fake provider
	RegisterProviderFactory("test_provider", func(cfg *Config) Provider {
		return &fakeProvider{name: "test_provider"}
	})

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")
	p := GetProvider("test_provider", cfg)
	if p == nil {
		t.Fatal("expected provider, got nil")
	}
	if p.Name() != "test_provider" {
		t.Errorf("Name() = %q, want test_provider", p.Name())
	}
}

func TestGetProviderUnknown(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")
	p := GetProvider("nonexistent_provider", cfg)
	if p != nil {
		t.Errorf("expected nil for unknown provider, got %v", p)
	}
}

func TestGetProviderCached(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	callCount := 0
	RegisterProviderFactory("cached_test", func(cfg *Config) Provider {
		callCount++
		return &fakeProvider{name: "cached_test"}
	})

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	p1 := GetProvider("cached_test", cfg)
	p2 := GetProvider("cached_test", cfg)

	if p1 != p2 {
		t.Error("expected same provider instance from cache")
	}
	if callCount != 1 {
		t.Errorf("factory called %d times, want 1 (cached)", callCount)
	}
}

func TestGetProviders(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	RegisterProviderFactory("prov_a", func(cfg *Config) Provider {
		return &fakeProvider{name: "prov_a"}
	})
	RegisterProviderFactory("prov_b", func(cfg *Config) Provider {
		return &fakeProvider{name: "prov_b"}
	})

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")
	providers := GetProviders(cfg)

	if len(providers) < 2 {
		t.Errorf("expected at least 2 providers, got %d", len(providers))
	}
	if providers["prov_a"] == nil {
		t.Error("missing prov_a")
	}
	if providers["prov_b"] == nil {
		t.Error("missing prov_b")
	}
}

func TestRefreshProviders(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	RegisterProviderFactory("refresh_test", func(cfg *Config) Provider {
		return &fakeProvider{name: "refresh_test"}
	})

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")
	p1 := GetProvider("refresh_test", cfg)
	if p1 == nil {
		t.Fatal("expected provider before refresh")
	}

	RefreshProviders()

	// After refresh, cache is cleared but factory still exists
	p2 := GetProvider("refresh_test", cfg)
	if p2 == nil {
		t.Fatal("expected provider after refresh")
	}
	// p1 and p2 should be different instances since cache was cleared
	if p1 == p2 {
		t.Error("expected different provider instances after refresh")
	}
}

func TestGetModelsForTier(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	// free_fast tier has "ollama:qwen3-8b" in default config
	models := GetModelsForTier(TierFreeFast, nil, cfg)
	if len(models) == 0 {
		t.Fatal("expected at least one model for free_fast tier")
	}
	if models[0].Provider != "ollama" {
		t.Errorf("first model provider = %q, want ollama", models[0].Provider)
	}
	if models[0].ModelID != "qwen3-8b" {
		t.Errorf("first model id = %q, want qwen3-8b", models[0].ModelID)
	}
}

func TestGetModelsForTierWithCapabilityFilter(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	// Default placeholders have CapabilityText
	textCap := CapabilityText
	models := GetModelsForTier(TierFreeFast, &textCap, cfg)
	if len(models) == 0 {
		t.Fatal("expected models with text capability")
	}

	// Image capability should filter out text-only placeholders
	imageCap := CapabilityImage
	models = GetModelsForTier(TierFreeFast, &imageCap, cfg)
	if len(models) != 0 {
		t.Errorf("expected no models with image capability, got %d", len(models))
	}
}

func TestResolveModel(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	tier := TierFreeFast
	info := ResolveModel("ollama", "qwen3-8b", &tier, cfg)
	if info == nil {
		t.Fatal("expected model info, got nil")
	}
	if info.Provider != "ollama" {
		t.Errorf("Provider = %q, want ollama", info.Provider)
	}
	if info.ModelID != "qwen3-8b" {
		t.Errorf("ModelID = %q, want qwen3-8b", info.ModelID)
	}
	if info.Tier != TierFreeFast {
		t.Errorf("Tier = %q, want free_fast", info.Tier)
	}
}

func TestResolveModelWithoutTier(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	info := ResolveModel("unknown_provider", "unknown_model", nil, cfg)
	if info != nil {
		t.Errorf("expected nil for unknown provider without tier, got %v", info)
	}
}

func TestResolveModelCaching(t *testing.T) {
	RefreshProviders()
	defer RefreshProviders()

	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	tier := TierHigh
	info1 := ResolveModel("claude", "claude-opus-4-6", &tier, cfg)
	info2 := ResolveModel("claude", "claude-opus-4-6", &tier, cfg)

	if info1 == nil || info2 == nil {
		t.Fatal("expected non-nil model info")
	}
	if info1.QualifiedName() != info2.QualifiedName() {
		t.Error("expected same model info from cache")
	}
}
