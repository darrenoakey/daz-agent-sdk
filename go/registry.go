package dazagentsdk

import (
	"strings"
	"sync"
)

// ProviderFactory is a function that creates a Provider instance.
type ProviderFactory func(cfg *Config) Provider

// providerFactories maps provider names to their factory functions.
var providerFactories = map[string]ProviderFactory{}

// providerCache holds lazily loaded provider instances.
var providerCache = map[string]Provider{}
var providerCacheMu sync.Mutex

// modelCache holds resolved ModelInfo objects keyed by "provider:model_id".
var modelCache = map[string]ModelInfo{}
var modelCacheMu sync.Mutex

// RegisterProviderFactory registers a factory function for a provider name.
// This is called by provider packages in their init() or setup functions.
func RegisterProviderFactory(name string, factory ProviderFactory) {
	providerCacheMu.Lock()
	defer providerCacheMu.Unlock()
	providerFactories[name] = factory
}

// GetProvider returns a provider by name, creating it lazily if needed.
// Returns nil if no factory is registered for the name.
func GetProvider(name string, cfg *Config) Provider {
	providerCacheMu.Lock()
	defer providerCacheMu.Unlock()

	if p, ok := providerCache[name]; ok {
		return p
	}

	factory, ok := providerFactories[name]
	if !ok {
		return nil
	}

	p := factory(cfg)
	if p == nil {
		return nil
	}
	providerCache[name] = p
	return p
}

// GetProviders returns all registered providers that can be loaded.
func GetProviders(cfg *Config) map[string]Provider {
	providerCacheMu.Lock()
	defer providerCacheMu.Unlock()

	for name, factory := range providerFactories {
		if _, ok := providerCache[name]; !ok {
			if p := factory(cfg); p != nil {
				providerCache[name] = p
			}
		}
	}

	result := make(map[string]Provider, len(providerCache))
	for k, v := range providerCache {
		result[k] = v
	}
	return result
}

// RefreshProviders clears the provider and model caches, forcing reload
// on next access.
func RefreshProviders() {
	providerCacheMu.Lock()
	defer providerCacheMu.Unlock()
	modelCacheMu.Lock()
	defer modelCacheMu.Unlock()

	for k := range providerCache {
		delete(providerCache, k)
	}
	for k := range modelCache {
		delete(modelCache, k)
	}
}

// GetModelsForTier returns the ordered list of ModelInfo objects for a tier,
// resolved from the config chain. If capability is non-nil, only models that
// include that capability are returned.
func GetModelsForTier(tier Tier, capability *Capability, cfg *Config) []ModelInfo {
	chain := GetTierChain(tier, cfg)
	var results []ModelInfo

	for _, entry := range chain {
		idx := strings.Index(entry, ":")
		if idx < 0 {
			continue
		}
		providerName := entry[:idx]
		modelID := entry[idx+1:]

		info := ResolveModel(providerName, modelID, &tier, cfg)
		if info == nil {
			continue
		}
		if capability != nil {
			found := false
			for _, c := range info.Capabilities {
				if c == *capability {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		results = append(results, *info)
	}

	return results
}

// ResolveModel looks up a ModelInfo by provider name and model ID.
// If the provider is not loaded, constructs a minimal placeholder ModelInfo
// so the tier chain is preserved for fallback ordering.
func ResolveModel(providerName, modelID string, tier *Tier, cfg *Config) *ModelInfo {
	qualified := providerName + ":" + modelID

	modelCacheMu.Lock()
	if info, ok := modelCache[qualified]; ok {
		modelCacheMu.Unlock()
		return &info
	}
	modelCacheMu.Unlock()

	// Try to get from provider - but for now, providers don't have static
	// model lists in Go (unlike Python). Build a placeholder.
	if tier == nil {
		return nil
	}

	info := ModelInfo{
		Provider:    providerName,
		ModelID:     modelID,
		DisplayName: providerName + "/" + modelID,
		Capabilities: []Capability{CapabilityText},
		Tier:        *tier,
	}

	modelCacheMu.Lock()
	modelCache[qualified] = info
	modelCacheMu.Unlock()

	return &info
}
